"""Capture a hydrated Streamlit screenshot through Chrome DevTools Protocol.

This script intentionally avoids Playwright/Selenium so visual smoke can run in
the current repo environment without adding dependencies.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)


class CdpClient:
    def __init__(self, websocket_url: str) -> None:
        if not websocket_url.startswith("ws://"):
            raise ValueError(f"Unsupported websocket URL: {websocket_url}")
        host_port, path = websocket_url[5:].split("/", 1)
        host, port_raw = host_port.split(":")
        self._socket = socket.create_connection((host, int(port_raw)), timeout=8)
        self._next_id = 0

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._socket.sendall(request.encode("ascii"))
        response = self._socket.recv(4096)
        if b" 101 " not in response:
            raise RuntimeError(f"CDP websocket handshake failed: {response[:160]!r}")

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass

    def call(self, method: str, params: dict[str, Any] | None = None, *, timeout_sec: float = 20) -> dict[str, Any]:
        message_id = self._send(method, params or {})
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            message = self._recv(timeout_sec=max(0.2, deadline - time.time()))
            if not message:
                continue
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"CDP {method} failed: {message['error']}")
                return dict(message.get("result") or {})
        raise TimeoutError(f"Timed out waiting for CDP response: {method}")

    def _send(self, method: str, params: dict[str, Any]) -> int:
        self._next_id += 1
        payload = json.dumps(
            {"id": self._next_id, "method": method, "params": params},
            separators=(",", ":"),
        ).encode("utf-8")
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = os.urandom(4)
        header.extend(mask)
        masked_payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._socket.sendall(header + masked_payload)
        return self._next_id

    def _recv(self, *, timeout_sec: float) -> dict[str, Any] | None:
        self._socket.settimeout(timeout_sec)
        first = self._socket.recv(2)
        if not first:
            raise RuntimeError("CDP websocket closed")
        byte_1, byte_2 = first
        opcode = byte_1 & 0x0F
        length = byte_2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if byte_2 & 0x80 else b""
        payload = bytearray(self._recv_exact(length))
        if opcode == 8:
            raise RuntimeError("CDP websocket closed by remote")
        if opcode in (9, 10):
            return None
        if mask:
            payload = bytearray(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return json.loads(payload.decode("utf-8"))

    def _recv_exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                raise RuntimeError("CDP websocket closed mid-frame")
            data.extend(chunk)
        return bytes(data)


def _request_json(url: str, *, timeout_sec: float = 2) -> Any:
    with urllib.request.urlopen(url, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_text(url: str, *, timeout_sec: float = 2) -> str:
    with urllib.request.urlopen(url, timeout=timeout_sec) as response:
        return response.read().decode("utf-8")


def _wait_for_health(port: int, *, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            content = _request_text(f"http://127.0.0.1:{port}/_stcore/health", timeout_sec=2)
            if content.strip().lower() == "ok":
                return
        except Exception as exc:  # noqa: BLE001 - diagnostic retry loop
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"Streamlit health did not become ok on port {port}: {last_error}")


def _find_chrome(explicit_path: str | None) -> str:
    candidates = (explicit_path,) if explicit_path else DEFAULT_CHROME_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Could not find Chrome or Edge. Pass --chrome-path explicitly.")


def _find_debug_page(debug_port: int, target_url: str, *, timeout_sec: float) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_pages: list[dict[str, Any]] = []
    while time.time() < deadline:
        try:
            pages = list(_request_json(f"http://127.0.0.1:{debug_port}/json", timeout_sec=2))
            last_pages = pages
            page_targets = [page for page in pages if page.get("type") == "page"]
            matching = [page for page in page_targets if str(page.get("url", "")).startswith(target_url)]
            if matching:
                return dict(matching[0])
            if page_targets:
                return dict(page_targets[0])
        except Exception:
            pass
        time.sleep(0.2)
    urls = [str(page.get("url", "")) for page in last_pages]
    raise TimeoutError(f"No Chrome page target found for {target_url}. Seen targets: {urls}")


def _wait_for_streamlit_ready(client: CdpClient, *, timeout_sec: float, min_text_len: int) -> dict[str, Any]:
    expression = r"""
(() => {
  const app = document.querySelector('[data-testid="stApp"]');
  const text = document.body?.innerText || '';
  const skeleton = document.querySelector('[data-testid="stAppSkeleton"]');
  return {
    state: app?.getAttribute('data-test-script-state') || '',
    connection: app?.getAttribute('data-test-connection-state') || '',
    textLength: text.length,
    textPreview: text.slice(0, 200),
    hasSkeleton: Boolean(skeleton),
    hasPageNotFound: text.includes('Page not found'),
    mainHtmlLength: document.querySelector('[data-testid="stMainBlockContainer"]')?.innerHTML?.length || 0
  };
})()
"""
    deadline = time.time() + timeout_sec
    last_status: dict[str, Any] = {}
    while time.time() < deadline:
        result = client.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
            timeout_sec=10,
        )
        last_status = dict(result.get("result", {}).get("value") or {})
        script_ready = str(last_status.get("state", "")) not in {"", "initial", "running"}
        connected = str(last_status.get("connection", "")) == "CONNECTED"
        enough_text = int(last_status.get("textLength") or 0) >= min_text_len
        no_skeleton = not bool(last_status.get("hasSkeleton"))
        if script_ready and connected and enough_text and no_skeleton:
            return last_status
        time.sleep(0.5)
    raise TimeoutError(f"Streamlit did not hydrate before timeout. Last status: {last_status}")


def _terminate(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", required=True)
    parser.add_argument("--app", default="app.py")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--debug-port", type=int, default=9222)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--min-text-len", type=int, default=300)
    parser.add_argument("--chrome-path", default=None)
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--allow-page-not-found", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chrome_path = _find_chrome(args.chrome_path)
    target_url = str(args.url).rstrip("/")

    streamlit_process: subprocess.Popen[Any] | None = None
    chrome_process: subprocess.Popen[Any] | None = None
    client: CdpClient | None = None
    chrome_profile = Path(tempfile.mkdtemp(prefix="streamlit-cdp-profile-"))

    try:
        if not args.reuse_server:
            streamlit_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    args.app,
                    "--server.port",
                    str(args.port),
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                ],
                cwd=Path.cwd(),
            )
        _wait_for_health(args.port, timeout_sec=args.timeout)

        chrome_process = subprocess.Popen(
            [
                chrome_path,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                f"--remote-debugging-port={args.debug_port}",
                f"--window-size={args.width},{args.height}",
                f"--user-data-dir={chrome_profile}",
                target_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        page = _find_debug_page(args.debug_port, target_url, timeout_sec=args.timeout)
        client = CdpClient(str(page["webSocketDebuggerUrl"]))
        client.call("Page.enable")
        client.call("Runtime.enable")
        client.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": args.width,
                "height": args.height,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        status = _wait_for_streamlit_ready(client, timeout_sec=args.timeout, min_text_len=args.min_text_len)
        if status.get("hasPageNotFound") and not args.allow_page_not_found:
            raise RuntimeError(
                "Streamlit rendered a Page not found modal. Use the root URL or pass --allow-page-not-found."
            )
        time.sleep(1.0)
        screenshot = client.call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False},
            timeout_sec=20,
        )
        output.write_bytes(base64.b64decode(str(screenshot["data"])))
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output": str(output),
                    "bytes": output.stat().st_size,
                    "url": target_url,
                    "streamlit": status,
                    "python": sys.executable,
                },
                ensure_ascii=True,
            )
        )
        return 0
    except (TimeoutError, RuntimeError, FileNotFoundError, urllib.error.URLError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "url": target_url}, ensure_ascii=True), file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
        _terminate(chrome_process)
        _terminate(streamlit_process)


if __name__ == "__main__":
    raise SystemExit(main())
