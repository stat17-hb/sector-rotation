#!/usr/bin/env python
"""Lightweight secret pattern scanner for pre-commit."""
from __future__ import annotations

import pathlib
import re
import sys


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-pat", re.compile(r"\bghp_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{80,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("private-key-header", re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA|PRIVATE) PRIVATE KEY-----")),
]

ECOS_ASSIGN_RE = re.compile(r'ECOS_API_KEY\s*=\s*"([^"]+)"')
KOSIS_ASSIGN_RE = re.compile(r'KOSIS_API_KEY\s*=\s*"([^"]+)"')


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    placeholder_tokens = (
        "your_",
        "here",
        "example",
        "dummy",
        "sample",
        "test",
        "여기에",
        "<redacted",
    )
    return any(token in lowered for token in placeholder_tokens)


def _read_text(path: pathlib.Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    return raw.decode("utf-8", errors="ignore")


def _line_of_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _scan_file(path: pathlib.Path) -> list[str]:
    findings: list[str] = []
    text = _read_text(path)
    if text is None:
        return findings

    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            line_no = _line_of_offset(text, match.start())
            findings.append(f"{path}:{line_no}: [{label}] potential secret pattern detected")

    for env_label, env_pattern in (("ecos-api-key", ECOS_ASSIGN_RE), ("kosis-api-key", KOSIS_ASSIGN_RE)):
        for match in env_pattern.finditer(text):
            value = match.group(1)
            if _is_placeholder(value):
                continue
            line_no = _line_of_offset(text, match.start())
            findings.append(f"{path}:{line_no}: [{env_label}] hardcoded API key assignment detected")

    return findings


def main(argv: list[str]) -> int:
    paths = [pathlib.Path(p) for p in argv if p]
    findings: list[str] = []
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        findings.extend(_scan_file(path))

    if findings:
        print("Secret scan failed. Remove or redact detected values.")
        for finding in findings:
            print(finding)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
