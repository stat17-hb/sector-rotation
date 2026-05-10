from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = "http://localhost:8510"
OUT_DIR = Path(".omx/artifacts/visual-eval")

VIEWPORTS = {
    "desktop": {"width": 1440, "height": 1100},
    "mobile": {"width": 390, "height": 900},
}

ROUTES = {
    "overview": "/",
    "signals": "/kr-signals",
    "research": "/kr-research",
    "flow": "/kr-flow",
}


def collect_layout_issues(page):
    return page.evaluate(
        """
        () => {
          const vw = window.innerWidth;
          const vh = window.innerHeight;
          const nodes = Array.from(document.querySelectorAll('body *'));
          const visible = nodes
            .map((el) => {
              const r = el.getBoundingClientRect();
              const s = window.getComputedStyle(el);
              const text = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 90);
              return {
                tag: el.tagName.toLowerCase(),
                cls: String(el.className || '').slice(0, 100),
                testid: el.getAttribute('data-testid') || '',
                role: el.getAttribute('role') || '',
                text,
                x: r.x, y: r.y, w: r.width, h: r.height,
                overflowX: el.scrollWidth > Math.ceil(el.clientWidth) + 2,
                overflowY: el.scrollHeight > Math.ceil(el.clientHeight) + 2,
                display: s.display,
                visibility: s.visibility,
                opacity: Number(s.opacity || '1')
              };
            })
            .filter((n) => n.w > 1 && n.h > 1 && n.display !== 'none' && n.visibility !== 'hidden' && n.opacity > 0);

          const offscreen = visible.filter((n) => n.x < -2 || n.x + n.w > vw + 2).slice(0, 25);
          const overflowing = visible.filter((n) => n.overflowX || n.overflowY).slice(0, 25);
          const tinyTextBlocks = visible
            .filter((n) => n.text && n.h < 9 && n.w > 20)
            .slice(0, 25);
          return {
            url: window.location.href,
            viewport: { width: vw, height: vh },
            bodyWidth: document.body.scrollWidth,
            documentWidth: document.documentElement.scrollWidth,
            horizontalOverflow: document.documentElement.scrollWidth > vw + 2,
            bodyTextSample: document.body.innerText.slice(0, 2000),
            hasPositionModeWarning: document.body.innerText.includes('The widget with key "position_mode"'),
            hasPageNotFound: document.body.innerText.includes('Page not found'),
            visibleCount: visible.length,
            offscreen,
            overflowing,
            tinyTextBlocks
          };
        }
        """
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for route_name, route in ROUTES.items():
            for viewport_name, viewport in VIEWPORTS.items():
                page = browser.new_page(viewport=viewport, device_scale_factor=1)
                url = f"{BASE_URL}{route}"
                page.goto(url, wait_until="networkidle", timeout=90000)
                page.wait_for_timeout(5000)
                key = f"{route_name}-{viewport_name}"
                screenshot_path = OUT_DIR / f"{key}.png"
                page.screenshot(path=screenshot_path, full_page=True)
                results[key] = {
                    "route": route,
                    "screenshot": str(screenshot_path),
                    "layout": collect_layout_issues(page),
                }
                page.close()
        browser.close()
    (OUT_DIR / "layout-report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(OUT_DIR), "count": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
