#!/usr/bin/env python3
"""Frontend smoketest.

Verifies that the Nginx-served frontend is not only serving HTML but
that the JavaScript bundle can execute in a headless browser when
Playwright is available.

Behaviour:

- If `playwright` is installed in the environment (`pip install playwright`
  and `playwright install chromium`), this script will launch a headless
  Chromium instance, load the portal page, and assert that some expected
  text is present in the rendered body.

- If `playwright` is NOT available, it falls back to a simple HTTP GET
  check and ensures that HTML is served. This keeps the smoketest usable
  in minimal environments while still allowing a deeper E2E check where
  supported.

The target URL can be overridden via FRONTEND_URL env var.
"""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8080/")


def _check_html_only() -> None:
    logger.info("Running frontend HTML smoke check against %s", FRONTEND_URL)
    try:
        resp = requests.get(FRONTEND_URL, timeout=10)
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Failed to reach frontend at %s: %s", FRONTEND_URL, exc)
        raise SystemExit(1)
    if resp.status_code != 200:
        logger.error("Frontend returned non-200 status: %s", resp.status_code)
        raise SystemExit(1)
    text = resp.text.lower()
    if "<html" not in text or "<body" not in text:
        logger.error("Frontend response does not look like HTML")
        raise SystemExit(1)
    logger.info("Frontend HTML check passed (status=%s, length=%s)", resp.status_code, len(resp.text))


def _check_with_playwright() -> None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        logger.info("playwright not installed; falling back to HTML-only check")
        _check_html_only()
        return

    logger.info("Running frontend headless browser smoke check against %s", FRONTEND_URL)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(FRONTEND_URL, wait_until="networkidle")
        time.sleep(1.0)
        body_text = (page.text_content("body") or "").lower()
        browser.close()

    expected_fragments = [
        "image tagger",
        "tagger workbench",
        "admin cockpit",
        "research explorer",
    ]
    if not any(fragment in body_text for fragment in expected_fragments):
        logger.error("Headless browser did not see expected portal text in body.")
        logger.debug("Body text was: %s", body_text[:500])
        raise SystemExit(1)

    logger.info("Frontend headless browser smoke check passed.")


def main() -> None:
    _check_with_playwright()


if __name__ == "__main__":
    main()
