"""Record assets/demo.gif from the running demo with headless Chromium.

    uv run python demo/app.py &                                   # the app must be up
    uv run --with playwright python -m playwright install chromium   # once
    uv run --with playwright python scripts/record_demo.py

The script walks through the Recommend tab and then the Code explorer tab, takes a screenshot
after each step and writes them as one GIF.
"""

import io
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
OUT = Path("assets/demo.gif")
SIZE = {"width": 1100, "height": 720}

frames: list[tuple[Image.Image, int]] = []


def shot(page, ms: int) -> None:
    page.wait_for_timeout(600)
    frames.append((Image.open(io.BytesIO(page.screenshot())).convert("RGB"), ms))


def pick(page, label: str, text: str) -> None:
    """Type into a Gradio dropdown and take the first match."""
    box = page.get_by_label(label)
    box.click()
    box.fill("")
    box.press_sequentially(text, delay=60)
    page.wait_for_timeout(700)
    box.press("Enter")
    page.wait_for_timeout(900)


def search_and_add(page, query: str, title: str, pause: int = 1200) -> None:
    box = page.get_by_label("Search products")
    box.fill("")
    box.press_sequentially(query, delay=60)
    page.wait_for_timeout(1200)
    page.get_by_label("Pick a match to add it to the history").click()
    page.wait_for_timeout(500)
    shot(page, 1100)
    page.get_by_role("option", name=title, exact=True).first.click()
    page.wait_for_timeout(1500)
    shot(page, pause)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=SIZE, color_scheme="light")
        page.goto(URL)
        page.get_by_label("Search products").wait_for()
        shot(page, 1500)

        search_and_add(page, "zelda", "The Legend of Zelda: Ocarina of Time")
        search_and_add(page, "mario kart", "Mario Kart 64")
        search_and_add(page, "smash", "Super Smash Bros.", pause=1800)

        page.get_by_role("button", name="Recommend").click()
        page.locator(".cards .code").first.wait_for(timeout=120000)
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Recommend").scroll_into_view_if_needed()
        page.mouse.wheel(0, 330)
        shot(page, 4500)

        page.get_by_label("Text-only IDs (TIGER)").check()
        page.get_by_text("Top 10 of 50 beams, Text-only IDs").wait_for(timeout=120000)
        page.wait_for_timeout(2500)
        shot(page, 4500)

        page.mouse.wheel(0, -2000)
        page.get_by_role("tab", name="Code explorer").click()
        page.locator(".cards.small .card >> visible=true").first.wait_for(timeout=60000)
        page.wait_for_timeout(2000)
        shot(page, 2500)
        for first, second in (("142", "234"), ("84", "246")):
            pick(page, "First token", first)
            pick(page, "Second token", second)
            page.wait_for_timeout(2500)
            shot(page, 4000)
        browser.close()

    OUT.parent.mkdir(exist_ok=True)
    images = [f.resize((880, 576), Image.LANCZOS).quantize(colors=128, method=Image.MEDIANCUT)
              for f, _ in frames]
    images[0].save(OUT, save_all=True, append_images=images[1:], loop=0, optimize=True,
                   duration=[ms for _, ms in frames])
    seconds = sum(ms for _, ms in frames) / 1000
    print(f"{OUT}: {len(frames)} frames, {seconds:.0f} s, {OUT.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
