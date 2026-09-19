"""Grava uma demonstração real da interface usando o fixture determinístico."""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "demo"
URL = "http://127.0.0.1:7861"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            record_video_dir=str(OUTPUT),
            record_video_size={"width": 1440, "height": 900},
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(URL, wait_until="networkidle")
        page.get_by_role("button", name="Analisar repositório").wait_for(
            timeout=30_000
        )
        page.wait_for_timeout(2_000)
        page.screenshot(path="/tmp/repo-agent-chat-demo-ready.png")

        page.get_by_role("textbox").fill("tests/fixtures/simple_api")
        page.get_by_role("button", name="Analisar repositório").click()
        page.get_by_text("carregado", exact=False).wait_for(timeout=90_000)
        page.wait_for_timeout(2_000)
        page.screenshot(path="/tmp/repo-agent-chat-demo-loaded.png")

        page.locator("#question-architecture").click()
        page.get_by_text("Análise concluída", exact=False).wait_for(timeout=120_000)
        page.wait_for_timeout(3_000)
        page.locator("details summary").click()
        page.wait_for_timeout(3_000)
        page.screenshot(path="/tmp/repo-agent-chat-demo-trace.png")

        video = page.video
        context.close()
        if video is not None:
            video.save_as(str(OUTPUT / "repo-agent-chat-demo.webm"))
        browser.close()


if __name__ == "__main__":
    main()
