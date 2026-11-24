
from pathlib import Path
from playwright.sync_api import sync_playwright

output = Path("S5_figures")
output.mkdir(exist_ok=True)

pages = {
    "S5_A_examples": (
        "https://mdsa-tools.readthedocs.io/en/latest/examples.html"
    ),
    "S5_B_data_generation": (
        "https://mdsa-tools.readthedocs.io/en/latest/"
        "examples/data_generation_workflow.html"
    ),
}

with sync_playwright() as p:
    browser = p.chromium.launch()

    context = browser.new_context(
        viewport={"width": 1050, "height": 850},
        device_scale_factor=3,
    )

    for name, url in pages.items():
        page = context.new_page()
        page.goto(url, wait_until="networkidle")

        # Remove navigation without changing the documentation.
        page.add_style_tag(content="""
            .wy-nav-side,
            .wy-nav-top,
            .rst-versions,
            .wy-breadcrumbs,
            .rst-footer-buttons,
            footer {
                display: none !important;
            }

            .wy-nav-content-wrap {
                margin-left: 0 !important;
            }

            .wy-nav-content {
                max-width: none !important;
                padding: 24px 36px !important;
            }
        """)

        page.screenshot(
            path=str(output / f"{name}.png"),
            full_page=False,
            animations="disabled",
        )
        page.close()

    browser.close()
