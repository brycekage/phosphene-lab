"""One look for the demo app and the Phase 6 pilot: a Gradio theme, a stylesheet, a head tag.

The direction is a calm, dark lab instrument: phosphenes are light on black, so the pages
are dark too, with one warm "phosphene amber" accent for actions. The map colors (truth,
estimate, error) come from the Okabe-Ito palette, which stays distinct under the common
forms of color blindness. Text colors pass WCAG AA on every surface (tested).
"""

from __future__ import annotations

import html
import inspect
import os

import gradio as gr

COLORS = {
    "bg": "#0b0d10",
    "surface": "#14171c",
    "raised": "#1b1f26",
    "border": "#2a3039",
    "text": "#ece9e2",
    "muted": "#a3a9b2",
    "accent": "#f5b544",
    "accent_hover": "#ffc965",
    "on_accent": "#1a1203",
    "truth": "#b8bcc4",  # gray rings: where the phosphenes really are
    "estimate": "#56b4e9",  # Okabe-Ito sky blue
    "error": "#d55e00",  # Okabe-Ito vermilion
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def build_theme() -> gr.themes.Base:
    c = COLORS
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )
    values = {
        "body_background_fill": c["bg"],
        "body_text_color": c["text"],
        "body_text_color_subdued": c["muted"],
        "background_fill_primary": c["surface"],
        "background_fill_secondary": c["bg"],
        "border_color_primary": c["border"],
        "border_color_accent": c["accent"],
        "color_accent": c["accent"],
        "color_accent_soft": c["raised"],
        "block_background_fill": c["surface"],
        "block_border_color": c["border"],
        "block_label_background_fill": c["surface"],
        "block_label_text_color": c["muted"],
        "block_label_border_color": c["border"],
        "block_title_text_color": c["text"],
        "block_info_text_color": c["muted"],
        "block_shadow": "none",
        "panel_background_fill": c["surface"],
        "panel_border_color": c["border"],
        "input_background_fill": c["raised"],
        "input_border_color": c["border"],
        "input_border_color_focus": c["accent"],
        "button_primary_background_fill": c["accent"],
        "button_primary_background_fill_hover": c["accent_hover"],
        "button_primary_text_color": c["on_accent"],
        "button_primary_border_color": c["accent"],
        "button_secondary_background_fill": c["raised"],
        "button_secondary_background_fill_hover": c["border"],
        "button_secondary_text_color": c["text"],
        "button_secondary_border_color": c["border"],
        "checkbox_label_background_fill": c["raised"],
        "checkbox_label_background_fill_hover": c["border"],
        "checkbox_label_background_fill_selected": c["raised"],
        "checkbox_label_border_color": c["border"],
        "checkbox_label_border_color_selected": c["accent"],
        "checkbox_label_text_color": c["text"],
        "checkbox_label_text_color_selected": c["text"],
        "checkbox_background_color": c["bg"],
        "checkbox_background_color_selected": c["accent"],
        "checkbox_border_color_selected": c["accent"],
        "slider_color": c["accent"],
        "link_text_color": c["accent"],
        "link_text_color_hover": c["accent_hover"],
        "loader_color": c["accent"],
    }
    # Same values in light and dark mode, and only the variables this Gradio version knows.
    known = set(inspect.signature(theme.set).parameters)
    both = {k: v for name, v in values.items() for k in (name, f"{name}_dark") if k in known}
    return theme.set(**both)


def _tokens() -> str:
    return "\n".join(f"  --ph-{k.replace('_', '-')}: {v};" for k, v in COLORS.items())


CSS = (
    ":root {\n"
    + _tokens()
    + "\n}\n"
    + """
/* Full width up to the cap; otherwise Gradio shrinks the page to its content and tabs jump */
.gradio-container {
  width: 100% !important; max-width: 1240px !important; margin: 0 auto !important;
}
footer, footer a { color: var(--ph-muted) !important; }

/* Header */
.ph-header { padding: 4px 0 0; }
.ph-eyebrow {
  font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--ph-accent); margin: 0 0 4px;
}
.ph-header h1 { font-size: 26px; line-height: 1.2; margin: 0 0 6px; color: var(--ph-text); }
.ph-lead { color: var(--ph-muted); font-size: 15px; max-width: 80ch; margin: 0 0 8px; }
.ph-pill {
  display: inline-block; font-size: 12px; padding: 2px 10px; border-radius: 999px;
  margin-left: 10px; vertical-align: 2px; letter-spacing: normal; text-transform: none;
  border: 1px solid var(--ph-border); color: var(--ph-muted); background: var(--ph-surface);
}

/* Instrument displays: black field, thin frame, nothing letterboxed in gray */
.ph-display, .ph-display .image-container, .ph-display .image-frame,
.ph-display button, .ph-display img { background: black !important; }
.ph-display { border: 1px solid var(--ph-border) !important; }
.ph-display img { object-fit: contain; }
.ph-clickable img { cursor: crosshair; }
@media (min-width: 900px) {
  .ph-sticky { position: sticky; top: 12px; align-self: flex-start; }
}

/* Readouts and captions */
.ph-readout { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0; }
.ph-chip {
  font-family: var(--font-mono); font-size: 13px; padding: 6px 10px; border-radius: 4px;
  background: var(--ph-raised); border: 1px solid var(--ph-border); color: var(--ph-text);
}
.ph-chip b { color: var(--ph-accent); font-weight: 600; }
.ph-note { color: var(--ph-muted); font-size: 14px; }
.ph-callout {
  border-left: 3px solid var(--ph-accent); background: var(--ph-raised);
  padding: 10px 14px; border-radius: 4px; color: var(--ph-text);
}
.ph-steps { margin: 0; padding-left: 20px; color: var(--ph-text); }
.ph-steps li { margin: 2px 0; }
.ph-legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: var(--ph-muted); }
.ph-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-right: 6px; vertical-align: middle; }
.ph-swatch.truth { border: 2px solid var(--ph-truth); width: 8px; height: 8px; }
.ph-swatch.estimate { background: var(--ph-estimate); }
.ph-swatch.error { background: var(--ph-error); border-radius: 1px; height: 3px; width: 14px; }
.ph-section-title {
  font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--ph-muted); margin: 6px 0 2px;
}

/* Pilot: big, calm answer targets */
.ph-answer-grid button { min-height: 48px; font-size: 18px; font-weight: 600; }
.ph-letter-grid {
  display: grid !important; grid-template-columns: repeat(13, minmax(0, 1fr)); gap: 8px;
}
.ph-letter-grid > * { min-width: 0 !important; width: auto !important; }
@media (max-width: 600px) {
  .ph-letter-grid { grid-template-columns: repeat(7, minmax(0, 1fr)); }
}
.ph-progress { height: 6px; background: var(--ph-raised); border-radius: 3px; overflow: hidden; }
.ph-progress > div { height: 100%; background: var(--ph-accent); }
.ph-feedback-right { border-left-color: var(--ph-estimate); }
.ph-feedback-wrong { border-left-color: var(--ph-error); }
.ph-kbd {
  font-family: var(--font-mono); font-size: 12px; padding: 1px 6px; border-radius: 3px;
  border: 1px solid var(--ph-border); background: var(--ph-raised); color: var(--ph-text);
}
.ph-results {
  width: 100%; border-collapse: collapse; margin: 4px 0 18px; border: none !important;
}
.ph-results thead, .ph-results tbody, .ph-results tr { border: none !important; }
.ph-results td, .ph-results th {
  padding: 10px 12px; text-align: left; border: none !important;
  border-bottom: 1px solid var(--ph-border) !important; background: transparent !important;
}
.ph-results th { color: var(--ph-muted); font-weight: 500; font-size: 13px; }
.ph-results td:last-child, .ph-results th:last-child { white-space: nowrap; text-align: right; }

/* Accessibility */
:focus-visible { outline: 2px solid var(--ph-accent) !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
@media (max-width: 720px) {
  .ph-header h1 { font-size: 24px; }
  .ph-lead { font-size: 15px; }
}
"""
)


def launch_options(description: str, js: str | None = None, max_width_px: int = 1240) -> dict:
    """``Blocks.launch`` settings shared by both apps: this look, no API/settings/history links.

    Hugging Face Spaces sets GRADIO_SERVER_NAME=0.0.0.0; locally the apps stay on loopback.
    A step-by-step page (the pilot) reads best in a narrower column than a tool (the demo).
    """
    width = f".gradio-container {{ max-width: {max_width_px}px !important; }}\n"
    options = {
        "server_name": os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        "theme": build_theme(), "css": CSS + width,
        "head": HEAD + f'<meta name="description" content="{html.escape(description)}">\n',
        "footer_links": ["gradio"], "run_history": False,
    }  # fmt: skip
    options["js"] = f"() => {{ ({NAME_IMAGE_BUTTONS_JS})(); {f'({js})();' if js else ''} }}"
    return options


# Gradio wraps each output image in an unnamed <button>; screen readers would announce only
# "button". Name each one after its panel's caption, again whenever the page changes.
NAME_IMAGE_BUTTONS_JS = """
() => {
  const name = () => document.querySelectorAll(".ph-display").forEach((block) => {
    const caption = block.querySelector("label, [data-testid='block-label']");
    const label = (caption && caption.textContent.trim()) || "Image";
    block.querySelectorAll("button:not([aria-label])").forEach((b) => {
      if (!b.textContent.trim()) b.setAttribute("aria-label", label);
    });
  });
  name();
  new MutationObserver(name).observe(document.body, { childList: true, subtree: true });
}
"""


# Dark in every browser, before first paint, so the pages never flash white.
HEAD = """
<meta name="color-scheme" content="dark">
<script>
  document.documentElement.classList.add("dark");
  document.addEventListener("DOMContentLoaded", () => document.body.classList.add("dark"));
</script>
"""
