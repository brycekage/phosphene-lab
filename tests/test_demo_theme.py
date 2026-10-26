import re

import gradio as gr

from phosphene_lab.demo import theme


def luminance(hex_color: str) -> float:
    """WCAG relative luminance of a #rrggbb color."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_colors_pass_wcag_aa_on_every_surface():
    c = theme.COLORS
    for surface in ("bg", "surface", "raised"):
        assert contrast(c["text"], c[surface]) >= 7.0, surface
        assert contrast(c["muted"], c[surface]) >= 4.5, surface
        assert contrast(c["accent"], c[surface]) >= 4.5, surface
    assert contrast(c["on_accent"], c["accent"]) >= 4.5


def test_map_colors_are_distinct_and_visible_on_black():
    c = theme.COLORS
    for name in ("truth", "estimate", "error"):
        assert contrast(c[name], "#000000") >= 4.5, name
    assert len({c["truth"], c["estimate"], c["error"]}) == 3


def test_rgb_matches_hex():
    assert theme.rgb("#56b4e9") == (86, 180, 233)


def test_theme_builds_and_css_uses_the_tokens():
    assert isinstance(theme.build_theme(), gr.themes.Base)
    css = theme.CSS
    for name, value in theme.COLORS.items():
        assert f"--ph-{name.replace('_', '-')}: {value}" in css
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css
    # Every color in the stylesheet comes from a token, not a stray literal.
    body = css.split("}", 1)[1]
    assert not re.findall(r"#[0-9a-fA-F]{6}\b", body)


def test_head_forces_dark_mode():
    assert "dark" in theme.HEAD
