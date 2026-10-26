"""Phase 5 interactive demo: see what an implant user would see, and play the patient.

Two tabs:

* **Viewer:** a photo or webcam snapshot, what the chosen encoder draws from it, and the
  phosphene percept a virtual patient would see, with sliders for electrode count, dead
  electrodes, adaptation, encoder, and priority mode.
* **Calibration game:** you are the patient. A flash appears; click where you saw it (or
  "saw nothing"). Active calibration picks every question and updates its map of the implant
  live, and a counter compares it with a full exhaustive pass.

Usage:
    python app/app.py [--config configs/demo.yaml] [--quick]

``--quick`` is a headless smoke run (no browser, no network weights), well under 2 minutes.
"""

from __future__ import annotations

import argparse
import time

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from skimage import data  # noqa: E402

from phosphene_lab.calibration.common import SECONDS_PER_QUERY  # noqa: E402
from phosphene_lab.config import load_config  # noqa: E402
from phosphene_lab.demo import (  # noqa: E402
    ELECTRODE_COUNTS,
    CalibrationGame,
    Viewer,
    ViewSettings,
)
from phosphene_lab.demo.theme import COLORS, launch_options  # noqa: E402
from phosphene_lab.simulator.patient import PatientParams  # noqa: E402

ENCODER_LABELS = {
    "E0": "E0 · Brightness",
    "E1": "E1 · Edges",
    "E2": "E2 · Scene (depth + objects)",
    "E3": "E3 · Learned",
}
MODE_LABELS = {"navigation": "Walking", "people": "People", "reading": "Reading"}

HEADER = """
<div class="ph-header">
  <p class="ph-eyebrow">Phosphene Lab · cortical visual prosthesis simulator
  <span class="ph-pill">All patients here are simulated</span></p>
  <h1>See what a visual implant user might see</h1>
  <p class="ph-lead">An implant in the visual cortex draws the world as a sparse grid of
  light dots called <em>phosphenes</em>. Choose a scene and an encoding strategy to see the
  result, or play the patient and help calibrate an implant.</p>
</div>
"""

ENCODER_HELP = """
<ul class="ph-steps">
  <li><b>E0 Brightness:</b> lights each dot by how bright the camera image is there.</li>
  <li><b>E1 Edges:</b> lights the outlines of things instead of their surfaces.</li>
  <li><b>E2 Scene:</b> two neural networks estimate depth and find objects, and only what
  matters for the chosen task is drawn (for walking: nearby obstacles).</li>
  <li><b>E3 Learned:</b> a small network trained through the simulator to make percepts
  that a reader can classify, while using little current.</li>
</ul>
<p class="ph-note">The left panel is what the encoder asks for; the right panel is what the
patient perceives once real-world limits apply: few dots, misplaced dots, dead electrodes,
and fading during steady viewing.</p>
"""

GAME_STEPS = """
<ol class="ph-steps">
  <li>Look at the small cross in the middle of the left panel.</li>
  <li>When a dot of light flashes, <b>click where you saw it</b>. Faint flashes are on
  purpose: if you see nothing, press <b>I saw nothing</b>.</li>
  <li>Each answer updates the system's map of the implant (right). Watch the error lines
  shrink.</li>
</ol>
"""

LEGEND = """
<div class="ph-legend">
  <span><span class="ph-swatch truth"></span>True position</span>
  <span><span class="ph-swatch estimate"></span>System's estimate</span>
  <span><span class="ph-swatch error"></span>Error</span>
</div>
"""


def sign_image(text: str = "EXIT") -> np.ndarray:
    """A plain sign for the reading mode: dark letters on a light board."""
    img = Image.new("RGB", (480, 320), (225, 225, 215))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 150)
    except OSError:
        font = ImageFont.load_default(size=150)
    box = draw.textbbox((0, 0), text, font=font)
    xy = ((480 - (box[2] - box[0])) / 2 - box[0], (320 - (box[3] - box[1])) / 2 - box[1])
    draw.text(xy, text, fill=(20, 20, 20), font=font)
    return np.asarray(img)


# Public-domain / CC0 photos bundled with scikit-image, plus a generated sign.
SAMPLES = {
    "Person": data.astronaut,
    "Coffee cup": data.coffee,
    "Cat": data.chelsea,
    "EXIT sign": sign_image,
}
# The priority mode that suits each sample, so choosing one shows it at its best.
SAMPLE_MODE = {"Person": "people", "Coffee cup": "navigation", "Cat": "navigation",
               "EXIT sign": "reading"}  # fmt: skip


def to_display(field: np.ndarray, px: int) -> np.ndarray:
    """(R, R) in [0, 1] or (R, R, 3) uint8 -> (px, px, 3) uint8, smoothly enlarged."""
    if field.ndim == 2:
        field = np.repeat((field.clip(0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    return np.asarray(Image.fromarray(field).resize((px, px), Image.BILINEAR))


def with_fixation(img: np.ndarray) -> np.ndarray:
    """Add a dim cross at the center of the visual field, where the patient looks."""
    img = img.copy()
    c, arm = img.shape[0] // 2, img.shape[0] // 40
    img[c, c - arm : c + arm + 1] = np.maximum(img[c, c - arm : c + arm + 1], 70)
    img[c - arm : c + arm + 1, c] = np.maximum(img[c - arm : c + arm + 1, c], 70)
    return img


def chips(*items: tuple[str, str]) -> str:
    return '<div class="ph-readout">' + "".join(
        f'<span class="ph-chip">{label} <b>{value}</b></span>' for label, value in items
    ) + "</div>"  # fmt: skip


def error_plot(game: CalibrationGame):
    c = COLORS
    fig, ax = plt.subplots(figsize=(6, 2.4))
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    n = len(game.history)
    ax.plot(range(n), game.history, color=c["estimate"], lw=2, label="Your answers")
    ax.axhline(
        game.target_error_deg, color=c["truth"], ls="--", lw=1,
        label=f"Full exhaustive pass ({game.exhaustive_queries} questions)",
    )  # fmt: skip
    ax.set_xlim(0, max(10, n))
    ax.set_ylim(0, max(game.history + [game.target_error_deg]) * 1.15)
    ax.set_xlabel("questions answered", color=c["muted"])
    ax.set_ylabel("map error (deg)", color=c["muted"])
    ax.tick_params(colors=c["muted"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["border"])
    ax.set_title("Map error as you answer", color=c["text"], fontsize=10, loc="left")
    legend = ax.legend(fontsize=8, frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(c["text"])
    fig.tight_layout()
    plt.close(fig)
    return fig


def game_status(game: CalibrationGame) -> str:
    s = game.status()
    electrode, current = game.pending
    html = chips(
        ("Questions answered", str(s["queries"])),
        ("Map error", f"{s['location_error_deg']:.2f}°"),
        ("Exhaustive pass", f"{s['exhaustive_queries']} questions → {s['target_error_deg']:.2f}°"),
    )
    if s["reached_at"] is not None:
        minutes = s["saved"] * SECONDS_PER_QUERY / 60
        html += (
            f'<div class="ph-callout">Matched a full exhaustive pass after '
            f"<b>{s['reached_at']} questions</b>: {s['saved']} fewer, about "
            f"{minutes:.0f} minutes of clinic time at {SECONDS_PER_QUERY:.0f} s per question."
            "</div>"
        )
    html += f'<p class="ph-note">Next flash: electrode {electrode} at {current:.0f} µA.</p>'
    return html


def build(cfg: dict, warm: bool = False) -> gr.Blocks:
    viewer = Viewer(cfg["checkpoint_dir"], resolution=cfg["render_resolution"])
    if warm:
        viewer.warm()
    px, res = cfg["display_px"], cfg["render_resolution"]
    gcfg = cfg["game"]
    population = load_config(gcfg["patient_config"], False)["patient_models"]
    params = PatientParams(**population[gcfg["patient_model"]])

    def encoder_choices(mode: str) -> list[tuple[str, str]]:
        return [(ENCODER_LABELS[k], k) for k in viewer.available_encoders(mode)]

    def on_mode(mode: str, encoder: str):
        available = viewer.available_encoders(mode)
        return gr.update(choices=encoder_choices(mode),
                         value=encoder if encoder in available else "E2")  # fmt: skip

    def on_sample(name: str):
        return SAMPLES[name](), SAMPLE_MODE[name]

    def run_view(image, encoder, mode, n, dead_pct, adaptation, seed):
        if image is None:
            empty = "Choose a sample, upload a photo, or take a webcam snapshot."
            return None, None, f'<p class="ph-note">{empty}</p>'
        if encoder not in viewer.available_encoders(mode):
            encoder = "E2"  # the encoder radio is switched to E2 at the same time
        start = time.perf_counter()
        settings = ViewSettings(
            int(n), float(dead_pct) / 100, float(adaptation), encoder, mode, int(seed or 0)
        )
        try:
            out = viewer.view(image, settings)
        except Exception as exc:  # shown to the visitor instead of a stack trace
            raise gr.Error(f"Could not process this image ({type(exc).__name__}). "
                           "Try another photo or a sample.") from exc  # fmt: skip
        stats = chips(
            ("Electrodes lit", f"{out.active_electrodes} / {int(n)}"),
            ("Current", f"{out.total_current_uA / 1000:.1f} mA"),
            ("Computed in", f"{time.perf_counter() - start:.1f} s"),
        )
        return to_display(out.drawn, px), to_display(out.percept, px), stats

    def new_game(seed):
        game = CalibrationGame(gcfg["n_electrodes"], params, int(seed), res)
        return game, seed, *game_outputs(game)

    def next_patient(seed):
        return new_game(int(seed) + 1)

    def game_outputs(game):
        return (
            with_fixation(to_display(game.flash(), px)),
            game.map_image(px),
            game_status(game),
            error_plot(game),
        )

    def clicked(game, evt: gr.SelectData):
        col, row = (int(v * res / px) for v in evt.index)
        game.answer((min(row, res - 1), min(col, res - 1)))
        return game, *game_outputs(game)

    def nothing(game):
        game.answer(None)
        return game, *game_outputs(game)

    def autoplay(game):
        game.autoplay(gcfg["autoplay_step"])
        return game, *game_outputs(game)

    no_buttons: list = []
    first = "Person"
    # Uploaded files sit in Gradio's cache; delete them after an hour at most.
    with gr.Blocks(title="Phosphene Lab", fill_width=True, delete_cache=(1800, 3600)) as demo:
        gr.HTML(HEADER, padding=False)
        with gr.Tab("What the patient sees"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=320):
                    gr.HTML('<p class="ph-section-title">1 · Scene</p>', padding=False)
                    sample = gr.Radio(list(SAMPLES), value=first, label="Try a sample",
                                      show_label=True)  # fmt: skip
                    image = gr.Image(
                        value=SAMPLES[first](), sources=["upload", "webcam", "clipboard"],
                        type="numpy", label="Or use your own photo or webcam", height=200,
                        buttons=no_buttons,
                    )  # fmt: skip
                    gr.HTML('<p class="ph-section-title">2 · Implant</p>', padding=False)
                    encoder = gr.Radio(
                        encoder_choices(SAMPLE_MODE[first]), value="E2", label="Encoder",
                        info="How the camera image becomes electrode currents",
                    )  # fmt: skip
                    mode = gr.Radio(
                        [(MODE_LABELS[m], m) for m in MODE_LABELS], value=SAMPLE_MODE[first],
                        label="Priority", info="What the scene encoders keep when dots are scarce",
                    )  # fmt: skip
                    with gr.Row():
                        n = gr.Dropdown(list(ELECTRODE_COUNTS), value=256, label="Electrodes",
                                        info="More electrodes, finer detail")  # fmt: skip
                        seed = gr.Number(value=0, precision=0, minimum=0, maximum=999,
                                         label="Virtual patient",
                                         info="Each patient's dots land differently")  # fmt: skip
                    dead = gr.Slider(0, 50, value=10, step=5, label="Dead electrodes (%)",
                                     info="Electrodes that produce no light")  # fmt: skip
                    adaptation = gr.Slider(
                        0, 5, value=0, step=0.5, label="Steady viewing (seconds)",
                        info="Phosphenes fade when the image does not change",
                    )  # fmt: skip
                with gr.Column(scale=7, min_width=360, elem_classes="ph-sticky"):
                    with gr.Row():
                        drawn = gr.Image(label="What the encoder asks for", height=px,
                                         interactive=False, buttons=no_buttons,
                                         elem_classes="ph-display")  # fmt: skip
                        percept = gr.Image(label="What the patient perceives", height=px,
                                           interactive=False, buttons=no_buttons,
                                           elem_classes="ph-display")  # fmt: skip
                    stats = gr.HTML(padding=False)
                    with gr.Accordion("What am I looking at?", open=False):
                        gr.HTML(ENCODER_HELP, padding=False)
                    gr.HTML(
                        '<p class="ph-note">Photos are processed on the server and deleted '
                        "within an hour. Nothing is shared or used for training.</p>"
                    )
            controls = [image, encoder, mode, n, dead, adaptation, seed]
            for c in (encoder, mode, n, dead, adaptation, seed):
                c.change(run_view, controls, [drawn, percept, stats])
            image.change(run_view, controls, [drawn, percept, stats])
            sample.input(on_sample, sample, [image, mode])
            mode.change(on_mode, [mode, encoder], encoder)
            demo.load(run_view, controls, [drawn, percept, stats])

        with gr.Tab("Calibration game"):
            gr.HTML(
                '<p class="ph-lead">Before an implant is useful, the clinic must learn where '
                "each electrode's dot appears. Here <b>you are the patient</b>.</p>" + GAME_STEPS
            )
            game = gr.State()
            game_seed = gr.State(0)
            with gr.Row(equal_height=False):
                with gr.Column(min_width=320):
                    flash = gr.Image(label="Your view: click where you saw the flash",
                                     height=px, interactive=False, buttons=no_buttons,
                                     elem_classes=["ph-display", "ph-clickable"])  # fmt: skip
                    with gr.Row():
                        saw_nothing = gr.Button("I saw nothing", variant="primary")
                        auto = gr.Button(f"Answer {gcfg['autoplay_step']} for me")
                with gr.Column(min_width=320):
                    game_map = gr.Image(label="The system's map of the implant", height=px,
                                        interactive=False, buttons=no_buttons,
                                        elem_classes="ph-display")  # fmt: skip
                    gr.HTML(LEGEND, padding=False)
            status = gr.HTML(padding=False)
            with gr.Row(equal_height=False):
                plot = gr.Plot(show_label=False, scale=3)
                with gr.Column(scale=1, min_width=200):
                    restart = gr.Button("New patient", variant="secondary")
                    gr.HTML('<p class="ph-note">Starts over with a different simulated '
                            "patient.</p>")  # fmt: skip
            outs = [game, flash, game_map, status, plot]
            flash.select(clicked, game, outs)
            saw_nothing.click(nothing, game, outs)
            auto.click(autoplay, game, outs)
            restart.click(next_patient, game_seed, [game, game_seed, *outs[1:]])
            demo.load(new_game, game_seed, [game, game_seed, *outs[1:]])
    return demo


def smoke(cfg: dict) -> None:
    """Exercise the viewer (E0, E1) and the game without a browser or network weights."""
    start = time.perf_counter()
    viewer = Viewer(cfg["checkpoint_dir"], resolution=cfg["render_resolution"])
    for name in ("E0", "E1"):
        out = viewer.view(data.astronaut(), ViewSettings(n_electrodes=256, encoder=name))
        print(f"viewer {name}: {out.active_electrodes} electrodes on")
    gcfg = cfg["game"]
    population = load_config(gcfg["patient_config"], False)["patient_models"]
    game = CalibrationGame(
        gcfg["n_electrodes"], PatientParams(**population[gcfg["patient_model"]]), 0,
        cfg["render_resolution"],
    )  # fmt: skip
    game.autoplay(gcfg["autoplay_step"])
    s = game.status()
    print(f"game: {s['queries']} queries, map error {s['location_error_deg']:.2f} deg")
    game_status(game)
    error_plot(game)
    build(cfg)
    print(f"UI built; smoke run took {time.perf_counter() - start:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--quick", action="store_true", help="headless smoke run")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    if args.quick:
        smoke(cfg)
        return
    description = "See what a cortical visual implant user might see, and help calibrate one."
    build(cfg, warm=True).launch(server_port=cfg["server_port"], **launch_options(description))


if __name__ == "__main__":
    main()
