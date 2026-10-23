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
    ENCODERS,
    CalibrationGame,
    Viewer,
    ViewSettings,
)
from phosphene_lab.encoders.scene import MODES  # noqa: E402
from phosphene_lab.simulator.patient import PatientParams  # noqa: E402


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
    "astronaut (people)": data.astronaut,
    "coffee cup": data.coffee,
    "cat": data.chelsea,
    "EXIT sign (reading)": sign_image,
}


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


def error_plot(game: CalibrationGame):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.plot(range(len(game.history)), game.history, color="tab:green", label="active (you)")
    ax.axhline(
        game.target_error_deg, color="gray", ls="--",
        label=f"full exhaustive pass ({game.exhaustive_queries} queries)",
    )  # fmt: skip
    ax.set_xlabel("queries answered")
    ax.set_ylabel("map error (deg)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def game_status(game: CalibrationGame) -> str:
    s = game.status()
    electrode, current = game.pending
    lines = [
        f"**Queries answered:** {s['queries']} · **map error:** {s['location_error_deg']:.2f} deg"
        f" · **a full exhaustive pass ({s['exhaustive_queries']} queries) reaches:** "
        f"{s['target_error_deg']:.2f} deg",
    ]
    if s["reached_at"] is not None:
        minutes = s["saved"] * SECONDS_PER_QUERY / 60
        lines.append(
            f"Reached exhaustive-pass accuracy after **{s['reached_at']} queries**, saving "
            f"**{s['saved']} queries** (~{minutes:.0f} clinic minutes at 5 s per query)."
        )
    lines.append(f"Next question: electrode {electrode} at {current:.0f} uA.")
    return "\n\n".join(lines)


def build(cfg: dict) -> gr.Blocks:
    viewer = Viewer(cfg["checkpoint_dir"], resolution=cfg["render_resolution"])
    px, res = cfg["display_px"], cfg["render_resolution"]
    gcfg = cfg["game"]
    population = load_config(gcfg["patient_config"], False)["patient_models"]
    params = PatientParams(**population[gcfg["patient_model"]])

    def encoder_choices(mode: str) -> list[tuple[str, str]]:
        return [(ENCODERS[k], k) for k in viewer.available_encoders(mode)]

    def run_view(image, encoder, mode, n, dropout, adaptation, seed):
        if image is None:
            return None, None, "Choose a sample, upload a photo, or take a webcam snapshot."
        if encoder not in viewer.available_encoders(mode):
            gr.Warning(f"{ENCODERS[encoder]} has no trained model for {mode} mode; using E2.")
            encoder = "E2"
        start = time.perf_counter()
        settings = ViewSettings(int(n), float(dropout), float(adaptation), encoder, mode, int(seed))
        out = viewer.view(image, settings)
        stats = (
            f"**{out.active_electrodes}** of {int(n)} electrodes on · total current "
            f"**{out.total_current_uA / 1000:.1f} mA** per frame · computed in "
            f"{time.perf_counter() - start:.1f} s"
        )
        return to_display(out.drawn, px), to_display(out.percept, px), stats

    def new_game(seed):
        game = CalibrationGame(gcfg["n_electrodes"], params, int(seed), res)
        return game, *game_outputs(game)

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

    with gr.Blocks(title="phosphene-lab") as demo:
        gr.Markdown(
            "# phosphene-lab\nA simulation of what a cortical visual prosthesis user might see, "
            "and of how fast the implant can be calibrated. All patients here are simulated."
        )
        with gr.Tab("Viewer"):
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(
                        sources=["upload", "webcam", "clipboard"], type="numpy",
                        label="Camera image", height=px,
                    )  # fmt: skip
                    sample = gr.Radio(list(SAMPLES), label="Sample images")
                drawn = gr.Image(label="What the encoder draws (in the visual field)", height=px)
                percept = gr.Image(label="Phosphene percept", height=px)
            stats = gr.Markdown()
            with gr.Row():
                mode = gr.Radio(list(MODES), value="navigation", label="Priority mode")
                encoder = gr.Radio(encoder_choices("navigation"), value="E2", label="Encoder")
            with gr.Row():
                n = gr.Dropdown(list(ELECTRODE_COUNTS), value=256, label="Electrodes")
                dropout = gr.Slider(0, 0.5, value=0.1, step=0.05, label="Dead electrodes")
                adaptation = gr.Slider(
                    0, 5, value=0, step=0.5, label="Adaptation: seconds of steady viewing"
                )
                seed = gr.Number(value=0, precision=0, label="Virtual patient")
            gr.Markdown(
                "E2 and E3 run two pretrained networks (depth and segmentation, ~115 MB, "
                "downloaded once); the first use takes ~20 s, then ~1 s per image on a laptop CPU."
            )
            controls = [image, encoder, mode, n, dropout, adaptation, seed]
            for c in controls:
                c.change(run_view, controls, [drawn, percept, stats])
            sample.change(lambda name: SAMPLES[name](), sample, image)
            mode.change(lambda m: gr.update(choices=encoder_choices(m)), mode, encoder)

        with gr.Tab("Calibration game"):
            gr.Markdown(
                "**You are the patient.** Look at the cross in the middle. When a flash appears, "
                "click where you saw it; if you see nothing, press *Saw nothing* (near-threshold "
                "flashes are faint, which is the point). The system chooses each question to learn "
                "as much as possible, and uses each answer to correct its guess for the whole "
                "array. Gray rings: where the phosphenes really are. Green: the current "
                "estimate. Red lines: how far off each estimate is; watch them shrink."
            )
            game = gr.State()
            with gr.Row():
                flash = gr.Image(label="Flash (click where you saw it)", height=px,
                                 interactive=False)  # fmt: skip
                game_map = gr.Image(label="Map: truth (gray rings), estimate (green), error (red)",
                                    height=px, interactive=False)  # fmt: skip
            with gr.Row():
                saw_nothing = gr.Button("Saw nothing")
                auto = gr.Button(f"Let the virtual patient answer {gcfg['autoplay_step']}")
                game_seed = gr.Number(value=0, precision=0, label="Patient")
                restart = gr.Button("New patient")
            status = gr.Markdown()
            plot = gr.Plot(label="Map error")
            outs = [game, flash, game_map, status, plot]
            flash.select(clicked, game, outs)
            saw_nothing.click(nothing, game, outs)
            auto.click(autoplay, game, outs)
            restart.click(new_game, game_seed, outs)
            demo.load(new_game, game_seed, outs)
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
    print(f"game: {game_status(game).splitlines()[0]}")
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
    build(cfg).launch(server_name="127.0.0.1", server_port=cfg["server_port"])


if __name__ == "__main__":
    main()
