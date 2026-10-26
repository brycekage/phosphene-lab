"""Phase 6 pilot page: volunteers name letters and spot obstacles in simulated implant vision.

Screens: welcome -> (per part: instructions -> practice with the answer shown -> scored
trials) -> results. The three encoders' pictures are shuffled together and never named until
the results. Nothing is stored: the session lives in the page and is gone when it closes.

Usage:
    python app/pilot.py [--config configs/phase6_pilot.yaml] [--quick]

Run ``scripts/make_pilot_stimuli.py`` first. ``--quick`` plays a full session headlessly
against the quick stimuli and builds the UI, without a browser.
"""

from __future__ import annotations

import argparse
import secrets
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from phosphene_lab.config import load_config
from phosphene_lab.demo.pilot import PilotSession, Trial, load_trials
from phosphene_lab.demo.theme import launch_options

BLOCKS = ("letters", "obstacles")
OBSTACLE_BUTTONS = [  # (label, answer, key class)
    ("← Left", "left", "ph-key-arrowleft"),
    ("↑ Ahead", "center", "ph-key-arrowup"),
    ("→ Right", "right", "ph-key-arrowright"),
    ("Nothing close", "none", "ph-key-space"),
]
OBSTACLE_WORDS = {"left": "on the left", "center": "ahead", "right": "on the right",
                  "none": "nothing close"}  # fmt: skip
CHANCE = {"letters": 1 / 26, "obstacles": 1 / 4}
ENCODER_NAMES = {  # (name, one-line description)
    "E0": ("E0 · Brightness", "lights each dot by how bright the camera image is there"),
    "E1": ("E1 · Edges", "lights the outlines of things"),
    "E2": ("E2 · Scene", "neural networks find what matters and draw only that"),
    "E3": ("E3 · Learned", "a network trained through the simulator"),
}

PART = {
    "letters": {
        "title": "Reading letters",
        "prompt": "Which letter is this?",
        "noun": "Letter",
        "body": """<p class="ph-lead">You will see a handwritten letter the way an implant
        might draw it: a few dots of light. Pick the letter you think it is. Letters may be
        capital or lowercase.</p>
        <p class="ph-note">Shortcut: type the letter on your keyboard.</p>""",
    },
    "obstacles": {
        "title": "Spotting obstacles",
        "prompt": "Where is the nearest obstacle?",
        "noun": "Scene",
        "body": """<p class="ph-lead">You will see a room the way an implant might draw it,
        as if you were standing in it. Say where something is blocking your path within
        about 1.5 meters, or that nothing is close.</p>
        <p class="ph-note">Shortcuts: <span class="ph-kbd">←</span>
        <span class="ph-kbd">↑</span> <span class="ph-kbd">→</span> and
        <span class="ph-kbd">Space</span> for nothing close.</p>""",
    },
}

WELCOME = """
<div class="ph-header">
  <p class="ph-eyebrow">Phosphene Lab · volunteer pilot</p>
  <h1>Can you see with an implant's eyes?</h1>
  <p class="ph-lead">People who are blind may one day see again through an implant in the
  visual cortex. It would not show a picture, only a few hundred dots of light. This short
  test asks how well people can use those dots, and which way of drawing them works best.</p>
</div>
<ul class="ph-steps">
  <li><b>About 10 minutes</b>, in two parts: reading letters, then spotting obstacles.</li>
  <li>Each part starts with <b>3 practice rounds</b> that show you the answer.</li>
  <li>Many pictures are hard on purpose. <b>Guess when unsure</b>; there is no penalty.</li>
</ul>
<div class="ph-callout" style="margin-top: 14px">
  <b>Your answers are not saved.</b> They stay in this browser tab and disappear when you
  close it. This is an informal pilot, not a research study, and no results are published.
</div>
<p class="ph-note" style="margin-top: 10px">Works best on a laptop or desktop, in a dim room.
Keyboard shortcuts are available.</p>
"""

# Keyboard shortcuts: a key clicks the visible button carrying the matching ph-key-* class.
# A short lockout stops a held key from answering the next picture before it is seen.
KEYS_JS = """
() => {
  let locked = false;
  document.addEventListener("keydown", (e) => {
    if (locked || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest && e.target.closest("input, textarea, select")) return;
    const name = e.key === " " ? "space" : e.key.toLowerCase();
    if (!/^([a-z]|arrow(left|right|up)|space|enter)$/.test(name)) return;
    const button = [...document.querySelectorAll(".ph-key-" + name)]
      .find((b) => b.offsetParent !== null && !b.disabled);
    if (!button) return;
    e.preventDefault();
    locked = true;
    setTimeout(() => { locked = false; }, 350);
    button.click();
  });
}
"""


def progress_html(session: PilotSession) -> str:
    trial = session.current
    task = trial.task
    part = BLOCKS.index(task) + 1
    if trial.practice:
        practice = [t for t in session.order if t.task == task and t.practice]
        done = sum(1 for t, _, _ in session.responses if t.task == task and t.practice)
        label, pct = f"Practice {done + 1} of {len(practice)}", 0
    else:
        done, total = session.progress()
        label, pct = f"{PART[task]['noun']} {done + 1} of {total}", 100 * done / total
    return (
        f'<p class="ph-section-title">Part {part} of {len(BLOCKS)} · {PART[task]["title"]}'
        f" · {label}</p>"
        f'<div class="ph-progress" role="progressbar" aria-valuenow="{pct:.0f}" '
        f'aria-valuemin="0" aria-valuemax="100"><div style="width: {pct:.0f}%"></div></div>'
    )


def answer_words(task: str, answer: str) -> str:
    return answer.upper() if task == "letters" else OBSTACLE_WORDS[answer]


def results_html(session: PilotSession, seconds: float) -> str:
    minutes = round(seconds / 60)
    took = "under a minute" if minutes < 1 else f"{minutes} minute{'s' * (minutes != 1)}"
    tables = []
    for task, cells in session.summary().items():
        rows = "".join(
            f"<tr><td><b>{ENCODER_NAMES[enc][0]}</b><br>"
            f'<span class="ph-note">{ENCODER_NAMES[enc][1]}</span></td>'
            f"<td><b>{correct} / {n}</b> &nbsp;{100 * correct / n:.0f}%</td></tr>"
            for enc, (correct, n) in cells.items()
        )
        tables.append(
            f'<p class="ph-section-title">{PART[task]["title"]} · guessing scores about '
            f"{100 * CHANCE[task]:.0f}%</p>"
            f'<table class="ph-results"><thead><tr><th>Drawing method</th><th>Your score</th>'
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )
    return (
        '<div class="ph-header"><p class="ph-eyebrow">Phosphene Lab · volunteer pilot</p>'
        "<h1>Thank you! Here is how you did</h1>"
        f'<p class="ph-lead">You finished in {took}. Each row is one way '
        "of turning a camera image into dots; you saw them mixed together without knowing "
        "which was which.</p></div>"
        + "".join(tables)
        + '<p class="ph-note">With 8 to 10 pictures per method, a '
        "difference of one or two pictures can be luck. The pilot looks for patterns across "
        "many volunteers.</p>"
        '<div class="ph-callout" style="margin-top: 12px">Nothing has been saved. Closing '
        "this tab erases your answers.</div>"
    )


def build(cfg: dict) -> gr.Blocks:
    stimuli = Path(cfg["stimuli_dir"])
    trials, classes = load_trials(stimuli)
    px = cfg["display_px"]

    @lru_cache(maxsize=256)
    def picture(path: str) -> np.ndarray:
        return np.asarray(Image.open(stimuli / path).convert("RGB"))

    with gr.Blocks(title="Phosphene Lab · Volunteer pilot", fill_width=True) as demo:
        session = gr.State()
        started = gr.State()

        with gr.Column(visible=True) as welcome:
            gr.HTML(WELCOME, padding=False)
            begin = gr.Button("Begin", variant="primary", size="lg",
                              elem_classes="ph-key-enter")  # fmt: skip

        with gr.Column(visible=False) as intro:
            intro_html = gr.HTML(padding=False)
            start_part = gr.Button("Start practice", variant="primary", size="lg",
                                   elem_classes="ph-key-enter")  # fmt: skip

        with gr.Column(visible=False) as trial_screen:
            progress = gr.HTML(padding=False)
            prompt = gr.HTML(padding=False)
            stimulus = gr.Image(show_label=False, interactive=False, height=px, buttons=[],
                                elem_classes="ph-display")  # fmt: skip
            with gr.Column(visible=False, elem_classes="ph-answer-grid") as letters_grid:
                with gr.Row(elem_classes="ph-letter-grid"):  # a CSS grid: 13 or 7 per row
                    letter_buttons = [
                        (ch, gr.Button(ch.upper(), min_width=0, elem_classes=f"ph-key-{ch}"))
                        for ch in "abcdefghijklmnopqrstuvwxyz"
                    ]
            with gr.Row(visible=False, elem_classes="ph-answer-grid") as obstacles_grid:
                obstacle_buttons = [
                    (value, gr.Button(label, min_width=120, elem_classes=key))
                    for label, value, key in OBSTACLE_BUTTONS
                ]

        with gr.Column(visible=False) as feedback_screen:
            feedback_html = gr.HTML(padding=False)
            with gr.Row():
                seen = gr.Image(label="What you saw", interactive=False, height=px * 3 // 4,
                                buttons=[], elem_classes="ph-display")  # fmt: skip
                truth = gr.Image(label="What the camera saw", interactive=False,
                                 height=px * 3 // 4, buttons=[],
                                 elem_classes="ph-display")  # fmt: skip
            next_button = gr.Button("Next", variant="primary", size="lg",
                                    elem_classes="ph-key-enter")  # fmt: skip

        with gr.Column(visible=False) as results_screen:
            results = gr.HTML(padding=False)
            again = gr.Button("Start over", size="lg")

        screens = [welcome, intro, trial_screen, feedback_screen, results_screen]
        outputs = [session, *screens, intro_html, progress, prompt, stimulus, letters_grid,
                   obstacles_grid, feedback_html, seen, truth, results]  # fmt: skip

        def show(screen) -> dict:
            return {s: gr.Column(visible=s is screen) for s in screens}

        def advance(s: PilotSession, t0: float) -> dict:
            """The screen that comes next: results, a part's instructions, or a picture."""
            if s.finished:
                return {session: s, **show(results_screen),
                        results: results_html(s, time.monotonic() - t0)}  # fmt: skip
            trial = s.current
            if s.at_block_start:
                part = BLOCKS.index(trial.task) + 1
                info = PART[trial.task]
                html = (f'<p class="ph-eyebrow">Part {part} of {len(BLOCKS)}</p>'
                        f"<h2>{info['title']}</h2>{info['body']}")  # fmt: skip
                return {session: s, **show(intro), intro_html: html}
            return {
                session: s, **show(trial_screen),
                progress: progress_html(s),
                prompt: f"<h2 style='margin: 4px 0'>{PART[trial.task]['prompt']}</h2>",
                stimulus: picture(trial.image),
                letters_grid: gr.Column(visible=trial.task == "letters"),
                obstacles_grid: gr.Row(visible=trial.task == "obstacles"),
            }  # fmt: skip

        def on_begin():
            s = PilotSession(trials, BLOCKS, seed=secrets.randbits(32))
            t0 = time.monotonic()
            return {started: t0, **advance(s, t0)}

        def on_start_part(s: PilotSession, t0: float):
            s.start_block()
            return advance(s, t0)

        def on_answer(choice: str):
            def handler(s: PilotSession, t0: float):
                if s is None or s.finished:  # a stray click after the end, or a reload
                    return {**show(welcome)}
                s.answer(choice)
                trial, given, correct = s.last
                if not trial.practice:
                    return advance(s, t0)
                right = answer_words(trial.task, trial.label)
                if correct:
                    html = (f'<div class="ph-callout ph-feedback-right">Correct: it was '
                            f"<b>{right}</b>.</div>")  # fmt: skip
                else:
                    html = (f'<div class="ph-callout ph-feedback-wrong">Not quite: it was '
                            f"<b>{right}</b> (you answered {answer_words(trial.task, given)})."
                            "</div>")  # fmt: skip
                return {session: s, **show(feedback_screen), feedback_html: html,
                        seen: picture(trial.image), truth: picture(trial.original)}  # fmt: skip

            return handler

        begin.click(on_begin, None, [started, *outputs])
        start_part.click(on_start_part, [session, started], outputs)
        next_button.click(lambda s, t0: advance(s, t0), [session, started], outputs)
        for value, button in letter_buttons + obstacle_buttons:
            button.click(on_answer(value), [session, started], outputs, trigger_mode="once")
        again.click(lambda: {session: None, **show(welcome)}, None, outputs)
    return demo


def smoke(cfg: dict) -> None:
    """Play a whole session against the quick stimuli and build the UI, without a browser."""
    start = time.perf_counter()
    trials, _ = load_trials(cfg["stimuli_dir"])
    session = PilotSession(trials, BLOCKS, seed=0)
    while not session.finished:
        if session.at_block_start:
            session.start_block()
        trial: Trial = session.current
        if not trial.practice:
            progress_html(session)
        session.answer(trial.label)
    results_html(session, 600.0)
    print(f"pilot: {len(session.responses)} answers, summary {session.summary()}")
    build(cfg)
    print(f"UI built; smoke run took {time.perf_counter() - start:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6_pilot.yaml")
    parser.add_argument("--quick", action="store_true", help="headless smoke run")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    if args.quick:
        smoke(cfg)
        return
    options = launch_options(
        "A 10-minute test of reading letters and spotting obstacles in simulated implant vision.",
        js=KEYS_JS, max_width_px=820,
    )  # fmt: skip
    build(cfg).launch(server_port=cfg["server_port"], **options)


if __name__ == "__main__":
    main()
