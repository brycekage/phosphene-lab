# phosphene-lab

Simulation sandbox for encoding and calibrating cortical visual prostheses.
The full plan is in `PROJECT_SPEC.md`; read the section for the current phase before starting work.

## Commands

- Install / update env: `py -m uv sync`
- Tests: `.venv/Scripts/python.exe -m pytest` (CI also runs `-m "not slow"`)
- Lint: `.venv/Scripts/python.exe -m ruff check .` and `ruff format .`
- Phase 0 smoke run: `.venv/Scripts/python.exe scripts/render_reference.py --quick`
- Phase 1 cross-check: `.venv/Scripts/python.exe scripts/compare_simulators.py [--quick]`
- Phase 2: `scripts/download_data.py` (~4 GB raw into `data/raw`, once), `scripts/train_e3.py`,
  `scripts/encoder_gallery.py`. Phase 3: `scripts/run_benchmark.py`. Phase 4:
  `scripts/run_calibration.py`, `scripts/closed_loop.py` (needs E3 checkpoints from train_e3),
  `scripts/pointing_task.py`. `scripts/obstacle_ceiling.py` measures no-simulator ceilings.
- Phase 5 demo: `.venv/Scripts/python.exe app/app.py` (http://127.0.0.1:7860; `--quick` is a
  headless smoke run). Logic in `src/phosphene_lab/demo/`. The README GIF is browser
  screenshots of the running app in `results/quick/demo_frames/`, assembled by
  `scripts/make_demo_gif.py`.
- E2 targets are cached in `data/<dataset>/targets_*.npy`, keyed by image content hash.
- Configs load via `phosphene_lab.config.load_config`; every `quick:` block must redirect outputs
  to `results/quick/` (git-ignored) so smoke runs never overwrite real results.

## Conventions

- Python >=3.11 (local venv is 3.11), CPU-only torch locally and in CI; Colab T4 for heavy training.
- One YAML config per experiment in `configs/`; every script takes `--config` and `--quick` (< 2 min).
- Seed everything. Scripts write to `results/`; nothing else does.
- Reference simulators (pulse2percept, dynaphos) are wrapped in `src/phosphene_lab/simulator/reference.py`.

## Rules (from PROJECT_SPEC.md §6)

1. Never fabricate results. Every number in the README or report comes from a file in `results/`
   produced by a script in `scripts/`. Report failures and underperformance as they are.
2. Tests before simulator code. Physics-like behavior is tested before encoders build on it.
3. Stay in budget: every run < 1 hour on a Colab T4 or laptop CPU. If not, shrink it and say so.
4. Every modeling simplification gets one plain-language sentence in the README "Assumptions" section.
5. Check in at phase boundaries: summarize, list open questions, wait for Bryce's go-ahead.
6. Teach as you go: after each phase add a "How it works" note in `docs/how-it-works/`
   for a smart non-specialist.
7. Small commits with clear messages, at least one per milestone.
8. Ask before large dependencies (> ~500 MB), paid services, scope changes, or collecting any
   human data (Phase 6 requires CMC IRB approval before publication).

If a citation cannot be verified, flag it instead of inventing details.
