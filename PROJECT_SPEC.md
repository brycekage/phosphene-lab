# Project Spec: phosphene-lab

**An end-to-end sandbox for encoding and calibrating cortical visual prostheses**

Owner: Bryce Kageyama
Builder: Claude Code (working with Bryce)
Target audience for the finished project: Neuralink (Blindsight team) and similar BCI companies (Science Corp, Cortigent, Paradromics)
Expected timeline: 8 to 10 weeks part-time

---

## 1. Why this project exists

Cortical visual prostheses like Neuralink's Blindsight stimulate visual cortex directly. Hardware can now place 1,000+ electrodes in V1, but three software problems block useful vision:

1. **Encoding.** Each electrode produces a "phosphene," a spot of light that is irregular in size, fades with repeated stimulation, and merges with its neighbors. Turning a camera frame into stimulation patterns that produce a recognizable percept is unsolved. Most systems still use naive brightness thresholding.
2. **Calibration.** Every patient's phosphene map is different. Clinicians currently spend weeks of manual sessions asking "where did you see the flash?" for each electrode.
3. **Evaluation.** The field lacks standard, reproducible benchmarks for comparing encoders.

This project builds a simulation sandbox that attacks all three: a differentiable cortical phosphene simulator, a set of encoders from naive to learned, an active-learning calibration method that cuts the number of patient queries, and a benchmark that ties them together. It extends Bryce's existing **Optical Vessel-Avoidance Insertion Planner** (U-Net retinal vessel segmentation modeled on Neuralink R1 insertion planning), so the portfolio tells one story: *where to place electrodes → what the patient sees → how to encode the world for them → how to calibrate fast*.

### Non-goals

- No hardware, no animal or patient data, no medical claims.
- Not a new biophysical model of cortex. Use simplified, literature-backed models and say so.
- Not a claim to beat published state of the art. The goal is a clean, honest, reproducible system with clear ablations.

---

## 2. Prior work to build on (read before Phase 1)

Claude Code should read the abstracts and method sections of these and cite them in the README. Do not claim novelty for ideas that come from these.

- **pulse2percept** (Beyeler et al.): open-source Python simulator for prosthetic vision. Recent versions include cortical models and implants, possibly including a Neuralink-style thread array. Check the installed version.
- **dynaphos** (van der Grinten et al., eLife 2024): biologically plausible, differentiable cortical phosphene simulator built for optimizing encoders.
- **Hybrid Neural Autoencoders** (Granley, Relic & Beyeler, NeurIPS 2022): end-to-end learned stimulus encoders through a differentiable forward model.
- **Human-in-the-loop optimization for deep stimulus encoding** (Granley et al., NeurIPS 2023): per-patient tuning of encoders with preference queries.
- **de Ruyter van Steveninck et al. 2022** (Journal of Vision): end-to-end optimization of prosthetic vision.
- **Wedge-dipole retinotopy model** (Polimeni et al. 2006) and **cortical magnification** (Horton & Hoyt 1991): mapping between V1 cortex coordinates and visual field.

If a citation cannot be verified, flag it to Bryce instead of inventing details.

---

## 3. Tech stack and constraints

- Python 3.11, PyTorch, NumPy, OpenCV, scikit-image, BoTorch/GPyTorch (calibration), pulse2percept and dynaphos (cross-validation of simulator), Gradio or Streamlit (demo), matplotlib/plotly (figures).
- Tooling: `uv` or `pip` + `pyproject.toml`, `ruff`, `pytest`, GitHub Actions CI, pre-commit.
- Configs in YAML (one config per experiment). Seed everything.
- **Compute budget:** every training or experiment script must finish in under 1 hour on a single Colab T4 or a laptop CPU for the small configs. Provide a `--quick` flag that runs in under 2 minutes for CI and smoke tests.
- Ask Bryce before adding any dependency over ~500 MB or any paid service.

---

## 4. Repository layout

```
phosphene-lab/
  README.md                  # results-first, with GIF demo
  PROJECT_SPEC.md            # this file
  pyproject.toml
  configs/                   # YAML experiment configs
  src/phosphene_lab/
    simulator/
      retinotopy.py          # wedge-dipole cortex <-> visual field
      array.py               # electrode array geometry + insertion sampling
      phosphene.py           # differentiable percept renderer
      patient.py             # VirtualPatient with hidden params + noisy responses
    encoders/
      naive.py               # E0 downsample / threshold
      edges.py               # E1 edge-based
      scene.py               # E2 segmentation + depth scene simplification
      learned.py             # E3 end-to-end CNN encoder
    calibration/
      baselines.py           # exhaustive grid + staircase
      active.py              # Bayesian active learning
    eval/
      observer.py            # "virtual observer" classifiers trained on percepts
      metrics.py
      tasks.py               # letter ID, object ID, obstacle detection
    integration/
      vessel_planner.py      # import placements from the insertion planner
  scripts/                   # one script per experiment, writes to results/
  results/                   # CSVs + figures, generated only by scripts
  app/                       # Gradio/Streamlit demo
  tests/
  report/                    # technical report (Markdown -> PDF)
```

---

## 5. Phases, deliverables, and acceptance criteria

Work one phase at a time. At the end of each phase: run tests, commit, and post a short summary to Bryce (what was built, key numbers, open questions) before starting the next phase.

### Phase 0: Setup (0.5 week)

- Scaffold the repo, `pyproject.toml`, ruff, pytest, pre-commit, GitHub Actions running `pytest` and `--quick` smoke tests.
- Install pulse2percept and dynaphos; write a notebook that renders one percept from each to confirm they work.

**Done when:** CI is green on an empty test suite and both reference simulators render an image.

### Phase 1: Differentiable cortical simulator (1.5 weeks)

Build a lightweight PyTorch simulator, owned and understood end to end, then cross-check it against dynaphos/pulse2percept.

- `retinotopy.py`: wedge-dipole mapping from V1 surface coordinates (mm) to visual field (degrees) and back. Cortical magnification drives phosphene size.
- `array.py`: electrode arrays of configurable size (64, 256, 1024, 1600). Support a regular grid and a "thread insertion" mode that samples sites semi-randomly over a cortical patch, with an optional exclusion mask (hook for the vessel planner).
- `phosphene.py`: render a percept image (default 128x128, ~20 degree field) from a stimulation vector. Model:
  - Gaussian phosphenes whose size scales with eccentricity.
  - Brightness as a sigmoid of current minus a per-electrode threshold.
  - Current spread so neighboring active electrodes merge.
  - Adaptation: brightness decays with repeated stimulation over frames (exponential, configurable time constant).
  - Electrode dropout (dead electrodes).
  - Hard limits: max current per electrode, max simultaneously active electrodes (power budget).
- `patient.py`: `VirtualPatient` with hidden ground-truth params (location jitter vs. retinotopic prediction, thresholds, size scale, dropout). Exposes noisy query responses: "point to where you saw it" (location + Gaussian noise) and "did you see it?" (yes/no with lapse rate).

**Tests (write first):** zero stimulation renders black; dead electrodes never light; phosphene size increases with eccentricity; gradients flow from percept to stimulation; adaptation reduces brightness over repeated frames; the same seed gives identical output.

**Done when:** tests pass, a figure in `results/` compares this simulator's percepts with dynaphos for the same input, and the README explains each modeling assumption in plain language.

### Phase 2: Encoders (2 weeks)

All encoders take an RGB frame plus a patient phosphene map and output a stimulation vector that respects the current and power limits.

- **E0 Naive:** sample image brightness at each phosphene's visual-field location, threshold.
- **E1 Edges:** Canny/Sobel edges sampled at phosphene locations.
- **E2 Scene simplification:** pretrained monocular depth (e.g., Depth Anything V2 Small) + lightweight semantic segmentation (e.g., torchvision DeepLabV3-MobileNetV3). Highlight near obstacles, walls/floor boundaries, people, and doors; suppress texture and background. Configurable priority modes ("navigation", "people", "reading").
- **E3 Learned end-to-end:** a small CNN maps image to stimulation, trained through the differentiable simulator. Loss = task loss from a decoder or observer + reconstruction loss on the simplified target + sparsity/power penalty. Train across randomized virtual patients so it generalizes to unseen phosphene maps (pass the phosphene map as input).

**Datasets:** EMNIST letters (reading), a COCO subset (objects/people), NYU Depth V2 (indoor navigation). Download scripts only; do not commit data.

**Done when:** each encoder runs on all three datasets, a side-by-side figure (original → E0..E3 percepts) exists at 256 and 1024 electrodes, and E3 training fits the compute budget.

### Phase 3: Benchmark (1 week)

- **Virtual observer:** train a fixed classifier on percepts for each task (letter ID, 10-class object ID, obstacle present + left/center/right). Train the observer per encoder on its own percepts so encoders are judged by how much task information survives, not by matching one observer.
- **Metrics:** task accuracy, SSIM vs. simplified target, mean active electrodes, mean charge.
- **Sweeps:** encoder × electrode count (64, 256, 1024, 1600) × dropout (0, 10, 30%) × patient variability. 3 seeds each, report mean ± std.

**Headline figure:** task accuracy vs. electrode count, one line per encoder, per task.

**Done when:** `scripts/run_benchmark.py --config configs/benchmark.yaml` regenerates every table and figure from scratch.

### Phase 4: Fast calibration (1.5 weeks)

Problem: for a new virtual patient, estimate each electrode's phosphene location and threshold with as few queries as possible.

- **Baselines:** exhaustive (every electrode, repeated localization + staircase threshold); random query order.
- **Method:** Bayesian active learning. Use the retinotopic model as a prior and a Gaussian process over cortex position for the residual, so one query informs neighboring electrodes. Choose the next electrode (and current level) by expected information gain. Use BoTorch/GPyTorch.
- **Metrics:** localization error (degrees) and threshold error vs. number of queries; estimated clinic time assuming ~5 s per query.
- **Close the loop:** rerun the Phase 3 benchmark for E2/E3 using (a) the true map, (b) the uncalibrated retinotopic guess, (c) the calibrated map at 100/300/1000 queries. This shows calibration's effect on actual task performance, which is the result a Neuralink engineer cares about.
- **Stretch:** preference-based tuning of encoder parameters per patient (in the style of Granley et al. 2023).

**Done when:** a figure shows queries-to-target-error for active vs. baselines, and the closed-loop table exists.

### Phase 5: Interactive demo (1 week)

Gradio app (deployable to Hugging Face Spaces):

- Webcam or uploaded image/video → side by side: original, simplified scene, phosphene percept.
- Sliders: electrode count, dropout, adaptation, encoder choice, priority mode.
- **Calibration game:** the user plays the patient. A flash appears, they click where they saw it, and the active-learning map updates live, with a counter of queries saved vs. exhaustive.
- Record a 30 to 60 second GIF/video for the README.

**Done when:** the app runs locally with one command and the GIF is in the README.

### Phase 6: Optional sighted-volunteer pilot (0.5 week)

- Web version of letter ID and obstacle tasks under E0 vs. E2 vs. E3 percepts, 5 to 10 sighted volunteers.
- Note in the report that this is an informal pilot. Any publication requires CMC IRB approval first; Claude Code should remind Bryce of this before collecting any data.

### Phase 7: Write-up and polish (1 week)

- **README:** one-paragraph pitch, demo GIF, headline figure, results table, how to reproduce, limitations, citations.
- **Technical report** (4 to 6 pages, arXiv style): problem, simulator assumptions, encoders, benchmark, calibration, results, limitations, future work.
- **Integration section:** load electrode placements from the vessel-avoidance planner (`integration/vessel_planner.py`) and show the full pipeline from insertion plan to percept in one figure.
- Short blog/LinkedIn post draft in plain language.

---

## 6. Rules for Claude Code

1. **Never fabricate results.** Every number in the README or report must come from a file in `results/` produced by a script in `scripts/`. If an experiment fails or underperforms, report it as is.
2. **Tests before simulator code.** Physics-like behavior (Phase 1 tests) must be tested before encoders build on it.
3. **Stay in budget.** If a run will exceed 1 hour on a T4, shrink it and tell Bryce.
4. **Explain assumptions.** Every modeling simplification gets one plain-language sentence in the README "Assumptions" section.
5. **Check in at phase boundaries.** Summarize, list open questions, and wait for Bryce's go-ahead.
6. **Teach as you go.** Bryce must be able to explain every component in an interview. After each phase, add a short "How it works" note in `docs/` written for a smart non-specialist.
7. **Small commits** with clear messages, one milestone per commit at minimum.
8. **Ask before** large dependencies, paid services, changing the scope, or collecting any human data.

---

## 7. Definition of done

- One command reproduces all results: `make reproduce` (or equivalent).
- CI green, test coverage on `simulator/` ≥ 80%.
- README leads with the demo GIF and the accuracy-vs-electrodes figure.
- Calibration results show a clear query reduction vs. exhaustive, with honest error bars.
- Technical report PDF in `report/`.
- Demo live on Hugging Face Spaces.

---

## 8. What this demonstrates to a Neuralink reviewer

- Understands the real bottleneck (encoding and calibration, not electrode count) and the neuroscience behind it (retinotopy, cortical magnification, adaptation).
- Can build a differentiable simulator and train models through it.
- Thinks about the clinic: power limits, dead electrodes, patient time.
- Runs rigorous, reproducible experiments with ablations and honest limitations.
- Connects to prior work (insertion planning) as one coherent pipeline.

---

## 9. Kickoff prompt for Claude Code

> Read PROJECT_SPEC.md in full. Then read the prior work in Section 2 (abstracts and methods) and summarize in 10 lines how each paper informs our design. Propose any changes to the spec, then start Phase 0. Stop at the end of Phase 0 and report back.
