# phosphene-lab

An end-to-end sandbox for encoding and calibrating cortical visual prostheses.
Work in progress; see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the plan.

## Quickstart

```bash
uv sync
uv run pytest
uv run python scripts/render_reference.py --quick   # writes results/phase0_reference.png
```

## Results so far (Phases 2-4)

All numbers below come from files in `results/`, written by the scripts named. Accuracy is a
virtual observer's (mean ± std over 3 seeds); "majority" is always guessing the most common
test class.

![Accuracy vs electrodes](results/benchmark_accuracy_vs_electrodes.png)

**Benchmark** (`scripts/run_benchmark.py`, `results/benchmark_summary.csv`), 256 electrodes,
no dead electrodes, default patients:

| task (majority) | E0 naive | E1 edges | E2 scene | E3 learned |
|---|---|---|---|---|
| letters (0.054) | 0.51 ± 0.03 | 0.45 ± 0.02 | 0.54 ± 0.01 | **0.69 ± 0.00** |
| objects (0.10) | 0.17 ± 0.04 | 0.15 ± 0.03 | **0.26 ± 0.02** | 0.23 ± 0.02 |
| obstacles (0.353) | 0.35 ± 0.00 | 0.35 ± 0.00 | **0.58 ± 0.02** | 0.33 ± 0.06 |

- E3 (learned) reads letters best at every array size and uses about 3x less total current than
  E0, but it drives every electrode at a low level (some below threshold), so its "active
  electrodes" count is not comparable to the others.
- **E3 failed on obstacles:** training collapsed to almost no stimulation
  (`results/phase2_e3_training.csv`, final mean current 0.0003 of max), so it sits at the
  majority-class guess, as do E0 and E1. Only E2 carries obstacle information.
- Accuracy changes little from 256 to 1600 electrodes, with 30% dead electrodes, or with patient
  variability, in this setup (see the dropout and variability figures). Part of the reason is
  the observer, which is retrained for every patient (see the calibration note below).
- Objects stay hard for every encoder (best: E2 at 1600 electrodes, 0.32 ± 0.04).

**Calibration** (`scripts/run_calibration.py`, `results/phase4_queries_to_target.csv`), 256
electrodes; the target is the error that one full exhaustive pass (2048 queries, ~171 clinic
minutes) reaches:

| patients | error | active needs | random needs |
|---|---|---|---|
| correlated | location | 32 queries (~3 min) | not reached in 2048 |
| correlated | threshold | 1088 queries (~91 min) | not reached in 2048 |
| independent | location | 1984 queries (~165 min) | not reached in 2048 |
| independent | threshold | 1248 queries (~104 min) | 1824 queries (~152 min) |

The large location win depends on most of the error being shared across the array (the
"correlated" assumption); when every electrode errs on its own, active calibration gains little
on location.

**How much must be shared?** (`configs/phase4_shared_sweep.yaml`,
`results/phase4_shared_sweep_queries_to_target.csv`): with total deviation held fixed, the
queries active calibration needs to match a full exhaustive pass on location fall from 1984
(nothing shared) to 1600, 704, 64, and 32 as the shared fraction rises through 0.25, 0.5, 0.75,
and 0.95; for thresholds they fall from 1248 to 864. Sharing pays off once about half of the
error is common to the array.

**Closing the loop** (`scripts/closed_loop.py`, `results/phase4_closed_loop_summary.csv`):

- With an observer retrained for each map (a fully adapted patient), task accuracy was the
  same, within noise, for the true map, the uncalibrated guess, and calibrated maps.
- With a fixed observer trained under the true map (a patient who has not adapted), the
  uncalibrated map costs letter accuracy and calibration recovers most of it: E2 0.41 ± 0.08
  uncalibrated, 0.44 ± 0.04 after 100 queries, 0.46 ± 0.06 after 1000, 0.46 ± 0.07 with the
  true map; E3 0.63 ± 0.05, 0.67 ± 0.03, 0.66 ± 0.04, 0.68 ± 0.03. These gaps are within about
  one standard deviation over 3 seeds, so they are suggestive, not conclusive. Objects and
  obstacles show no difference.
- SSIM to the intended picture improves clearly (E2 letters: 0.462 uncalibrated, 0.517 after 100
  queries, 0.533 with the true map; `results/phase4_closed_loop.csv`).

See [the Phase 4 note](docs/how-it-works/04-calibration.md).

**Compute budget:** E3 trains in about 4-5 min per task on a laptop CPU, calibration takes
under 2 min, and the closed loop about 12 min. The full benchmark took about 72 min of compute,
over the 1-hour budget, even after shrinking the observer and the letter set; it runs as
separate (task, seed) parts.

## How it works

- [Phase 0: the reference simulators](docs/how-it-works/00-setup.md)
- [Phase 1: the phosphene simulator](docs/how-it-works/01-simulator.md)
- [Phase 2: encoders](docs/how-it-works/02-encoders.md)
- [Phase 3: the benchmark](docs/how-it-works/03-benchmark.md)
- [Phase 4: fast calibration](docs/how-it-works/04-calibration.md)

## Assumptions

Every simplification in the simulator, in one sentence each:

- **Retinotopy:** the visual field maps onto V1 through the wedge-dipole model (Polimeni et al.,
  2006), which was fitted to macaque cortex, with human magnification constants from Horton &
  Hoyt (1991), so the map is a cross-species approximation.
- **Two hemispheres, one frame:** both hemispheres use the same map mirrored left-right, and we
  ignore V1's real folding and orientation in the skull.
- **Implant location:** default arrays cover a cortical rectangle spanning roughly 1-8 deg of
  eccentricity, which leaves the central ~1 deg and the regions straight above and below
  fixation without phosphenes.
- **Thread arrays:** each insertion site counts as one electrode; contacts along a thread's depth
  are not modeled.
- **Phosphene shape:** every phosphene is a round Gaussian blob; real phosphenes can be
  elongated, colored, or oddly shaped.
- **Phosphene size:** current spreads over a cortical disk of radius sqrt(I / 675 uA/mm^2), as in
  dynaphos, and the disk is converted to degrees by the cortical magnification at that location.
- **Pixel floor:** phosphenes smaller than a pixel are drawn with a half-pixel blur so they stay
  visible as dots, which makes small central phosphenes look larger and brighter than a
  sub-pixel-accurate renderer would.
- **Brightness:** brightness rises along a sigmoid around each electrode's threshold and does not
  model pulse width or frequency.
- **Merging:** overlapping phosphenes add up and then saturate smoothly, which is a rendering
  choice rather than a model of how the brain combines them.
- **Adaptation:** repeated stimulation dims a phosphene through a single leaky charge trace
  (time constant 1 s by default); real adaptation has several time scales.
- **Power budget:** limits are a per-electrode current cap and a maximum number of
  simultaneously active electrodes, not a model of charge density or tissue safety.
- **Virtual patients:** patient differences are random draws (position error, log-normal
  thresholds and sizes, dead electrodes) with made-up but plausible spreads, not fits to
  patient data.
- **Patient responses:** "did you see it?" answers follow a cumulative-normal curve with random
  lapses, and pointing answers add round Gaussian noise; real patients also show biases.

Encoders and tasks (Phase 2):

- **Image placement:** every encoder stretches the camera image over the smallest box that
  contains the patient's working phosphenes, so the whole image lands where the implant can
  draw; the uncovered meridian strip still cuts through the middle of a centered letter.
- **Fixed pulse amplitude:** E0 and E1 switch electrodes between off and a fixed 60 uA, and E2
  scales up to 60 uA, rather than tuning current to each electrode's threshold.
- **Scene networks:** E2 uses Depth Anything V2 Small (relative depth, so "near" means near for
  that image) and SegFormer-B0 trained on ADE20K photos, both run at reduced resolution on CPU.
- **Obstacle labels:** an NYU frame's label is the left, center, or right third whose lower
  two thirds are at least 15% nearer than 1.5 m (the most blocked third wins), otherwise
  "none"; this is a rule on the depth map, not a human annotation.
- **Object subset:** the object task is 10 COCO categories that are most often the largest
  object in a val2017 image, 70 images each, cropped to that object, so it is small and easier
  than full scene recognition.
- **Learned encoder inputs:** E3 shrinks images so the long side is at most 48 px, and it is
  trained only on 256- and 1024-electrode grids with default patients.

Evaluation (Phase 3):

- **Virtual observer:** task accuracy comes from a small CNN trained on each encoder's percepts,
  which measures how much task information survives but not how a person would learn to read
  phosphenes.
- **Charge proxy:** "total current per frame" sums the currents of active electrodes and stands
  in for charge, assuming every electrode fires the same pulse train.
- **Known map during benchmarking:** Phase 3 encoders use each patient's true phosphene map, as
  if calibration were perfect; Phase 4 removes that assumption.

Calibration (Phase 4):

- **Population knowledge:** calibration knows the population's psychometric slope, lapse rate,
  pointing noise, and how much patients vary, but nothing about the specific patient.
- **Query cost:** every query is one stimulation plus a pointing answer ("nothing" counts as an
  answer) and is assumed to take 5 s of clinic time.
- **Correlated deviations:** the Phase 4 patients deviate from the plan mostly through how the
  whole array landed (a shift and a small rotation per hemisphere) and a smooth excitability
  field over cortex; the sizes of these effects are plausible guesses, not measurements.

## Acknowledgements

Reference percepts use [pulse2percept](https://github.com/pulse2percept/pulse2percept) (BSD) and
[dynaphos](https://github.com/neuralcodinglab/dynaphos) (GPL-3.0; default parameters copied into
`configs/dynaphos_params.yaml`).
