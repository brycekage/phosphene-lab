# phosphene-lab

An end-to-end sandbox for encoding and calibrating cortical visual prostheses.
Work in progress; see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the plan.

## Quickstart

```bash
uv sync
uv run pytest
uv run python scripts/render_reference.py --quick   # writes results/phase0_reference.png
```

## How it works

- [Phase 0: the reference simulators](docs/how-it-works/00-setup.md)
- [Phase 1: the phosphene simulator](docs/how-it-works/01-simulator.md)

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

## Acknowledgements

Reference percepts use [pulse2percept](https://github.com/pulse2percept/pulse2percept) (BSD) and
[dynaphos](https://github.com/neuralcodinglab/dynaphos) (GPL-3.0; default parameters copied into
`configs/dynaphos_params.yaml`).
