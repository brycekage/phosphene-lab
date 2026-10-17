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

## Acknowledgements

Reference percepts use [pulse2percept](https://github.com/pulse2percept/pulse2percept) (BSD) and
[dynaphos](https://github.com/neuralcodinglab/dynaphos) (GPL-3.0; default parameters copied into
`configs/dynaphos_params.yaml`).
