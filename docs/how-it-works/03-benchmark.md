# How it works: Phase 3, the benchmark

Pictures of percepts (Phase 2) are persuasive but subjective. The benchmark turns "which
encoder is better?" into numbers, and asks how the answer changes with the number of
electrodes, with dead electrodes, and with how different patients are.

## The virtual observer (`eval/observer.py`)

We cannot ask a real person to look at thousands of percepts, so a small neural network plays
the patient. For each encoder we render percepts of the training images, train a fresh
classifier on them, and test it on percepts of images it has never seen. If the classifier can
tell the letters (or objects, or obstacle positions) apart, the information survived the trip
through the implant.

Each encoder gets *its own* observer. That matters: a learned encoder might invent a code that
looks nothing like the original image yet is perfectly readable once learned, and real patients
do learn. Judging every encoder with one fixed reader would punish that.

## One benchmark cell (`eval/benchmark.py`)

1. Make a virtual patient: an electrode grid of a given size, a dropout rate (fraction of dead
   electrodes), and a variability level (how far thresholds, sizes and positions stray).
2. The encoder turns every image into currents, using the patient's phosphene map.
3. The simulator renders what the patient sees, always with the patient's *true* map.
4. Train the observer on training percepts; score accuracy on test percepts.
5. Also record: **SSIM** between the percept and E2's simplified target (does it look like the
   intended picture?), **active electrodes per frame**, and **total current per frame** (a
   stand-in for charge, which drives power use and tissue safety).

Within a seed, every encoder sees the *same* patient, so differences between encoders are not
luck of the draw. Three seeds give a mean and spread.

## The sweep (`configs/benchmark.yaml`)

- electrode counts 64, 256, 1024, 1600, with 0, 10, and 30% dead electrodes;
- patient variability low / default / high, at 256 electrodes;
- 3 tasks × 4 encoders × 3 seeds for each of those.

The full cross of everything would not fit the one-hour compute budget on a laptop CPU, so
variability is swept at one array size, the letter task uses 50 training and 20 test images
per letter, and the observer is a small network on 32 × 32 percepts. Those choices are in the
config and in the README's budget notes.

E3 was trained only on 256- and 1024-electrode arrays and default patients, so the 64- and
1600-electrode and high-variability cells also test whether it generalizes.

## Results

`scripts/run_benchmark.py` writes every cell to `results/benchmark.csv`, a mean ± std table to
`results/benchmark_summary.csv`, and three figures: accuracy vs electrode count (the headline),
vs dropout, and vs variability. The dashed line in each panel is chance.
