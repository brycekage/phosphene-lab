# How it works: Phase 4, fast calibration

After surgery nobody knows exactly where each electrode's phosphene appears, how much current
it needs, or whether it works at all. The surgical plan gives a first guess (from the
retinotopic map), but the real array lands a little shifted and turned, and tissue differs.
**Calibration** measures the truth by asking the patient questions: "stimulate electrode 37 at
40 uA. Did you see something? Point to it." Each question takes clinic time (we assume 5 s),
and patients tire, so the goal is a good map from as few questions as possible.

All code is in `src/phosphene_lab/calibration/`. Calibration never reads the virtual patient's
hidden truth; it only uses the answers. (A test enforces this: the truth raises an error if
touched mid-calibration.)

## What the system believes (`model.py`)

- **Thresholds.** For each electrode we keep a probability for every possible threshold from 1
  to 100 uA, plus a probability that the electrode is dead. Each yes/no answer updates these
  with the known shape of the "seen / not seen" curve, including occasional random answers.
- **Locations.** We model how far each electrode's true cortical position is from the planned
  one, as a **Gaussian process**: a statistical model that says nearby electrodes tend to be off
  in the same direction. When the patient points at one phosphene, the model also moves its
  neighbors' estimates, and it becomes more confident about them. We do this in cortical
  millimeters rather than visual degrees, because a 1 mm surgical shift is one number on
  cortex but a tiny shift near the center of gaze and a large one in the periphery.

## Three ways to choose the next question

- **Exhaustive** (`baselines.py`): go through the electrodes in order. For each one, five
  questions home in on the threshold (bisection), then three more at a clearly visible current
  pin down the location. This is the thorough clinical protocol.
- **Random order:** the same per-electrode protocol, but each question goes to a random
  electrode, so a partial session covers the whole array thinly instead of the first rows well.
- **Active** (`active.py`): before every question, score every electrode at every candidate
  current by **expected information gain**: how much, on average, the answer would reduce
  our uncertainty about that electrode's threshold, plus (if it is likely to be seen) its
  location. Ask the best question. The active method also shares threshold evidence between
  neighbors through a second Gaussian process, using "leave-one-out" predictions so no answer
  is counted twice.

## The honest catch

Sharing only helps if neighbors really are off in the same way. We therefore test two kinds of
virtual patient:

- **correlated:** most of the error comes from where the whole array landed, plus a smooth
  excitability pattern over cortex. This is our best guess for a rigid array.
- **independent:** every electrode is off on its own (the Phase 1-3 default). Here there is
  little for the Gaussian process to share, and the active method can only win by asking
  better questions.

## Closing the loop

A better map only matters if the patient sees better. `scripts/closed_loop.py` reruns the
Phase 3 benchmark for E2 and E3 while the encoders use (a) the true map, (b) the uncalibrated
surgical-plan guess, and (c) the active method's map after 100, 300, and 1000 questions.
Percepts are always rendered with the *true* map, because that is what the patient actually
sees. One subtlety: because every encoder stretches the image over the phosphenes it believes
in, an error that moves *every* phosphene the same way costs nothing. Only errors that differ
between electrodes scramble the picture.

## What closing the loop showed (and why it is a limitation)

Task accuracy did **not** change with calibration: the uncalibrated guess scored as well as
the true map, for every task and encoder. The maps really did improve, though, and SSIM (does
the percept look like the intended picture?) recovered most of its loss within the first 100
questions. The explanation is in how we score: each encoder gets a fresh observer trained on
*this patient's* percepts, so it learns any distortion that stays the same from frame to
frame, like a patient who adapts perfectly. Calibration matters most before a patient has
adapted, or when the map has to be right the first time (pointing at an obstacle, reaching for
a cup), and our observer-accuracy metric does not capture that. A fixed observer, or a
location-based task, would be the next step.

## Results

`results/phase4_calibration.png` shows error against number of questions (with clinic minutes
on the top axis); `results/phase4_queries_to_target.csv` lists how many questions each method
needs to match one full exhaustive pass; `results/phase4_closed_loop_summary.csv` has task
accuracy per map.
