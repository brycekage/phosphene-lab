# How it works: Phase 1, the phosphene simulator

The simulator answers one question: *if we send these currents to these electrodes, what does
the patient see?* It has four parts, each in `src/phosphene_lab/simulator/`.

## 1. Retinotopy (`retinotopy.py`): where on the cortex is each part of the visual field?

V1 is a map of the visual field, but a heavily distorted one. The center of gaze gets a huge
share of cortex: about 10 mm of cortex per degree near the fovea, against under 2 mm per degree
at 8 degrees out. That ratio is the **cortical magnification**. We use a standard formula, the
*wedge-dipole* map, which converts a visual-field point into a cortex position in mm and back.
The left visual field lands on the right hemisphere and vice versa.

*Check:* our map gives exactly the same numbers as dynaphos's implementation of the same formula,
and its magnification stays within 10% of Horton & Hoyt's human measurements.

## 2. Electrode arrays (`array.py`): where are the electrodes?

Electrodes are placed on a rectangle of cortex in each hemisphere, either as a regular grid (like
a Utah array) or at random sites with a minimum spacing (like flexible threads). Thread mode
takes an *exclusion mask*, so electrodes can avoid blood vessels; Phase 7 plugs in the vessel
planner here. Because of the distorted map, a rectangle on cortex becomes a **bowtie** in the
visual field: dense near the center, sparse in the periphery.

## 3. The renderer (`phosphene.py`): currents in, image out

For each electrode:

- **Where:** the phosphene appears at the electrode's visual-field location.
- **How big:** more current activates a bigger patch of cortex. The same patch covers more of
  the visual field in the periphery (low magnification), so peripheral phosphenes are larger.
- **How bright:** nothing below the electrode's threshold, then brighter with more current,
  leveling off. Zero current is exactly black.
- **Over time:** each electrode keeps a running total of recent charge. Keep stimulating and the
  phosphene fades (adaptation); stop, and it recovers within a few seconds.

Overlapping phosphenes add up, so neighbors merge into one blob. The whole thing is
**differentiable**: PyTorch can tell how the image would change if any current changed. That is
what lets Phase 2 train an encoder through the simulator.

*Speed:* 1600 electrodes x 32 images renders forward and backward in about 0.3 s on a laptop
CPU. That's possible because a round Gaussian splits into an x part and a y part, which turns
the whole image into a single matrix multiply.

## 4. Virtual patients (`patient.py`): every brain is different

A `VirtualPatient` secretly differs from the textbook prediction. Phosphenes are shifted (the
electrodes are not exactly where planned), thresholds vary from electrode to electrode, sizes
vary, and some electrodes are dead. You can only learn about a patient the way a clinician
does:

- *"Did you see a flash?"* The answer is more likely "yes" at higher current, and sometimes
  random (patients lapse).
- *"Point to where you saw it."* The answer lands near the true spot, plus noise.

Phase 4's calibration has to recover the hidden map from as few of these questions as possible.

## How it compares to dynaphos

`scripts/compare_simulators.py` gives both simulators the same 256 phosphene locations and the
same current (`results/phase1_vs_dynaphos.png`, numbers in `results/phase1_vs_dynaphos.csv`):

- Locations and orientation line up.
- Phosphene size versus eccentricity agrees to within 0.5%. That's expected, because we use the
  same size equation; it confirms the implementation, not the biology.
- The images correlate at 0.66, and ours is brighter on average (mean 0.056 vs 0.013). Most of
  the gap comes from the smallest, central phosphenes. They are far smaller than a pixel, and
  dynaphos draws them at their true size, so they nearly vanish. We deliberately draw them at
  least half a pixel wide so every working electrode stays visible (see *Assumptions* in the
  README).
