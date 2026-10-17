# How it works: Phase 0, the reference simulators

Before building our own simulator, we installed two published ones so we have something to
check ours against.

**What a "percept" is.** A cortical implant puts electrodes into the visual cortex (V1). Running
current through one electrode makes the person see a small spot of light called a *phosphene*.
A simulator turns a list of electrode currents into a picture of what the person would see.
That picture is the percept.

**pulse2percept** (Beyeler lab) is a general prosthetic-vision simulator. We used its cortical
"scoreboard" model with the Orion implant (60 electrodes). In the scoreboard model each electrode
lights up a blob whose size depends on how far the current spreads in tissue.

**dynaphos** (van der Grinten et al., 2024) is built for training encoders. It is
*differentiable*: PyTorch can compute how the percept would change if each electrode's current
changed slightly. Phase 2's learned encoder needs exactly that. It also models time: a
phosphene's brightness builds up over frames and fades with repeated stimulation.

**What the Phase 0 figure shows** (`results/phase0_reference.png`): in the dynaphos panel,
phosphenes far from the center of gaze are larger than central ones. That's *cortical
magnification*: V1 spends much more tissue on the center of vision, so one electrode covers a
tiny patch of central vision but a large patch of peripheral vision. Our Phase 1 simulator has
to reproduce this effect, and a test will check that it does.

**Why both?** They make different modeling choices. If our simulator roughly agrees with both,
that's evidence we haven't made a basic mistake. Where they disagree, we have to pick one
assumption and write it down.
