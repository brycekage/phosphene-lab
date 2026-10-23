# How it works: Phase 5, the interactive demo

The first four phases produced numbers and figures. This phase makes the same machinery
something you can play with: `uv run python app/app.py` opens a web page (built with Gradio)
with two tabs. Nothing in the demo is new science; it runs exactly the simulator, encoders, and
calibration code that produced the results, so what you see is what was measured.

All of the logic lives in `src/phosphene_lab/demo/` and is tested without a browser;
`app/app.py` only lays out the page and passes clicks and slider values through.

## The viewer

You give it a picture (one of the samples, your own photo, or a webcam snapshot), and it shows
three things side by side:

1. **The camera image.**
2. **What the encoder draws.** Each encoder decides what part of the picture is worth showing:
   the naive encoder (E0) uses brightness, the edge encoder (E1) uses outlines, and the scene
   encoder (E2) uses two pretrained networks to find what matters for the chosen *priority
   mode* (nearby obstacles for navigation, people, or text for reading). The learned encoder
   (E3) draws whatever its training taught it. The picture is shown where it lands in the
   visual field: stretched over the region the implant can reach.
3. **The phosphene percept.** What a simulated implant user would see: one glowing dot per
   working electrode that was switched on, each with the position, size, and brightness the
   simulator gives it.

The sliders change the implant and the patient: how many electrodes, what fraction are dead,
and how long the same image has been shown. That last one is **adaptation**: a phosphene that
is driven steadily fades over a few seconds, which is why real implants struggle with still
scenes.

The point the viewer makes quickly is the one the benchmark makes slowly: with a few hundred
dots, *what you choose to draw* matters more than drawing the picture faithfully. Feeding the
naive encoder a cluttered photo lights up whatever happens to be bright; the scene encoder
shows the person and nothing else.

## The calibration game

Here **you are the patient**. A hidden virtual patient has a real phosphene map that the system
does not know. Each round, the system picks one electrode and a current (using the active
calibration from Phase 4: the question it expects to teach it the most), the flash appears
where that patient would really see it, and you click on it. If you see nothing (the current
may be below that electrode's threshold, and near-threshold flashes are faint), you press
*Saw nothing*. That is an answer too: it tells the system the threshold is higher.

On the right, gray rings mark where the phosphenes really are, green dots mark where the system
currently thinks they are, and a red line joins each pair. As you answer, the red lines shrink,
and not only for the electrodes you were asked about: because the model knows that a misplaced
array shifts all its neighbors together, one answer corrects a whole region.

The counter compares you with the standard approach, a full exhaustive pass of 8 questions per
electrode (512 questions for this 64-electrode array). When your map is as accurate as that
pass would make it, the page tells you how many questions you saved. For the patient in the
recording above, that happened after 111 questions, saving about 33 minutes of clinic time at
5 s per question. Patients differ: on other virtual patients it can take as few as 10.

If you get tired of clicking, *Let the virtual patient answer 10* hands the next ten questions
to the simulated patient, who answers with the same noise and lapses as in Phase 4.

## What the demo cannot show

A computer screen shows grayscale blobs, not phosphenes; real phosphenes can flicker, have
color, and look different from person to person. And a sighted player clicking a dot on a
screen is far more precise than a blind patient pointing at a flash, so the game is easier
than the clinic. The Assumptions section of the README lists these simplifications.
