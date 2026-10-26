# How it works: Phase 6, the volunteer pilot page

Every result so far was scored by a computer program, the *virtual observer*: a small network
trained to read percepts. The obvious question is whether people agree with it. If E2 and E3
beat E0 for the observer, do they also help a human? The pilot page lets sighted volunteers
answer that, in about ten minutes, in a web browser.

`uv run python app/pilot.py` opens it at http://127.0.0.1:7861.

## What a volunteer does

1. **Welcome.** What the test is, how long it takes, and that nothing is saved.
2. **Part 1, reading letters.** A handwritten letter (from the EMNIST set the benchmark uses)
   is shown the way a simulated implant would draw it. The volunteer clicks or types the letter.
3. **Part 2, spotting obstacles.** A photo of a room (from NYU Depth V2) is shown the same way.
   The volunteer says where the nearest obstacle is (left, ahead, right) or that nothing is
   close.
4. **Results.** Their score for each drawing method, next to what guessing would score.

Each part starts with three practice pictures. After a practice answer, the page shows the
right answer and the camera image, so the volunteer learns what the dots mean. The scored
pictures come without feedback.

## Making it a fair comparison

- **Blind to the method.** E0, E2, and E3 pictures are shuffled together, and the method is only
  named on the results screen, so a volunteer cannot try harder on one of them.
- **Same questions, different pictures.** Every method gets the same letters (10 of them) or
  the same obstacle classes (2 of each), but different photos of them, so no one sees a picture
  twice and learns it.
- **Same patient as the benchmark.** All pictures come from one benchmark condition: 256
  electrodes, 10% dead, a patient E3 never trained on. Each image is encoded exactly as in the
  benchmark, then drawn at a higher resolution, because the benchmark's 32-pixel percepts are
  too small to look at.
- **Checked by eye.** Every letter was compared with its label before use. One EMNIST image is
  labeled "n" but written as "m"; the config excludes it and says why.

The pictures are made once by `scripts/make_pilot_stimuli.py` and saved in
`results/pilot_stimuli/`, so the page needs no datasets and no neural networks. That makes it
start in seconds, and it means every volunteer sees exactly the same set.

## What it does not do (yet)

The page saves nothing. Answers live in the browser tab and disappear when it closes. Saving
answers would be collecting human data, which needs approval from the CMC IRB (the college's
ethics board) before any result can be published. So, for now, the page is a working,
polished instrument that is ready for that step, and the numbers it shows are only for the
person who just played.

Two honest limits apply even after approval. First, sighted volunteers looking at a screen are
not blind implant users: they have never learned to read phosphenes, and a screen shows
grayscale blobs, not what a phosphene looks like. Second, the obstacle answers come from depth
sensors (the nearest blocked third of the floor within 1.5 m), which a person standing in the
room might judge differently.
