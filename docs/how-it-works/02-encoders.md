# How it works: Phase 2, encoders

An **encoder** is the software between the camera and the implant. Every frame it decides which
electrodes to stimulate and how hard. The implant can only draw a few hundred blurry dots, so
the encoder's real job is choosing *what is worth drawing*. All four encoders live in
`src/phosphene_lab/encoders/` and share the plumbing in `base.py`.

## The shared plumbing (`base.py`)

1. **Place the image where the implant can draw.** The phosphenes cover a bowtie-shaped patch
   of the visual field (Phase 1), with a gap straight above and below where you're looking. We
   stretch the camera image over the smallest box that holds every *working* phosphene, so no
   part of the picture lands where nothing can be shown. The gap is still there, as a thin
   vertical strip through the middle of the box.
2. **Read the image at each phosphene.** For each electrode we look up the image value where
   its phosphene appears (bilinear interpolation, which keeps everything differentiable).
3. **Respect the hardware.** Dead electrodes are switched off, currents are capped, and at most
   `max_active` electrodes fire at once (the strongest win).

## E0, naive (`naive.py`)

Bright spot in the image means the electrode is on. That is how early prostheses worked. It
is a fair test of what raw brightness gets you: fine for a white letter on black, poor for a
cluttered room, where texture lights up everything.

## E1, edges (`edges.py`)

Only object outlines, found with a Sobel filter and slightly blurred so thin lines are not
missed between phosphenes. Outlines carry most of an object's shape in far fewer dots, but they
also outline every bit of texture.

## E2, scene simplification (`scene.py`)

Two pretrained networks describe the scene: **Depth Anything V2** estimates how near each pixel
is, and **SegFormer**, trained on the ADE20K dataset, labels pixels as wall, floor, door,
person, chair, and so on. A rule then draws only what matters for the current mode:

- **navigation:** near obstacles filled in (nearer is brighter), the edge of the floor and door
  frames outlined, walls and floor left dark;
- **people:** people filled in, nearby obstacles faintly outlined;
- **reading:** just the strokes, whichever of light-on-dark or dark-on-light is the minority.

We used ADE20K instead of the spec's example (DeepLabV3 on the VOC dataset) because VOC has no
wall, floor, or door labels, and navigation needs them. The simplified image E2 produces is
also the "target picture" E3 learns to draw.

## E3, learned (`learned.py`)

Instead of hand-writing a rule, we let a small neural network learn one. It works because the
Phase 1 simulator is differentiable, so we can ask "if this electrode's current went up a
little, would the patient's percept look more like the target?" and nudge the network that way.

- **Inputs:** the image plus two extra channels describing *this patient*: where their
  phosphenes are, and how much current each needs.
- **Output:** a "stimulation map" over the image, read at each phosphene's location. Because
  of that design the same network works for 64, 256, 1024, or 1600 electrodes, in any layout.
- **Training signal:** three goals at once. The percept should look like E2's simplified
  scene; a small classifier trained alongside should recognize what is shown (the letter, the
  object, where the obstacle is); and it should use as little current as possible.
- **Generalizing across patients:** every training step draws a brand-new virtual patient, so
  the network cannot memorize one person's phosphene map.

This follows earlier work that trained encoders through differentiable phosphene models
(Granley et al., NeurIPS 2022; de Ruyter van Steveninck et al., 2022). The idea is not new; the
point here is to have it working inside one benchmark, next to the simple baselines.

## The data

`scripts/download_data.py` fetches three datasets, one per task:

- **EMNIST letters** for reading (26 classes);
- **10 COCO categories** for object recognition, each image cropped to its main object;
- **NYU Depth V2** indoor scenes for navigation, labeled "obstacle on the left / center /
  right / none" by a rule on the depth map (anything closer than 1.5 m in the lower part of
  the frame). Frames from the same room never appear in both training and testing.

Two data bugs were caught while building this phase. EMNIST's test set is sorted by letter, so
"the first 1000 test images" contained only two letters; every dataset is now shuffled when it
is saved. The obstacle rule at 1.0 m labeled 78% of frames "none", which would have rewarded an
encoder for showing nothing, so the threshold moved to 1.5 m.

## Results

`scripts/encoder_gallery.py` renders the side-by-side figure (`results/phase2_gallery.png`):
for sample test images, the placed original, E2's target, and what a held-out virtual patient
would see under E0-E3, at 256 and 1024 electrodes. `scripts/train_e3.py` records E3's training
time and final losses in `results/phase2_e3_training.csv`. Phase 3 turns these pictures into
numbers.
