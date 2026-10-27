// The pilot page's session rules (app/pilot_site/session.js), run by tests/test_pilot_site.py.
import assert from "node:assert/strict";
import test from "node:test";

import { PilotSession, assetUrl, seeded } from "../../app/pilot_site/session.js";

const ENCODERS = ["E0", "E2", "E3"];
const BLOCKS = ["letters", "obstacles"];

function makeTrials() {
  const trials = [];
  for (const [task, labels] of [["letters", ["a", "b", "c"]], ["obstacles", ["left", "none"]]]) {
    trials.push({ task, encoder: "E0", label: labels[0], image: `${task}_p.png`,
                  original: "o.png", practice: true });
    for (const encoder of ENCODERS) {
      for (const label of labels) {
        trials.push({ task, encoder, label, image: `${task}_${encoder}_${label}.png`,
                      original: "o.png", practice: false });
      }
    }
  }
  return trials;
}

test("each block runs its practice first, then its scored trials shuffled", () => {
  const order = new PilotSession(makeTrials(), BLOCKS, seeded(1)).order;
  assert.deepEqual(order.slice(0, 10).map((t) => t.task), Array(10).fill("letters"));
  assert.ok(order[0].practice && !order.slice(1, 10).some((t) => t.practice));
  assert.ok(order[10].practice && order[10].task === "obstacles");
  assert.equal(new Set(order.map((t) => t.image)).size, makeTrials().length);
  const other = new PilotSession(makeTrials(), BLOCKS, seeded(2)).order;
  assert.notDeepEqual(other.map((t) => t.image), order.map((t) => t.image));
});

test("the same seed gives the same order", () => {
  const a = new PilotSession(makeTrials(), BLOCKS, seeded(7)).order.map((t) => t.image);
  const b = new PilotSession(makeTrials(), BLOCKS, seeded(7)).order.map((t) => t.image);
  assert.deepEqual(a, b);
});

test("an intro is due at the start of each block, once", () => {
  const session = new PilotSession(makeTrials(), BLOCKS, seeded(0));
  assert.ok(session.atBlockStart);
  session.startBlock();
  assert.ok(!session.atBlockStart);
  for (let i = 0; i < 10; i++) session.answer("a");
  assert.ok(session.atBlockStart && session.current.task === "obstacles");
});

test("scoring ignores practice and counts per encoder", () => {
  const session = new PilotSession(makeTrials(), BLOCKS, seeded(0));
  while (!session.finished) {
    const trial = session.current;
    const choice = trial.encoder === "E2" ? trial.label : "zzz";
    const correct = session.answer(choice);
    assert.equal(correct, trial.encoder === "E2");
    assert.deepEqual(session.last, { trial, choice, correct });
  }
  const summary = session.summary();
  assert.deepEqual(summary.letters, { E0: [0, 3], E2: [3, 3], E3: [0, 3] });
  assert.deepEqual(summary.obstacles.E2, [2, 2]);
  assert.deepEqual(Object.keys(summary.letters), ["E0", "E2", "E3"]);
  assert.equal(session.current, null);
  assert.throws(() => session.answer("a"), /finished/);
});

test("progress counts scored trials within the current block", () => {
  const session = new PilotSession(makeTrials(), BLOCKS, seeded(0));
  session.startBlock();
  assert.deepEqual(session.progress(), [0, 9]); // practice does not count
  session.answer("a"); // the practice trial
  assert.deepEqual(session.progress(), [0, 9]);
  session.answer("a");
  assert.deepEqual(session.progress(), [1, 9]);
  assert.deepEqual(session.practiceProgress(), [1, 1]);
});

test("asset URLs carry a private Space's sign token, and nothing else", () => {
  assert.equal(assetUrl("stimuli/a.png", ""), "stimuli/a.png");
  assert.equal(assetUrl("stimuli/a.png", "?x=1"), "stimuli/a.png");
  // Passed on exactly as Hugging Face wrote it (a JWT: letters, digits, "-", "_", ".").
  assert.equal(assetUrl("stimuli/a.png", "?x=1&__sign=eyJ.a-b_c"), "stimuli/a.png?__sign=eyJ.a-b_c");
});
