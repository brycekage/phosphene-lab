// One volunteer's run through the pilot: the rules, with no page code (tested in Node).
//
// Each block (task) runs its practice trials first, then its scored trials with every
// encoder shuffled together, so a volunteer cannot tell which encoder made a picture.
// Nothing is stored: the session lives in the page and is gone when the tab closes.

// A small seeded random generator (mulberry32), so tests can fix the order.
export function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(items, random) {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export class PilotSession {
  constructor(trials, blocks, random = Math.random) {
    this.order = [];
    this.blockStarts = new Set();
    for (const task of blocks) {
      this.blockStarts.add(this.order.length);
      const practice = trials.filter((t) => t.task === task && t.practice);
      const scored = trials.filter((t) => t.task === task && !t.practice);
      this.order.push(...practice, ...shuffle(scored, random));
    }
    this.position = 0;
    this.responses = [];
    this.introShown = new Set();
  }

  get finished() {
    return this.position >= this.order.length;
  }

  get current() {
    return this.finished ? null : this.order[this.position];
  }

  get last() {
    return this.responses.at(-1) ?? null;
  }

  // True until the volunteer has seen the instructions for the block about to start.
  get atBlockStart() {
    return this.blockStarts.has(this.position) && !this.introShown.has(this.position);
  }

  startBlock() {
    this.introShown.add(this.position);
  }

  answer(choice) {
    const trial = this.current;
    if (trial === null) throw new Error("the session is finished");
    const correct = choice === trial.label;
    this.responses.push({ trial, choice, correct });
    this.position += 1;
    return correct;
  }

  // [answered, total] for the trials of the current block that match `practice`.
  #count(practice) {
    const task = (this.current ?? this.order.at(-1)).task;
    const total = this.order.filter((t) => t.task === task && t.practice === practice).length;
    const done = this.responses.filter(
      (r) => r.trial.task === task && r.trial.practice === practice,
    ).length;
    return [done, total];
  }

  progress() {
    return this.#count(false);
  }

  practiceProgress() {
    return this.#count(true);
  }

  // task -> encoder -> [correct, answered], scored trials only, encoders in order.
  summary() {
    const out = {};
    for (const { trial, correct } of this.responses) {
      if (trial.practice) continue;
      const cells = (out[trial.task] ??= {});
      const cell = (cells[trial.encoder] ??= [0, 0]);
      cell[0] += Number(correct);
      cell[1] += 1;
    }
    for (const task of Object.keys(out)) {
      const sorted = Object.entries(out[task]).sort(([a], [b]) => (a < b ? -1 : 1));
      out[task] = Object.fromEntries(sorted);
    }
    return out;
  }
}
