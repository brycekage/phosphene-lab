// Phase 6 pilot page: welcome -> (per part: instructions -> practice with the answer shown ->
// scored trials) -> results. The three encoders' pictures are shuffled together and never
// named until the results. Nothing is stored or sent anywhere: the page only reads its own
// pictures, and the answers are gone when the tab closes.
import { PilotSession } from "./session.js";

const BLOCKS = ["letters", "obstacles"];
const LETTERS = "abcdefghijklmnopqrstuvwxyz".split("");
const OBSTACLE_BUTTONS = [ // [label, answer, key]
  ["← Left", "left", "arrowleft"],
  ["↑ Ahead", "center", "arrowup"],
  ["→ Right", "right", "arrowright"],
  ["Nothing close", "none", "space"],
];
const OBSTACLE_WORDS = { left: "on the left", center: "ahead", right: "on the right",
                         none: "nothing close" };
const CHANCE = { letters: 1 / 26, obstacles: 1 / 4 };
const ENCODER_NAMES = { // [name, one-line description]
  E0: ["E0 · Brightness", "lights each dot by how bright the camera image is there"],
  E1: ["E1 · Edges", "lights the outlines of things"],
  E2: ["E2 · Scene", "neural networks find what matters and draw only that"],
  E3: ["E3 · Learned", "a network trained through the simulator"],
};
const PART = {
  letters: {
    title: "Reading letters",
    prompt: "Which letter is this?",
    noun: "Letter",
    body: `<p class="ph-lead">You will see a handwritten letter the way an implant might draw
      it: a few dots of light. Pick the letter you think it is. Letters may be capital or
      lowercase.</p>
      <p class="ph-note">Shortcut: type the letter on your keyboard.</p>`,
  },
  obstacles: {
    title: "Spotting obstacles",
    prompt: "Where is the nearest obstacle?",
    noun: "Scene",
    body: `<p class="ph-lead">You will see a room the way an implant might draw it, as if you
      were standing in it. Say where something is blocking your path within about 1.5 meters,
      or that nothing is close.</p>
      <p class="ph-note">Shortcuts: <span class="ph-kbd">←</span> <span class="ph-kbd">↑</span>
      <span class="ph-kbd">→</span> and <span class="ph-kbd">Space</span> for nothing
      close.</p>`,
  },
};

const $ = (id) => document.getElementById(id);
const SCREENS = ["welcome", "intro", "trial", "feedback", "results"];
let trials = [];
let session = null;
let started = 0;

function show(screen, focus) {
  for (const name of SCREENS) $(name).hidden = name !== screen;
  window.scrollTo(0, 0);
  (focus ?? $(screen).querySelector("h1, h2"))?.focus();
}

const picture = (path) => `stimuli/${path}`;

function answerWords(task, answer) {
  return task === "letters" ? answer.toUpperCase() : OBSTACLE_WORDS[answer];
}

function makeButtons() {
  for (const ch of LETTERS) {
    const b = Object.assign(document.createElement("button"), { textContent: ch.toUpperCase() });
    b.className = `ph-button ph-key-${ch}`;
    b.addEventListener("click", () => onAnswer(ch));
    $("letters-grid").append(b);
  }
  for (const [label, value, key] of OBSTACLE_BUTTONS) {
    const b = Object.assign(document.createElement("button"), { textContent: label });
    b.className = `ph-button ph-key-${key}`;
    b.addEventListener("click", () => onAnswer(value));
    $("obstacles-grid").append(b);
  }
}

function setProgress(trial) {
  const part = BLOCKS.indexOf(trial.task) + 1;
  let label, pct;
  if (trial.practice) {
    const [done, total] = session.practiceProgress();
    [label, pct] = [`Practice ${done + 1} of ${total}`, 0];
  } else {
    const [done, total] = session.progress();
    [label, pct] = [`${PART[trial.task].noun} ${done + 1} of ${total}`, (100 * done) / total];
  }
  $("progress-label").textContent =
    `Part ${part} of ${BLOCKS.length} · ${PART[trial.task].title} · ${label}`;
  $("progress").setAttribute("aria-valuenow", pct.toFixed(0));
  $("progress").firstElementChild.style.width = `${pct}%`;
}

// The screen that comes next: results, a part's instructions, or a picture.
function advance() {
  if (session.finished) {
    $("results-body").innerHTML = resultsHtml((performance.now() - started) / 1000);
    show("results");
    return;
  }
  const trial = session.current;
  if (session.atBlockStart) {
    const info = PART[trial.task];
    $("intro-part").textContent = `Part ${BLOCKS.indexOf(trial.task) + 1} of ${BLOCKS.length}`;
    $("intro-title").textContent = info.title;
    $("intro-body").innerHTML = info.body;
    show("intro");
    return;
  }
  setProgress(trial);
  $("prompt").textContent = PART[trial.task].prompt;
  const img = $("stimulus");
  img.src = picture(trial.image);
  img.alt = `Picture ${session.position + 1}, drawn in dots of light`;
  $("letters-grid").hidden = trial.task !== "letters";
  $("obstacles-grid").hidden = trial.task !== "obstacles";
  show("trial", $("prompt"));
  const next = session.order[session.position + 1]; // fetch the next picture in advance
  if (next) new Image().src = picture(next.image);
}

function onAnswer(choice) {
  if (!session || session.finished || $("trial").hidden) return;
  session.answer(choice);
  const { trial, correct } = session.last;
  if (!trial.practice) {
    advance();
    return;
  }
  const right = answerWords(trial.task, trial.label);
  const message = $("feedback-message");
  message.className = `ph-callout ${correct ? "ph-feedback-right" : "ph-feedback-wrong"}`;
  message.innerHTML = correct
    ? `Correct: it was <b>${right}</b>.`
    : `Not quite: it was <b>${right}</b> (you answered ${answerWords(trial.task, choice)}).`;
  $("seen").src = picture(trial.image);
  $("seen").alt = "The dots you were shown";
  $("truth").src = picture(trial.original);
  $("truth").alt = `The camera image: ${right}`;
  show("feedback", message);
}

function resultsHtml(seconds) {
  const minutes = Math.round(seconds / 60);
  const took = minutes < 1 ? "under a minute" : `${minutes} minute${minutes === 1 ? "" : "s"}`;
  const tables = Object.entries(session.summary()).map(([task, cells]) => {
    const rows = Object.entries(cells).map(([enc, [correct, n]]) =>
      `<tr><td><b>${ENCODER_NAMES[enc][0]}</b><br><span class="ph-note">${ENCODER_NAMES[enc][1]}`
      + `</span></td><td><b>${correct} / ${n}</b> &nbsp;${Math.round((100 * correct) / n)}%`
      + "</td></tr>").join("");
    return `<p class="ph-section-title">${PART[task].title} · guessing scores about `
      + `${Math.round(100 * CHANCE[task])}%</p><table class="ph-results"><thead><tr>`
      + `<th>Drawing method</th><th>Your score</th></tr></thead><tbody>${rows}</tbody></table>`;
  });
  return `<header class="ph-header"><p class="ph-eyebrow">Phosphene Lab · volunteer pilot</p>
    <h2 tabindex="-1">Thank you! Here is how you did</h2>
    <p class="ph-lead">You finished in ${took}. Each row is one way of turning a camera image
    into dots; you saw them mixed together without knowing which was which.</p></header>
    ${tables.join("")}
    <p class="ph-note">With 8 to 10 pictures per method, a difference of one or two pictures
    can be luck. The pilot looks for patterns across many volunteers.</p>
    <div class="ph-callout">Nothing has been saved. Closing this tab erases your answers.</div>`;
}

// Keyboard shortcuts: a key clicks the visible button carrying the matching ph-key-* class.
// A short lockout stops a held key from answering the next picture before it is seen.
let locked = false;
document.addEventListener("keydown", (e) => {
  if (locked || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
  const name = e.key === " " ? "space" : e.key.toLowerCase();
  if (!/^([a-z]|arrow(left|right|up)|space|enter)$/.test(name)) return;
  if (name === "enter" && e.target instanceof HTMLButtonElement) return; // already a click
  const button = [...document.querySelectorAll(`.ph-key-${name}`)]
    .find((b) => b.offsetParent !== null && !b.disabled);
  if (!button) return;
  e.preventDefault();
  locked = true;
  setTimeout(() => { locked = false; }, 350);
  button.click();
});

$("begin").addEventListener("click", () => {
  session = new PilotSession(trials, BLOCKS);
  started = performance.now();
  advance();
});
$("start-part").addEventListener("click", () => {
  session.startBlock();
  advance();
});
$("next").addEventListener("click", advance);
$("again").addEventListener("click", () => {
  session = null;
  show("welcome");
});

makeButtons();
fetch("stimuli/manifest.json")
  .then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then((manifest) => {
    trials = manifest.trials;
    Object.assign($("begin"), { disabled: false, textContent: "Begin" });
  })
  .catch((err) => {
    $("begin").textContent = "Unavailable";
    $("load-error").hidden = false;
    $("load-error").textContent =
      `The pictures could not be loaded (${err.message}). Please reload the page.`;
  });
