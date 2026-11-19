// Clinic Wait-Time Simulator. The trained network runs in plain JavaScript; no libraries.
const $ = (id) => document.getElementById(id);
const STAFF_COLORS = { 1: "#A5B0EE", 2: "#7E8CE0", 3: "#5566CC", 4: "#3A49B0", 5: "#1F2B78" }; // one hue, light to dark
let M; // model.json
const state = { hour: 11, day: "Mon", staff: 2, queue: 12, target: 45 };
const saved = [];

const relu = (x) => (x > 0 ? x : 0);
function predict(hour, day, staff, queue) {
  const raw = { hour, staff_on_shift: staff, patients_in_queue: queue };
  for (const d of M.days) raw["day_" + d] = d === day ? 1 : 0;
  let a = M.features.map((f, i) => (raw[f] - M.mean[i]) / M.std[i]);
  M.layers.forEach((L, li) => {
    const out = L.b.slice();
    for (let j = 0; j < out.length; j++) for (let i = 0; i < a.length; i++) out[j] += a[i] * L.W[i][j];
    a = li < M.layers.length - 1 ? out.map(relu) : out;
  });
  return Math.max(0, a[0]);
}
const fmtHour = (h) => (h === 12 ? "12 PM" : h > 12 ? h - 12 + " PM" : h + " AM");
const hm = (m) => { const r = Math.round(m); return r < 60 ? "" : `About ${Math.floor(r / 60)} h ${r % 60} min`; };
const setState = (patch) => { Object.assign(state, patch); render(); };

// ---------- 1. simulator ----------
function render() {
  const c = state;
  $("o-hour").textContent = fmtHour(c.hour);
  $("o-queue").textContent = c.queue;
  $("o-target").textContent = c.target + " min";
  $("hour").value = c.hour; $("queue").value = c.queue; $("target").value = c.target;
  document.querySelectorAll("#days button").forEach((b) => b.setAttribute("aria-checked", b.dataset.v === c.day));
  document.querySelectorAll("#staffseg button").forEach((b) => b.setAttribute("aria-checked", +b.dataset.v === c.staff));

  const p = predict(c.hour, c.day, c.staff, c.queue);
  $("pred").textContent = Math.round(p);
  $("hm").textContent = hm(p);
  const diff = p - M.baseline_wait;
  $("vs-base").textContent = `The median wait in the training data is ${M.baseline_wait} min. This scenario is ${Math.abs(Math.round(diff))} min ${diff >= 0 ? "longer" : "shorter"}.`;
  $("range-note").textContent = `On held-out visits the model is off by about ${M.metrics.model_mae} min on average, so treat this as an estimate.`;

  const preds = [1, 2, 3, 4, 5].map((s) => predict(c.hour, c.day, s, c.queue));
  const max = Math.max(...preds, 1);
  $("bars").innerHTML = preds.map((v, i) => {
    const s = i + 1;
    return `<button type="button" class="bar${s === c.staff ? " cur" : ""}" data-n="${s}" aria-pressed="${s === c.staff}" aria-label="${s} staff, ${Math.round(v)} minutes"><span>${s} staff</span><span class="track"><span class="fill" style="width:${(v / max) * 100}%;background:${STAFF_COLORS[s]}"></span></span><span class="v">${Math.round(v)} min</span></button>`;
  }).join("");
  const gains = preds.slice(0, 4).map((v, i) => v - preds[i + 1]);
  $("insight").textContent = c.queue === 0
    ? "With nobody waiting, staffing has little effect. Add patients to the queue to see the difference."
    : `Adding a second staff member saves about ${Math.round(gains[0])} min here. Adding a fifth saves only about ${Math.round(gains[3])} min. Each extra person helps less than the one before.`;

  // planner
  const ok = preds.findIndex((v) => v <= c.target);
  if (ok >= 0) {
    const s = ok + 1;
    $("plan").innerHTML = `<div>Smallest staff count that meets ${c.target} min: <b>${s}</b> (predicted ${Math.round(preds[ok])} min).</div>` +
      (s !== c.staff ? `<button type="button" class="btn ghost apply" data-n="${s}">Apply ${s} staff</button>` : `<div class="hint">Already applied.</div>`);
  } else {
    $("plan").innerHTML = `<div>No staffing level from 1 to 5 meets ${c.target} min for this queue. The best case is <b>${Math.round(preds[4])} min</b> with 5 staff.</div>`;
  }
  renderCmp();
}

function renderCmp() {
  const body = document.querySelector("#cmp tbody");
  body.innerHTML = saved.map((r, i) => `<tr><td>${r.day}</td><td>${fmtHour(r.hour)}</td><td>${r.staff}</td><td>${r.queue}</td><td class="num">${r.pred}</td><td><button type="button" data-i="${i}" aria-label="Remove scenario ${i + 1}">Remove</button></td></tr>`).join("");
  $("cmp-empty").hidden = saved.length > 0;
  $("cmp").hidden = saved.length === 0;
  $("clear-cmp").disabled = saved.length === 0;
  $("add-cmp").disabled = saved.length >= 8;
}

const PRESETS = [
  { n: "Monday morning rush", s: { day: "Mon", hour: 9, staff: 2, queue: 18 } },
  { n: "Quiet afternoon", s: { day: "Wed", hour: 15, staff: 3, queue: 3 } },
  { n: "Fully staffed", s: { day: "Thu", hour: 11, staff: 5, queue: 12 } },
  { n: "Understaffed", s: { day: "Tue", hour: 12, staff: 1, queue: 14 } },
];
function setupControls() {
  $("presets").innerHTML = PRESETS.map((p, i) => `<button type="button" data-i="${i}">${p.n}</button>`).join("");
  $("presets").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState(PRESETS[+b.dataset.i].s); });
  $("days").innerHTML = M.days.map((d) => `<button type="button" role="radio" data-v="${d}">${d}</button>`).join("");
  $("days").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState({ day: b.dataset.v }); });
  $("staffseg").innerHTML = [1, 2, 3, 4, 5].map((n) => `<button type="button" role="radio" data-v="${n}">${n}</button>`).join("");
  $("staffseg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState({ staff: +b.dataset.v }); });
  $("bars").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState({ staff: +b.dataset.n }); });
  $("plan").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState({ staff: +b.dataset.n }); });
  $("hour").addEventListener("input", (e) => setState({ hour: +e.target.value }));
  $("queue").addEventListener("input", (e) => setState({ queue: +e.target.value }));
  $("target").addEventListener("input", (e) => setState({ target: +e.target.value }));
  $("q-minus").addEventListener("click", () => setState({ queue: Math.max(0, state.queue - 1) }));
  $("q-plus").addEventListener("click", () => setState({ queue: Math.min(20, state.queue + 1) }));
  $("add-cmp").addEventListener("click", () => {
    if (saved.length >= 8) return;
    saved.push({ day: state.day, hour: state.hour, staff: state.staff, queue: state.queue, pred: Math.round(predict(state.hour, state.day, state.staff, state.queue)) });
    renderCmp();
  });
  $("clear-cmp").addEventListener("click", () => { saved.length = 0; renderCmp(); });
  document.querySelector("#cmp tbody").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { saved.splice(+b.dataset.i, 1); renderCmp(); } });
}

// ---------- 2. challenge ----------
const game = { you: 0, ai: 0, cur: null, done: false };
function newRound() {
  game.cur = M.test_rows[Math.floor(Math.random() * M.test_rows.length)];
  game.done = false;
  const r = game.cur;
  $("case").innerHTML = [["Day", r.day_of_week], ["Time", fmtHour(r.hour)], ["Staff", r.staff_on_shift], ["Waiting", r.patients_in_queue]]
    .map(([k, v]) => `<div class="tag"><small>${k}</small>${v}</div>`).join("");
  $("reveal").innerHTML = "";
  $("guess").disabled = false;
  $("guess-btn").textContent = "Submit guess";
}
function guessBtn() {
  if (game.done) return newRound();
  const r = game.cur, g = +$("guess").value;
  const youErr = Math.abs(g - r.wait_minutes), aiErr = Math.abs(r.predicted - r.wait_minutes);
  const youWin = youErr < aiErr;
  youWin ? game.you++ : game.ai++;
  $("s-you").textContent = game.you; $("s-ai").textContent = game.ai;
  game.done = true; $("guess").disabled = true;
  $("guess-btn").textContent = "Next visit";
  $("reveal").innerHTML = `<div class="win">${youWin ? "You were closer this round." : "The model was closer this round."}</div>
    <div class="row"><div class="tag"><small>Actual wait</small>${r.wait_minutes} min</div><div class="tag"><small>Your guess</small>${g} min (off by ${Math.round(youErr)})</div><div class="tag"><small>Model</small>${r.predicted} min (off by ${Math.round(aiErr)})</div></div>`;
}

// ---------- 3. results ----------
const hidden = new Set();
const isWorst = (r) => r.wait_minutes === M.worst.wait_minutes && r.predicted === M.worst.predicted && r.hour === M.worst.hour && r.day_of_week === M.worst.day_of_week;

function renderResults() {
  const m = M.metrics;
  $("f-mae").textContent = m.model_mae; $("f-base").textContent = m.baseline_mae; $("f-n").textContent = M.n_test;
  const ref = [1, 2, 3, 4, 5].map((s) => predict(11, "Mon", s, 12));
  $("finding").textContent = `Waits fall as staff are added, but with diminishing returns. For a Monday at 11 AM with 12 patients waiting, the model predicts about ${Math.round(ref[0])} min with 1 staff, ${Math.round(ref[1])} min with 2 and ${Math.round(ref[4])} min with 5. The second staff member cuts the wait by about ${Math.round(ref[0] - ref[1])} min; the fifth by only about ${Math.round(ref[3] - ref[4])} min. On held-out visits the network's average error is ${m.model_mae} min, compared with ${m.baseline_mae} min for always guessing the median.`;
  document.querySelector("#metrics tbody").innerHTML = [
    ["Neural network, mean absolute error", m.model_mae + " min"],
    [`Baseline (median ${M.baseline_wait} min), mean absolute error`, m.baseline_mae + " min"],
    ["Error reduction versus baseline", (m.baseline_mae / m.model_mae).toFixed(1) + "× smaller"],
    ["Neural network, R²", m.model_r2],
    ["Validation error (used for tuning)", m.val_mae + " min"],
  ].map(([k, v]) => `<tr><th scope="row">${k}</th><td>${v}</td></tr>`).join("");
  document.querySelector("#by-staff tbody").innerHTML = Object.entries(M.mae_by_staff)
    .map(([s, v]) => `<tr><th scope="row">${s} staff</th><td>${v} min</td></tr>`).join("");

  const w = M.worst;
  $("mistake").innerHTML = `<h3>Largest miss</h3><p>A ${w.day_of_week} ${fmtHour(w.hour)} visit with ${w.staff_on_shift} staff and ${w.patients_in_queue} patients waiting actually waited ${w.wait_minutes} min, but the model predicted ${w.predicted} min, an error of ${Math.abs(w.error)} min. Other Monday visits with 5 staff and 13 to 17 patients waiting ranged from about 37 to 70 min, so the model predicted near the middle of that spread and this visit was unusually quick. Most of this error is random variation in the data that the four inputs cannot explain. See section 4 for what might have helped.</p>`;
  $("help-text").innerHTML = `The largest miss is mostly noise, but three changes could reduce errors like it:
    <ul><li><b>More data in sparse corners.</b> Only 22 of the 2,000 visits fall in the Monday, 5 staff, 13 to 17 waiting corner, so the model sees few examples there.</li>
    <li><b>Another input.</b> A measure such as patients seen per staff member per hour could explain the unusually quick visit.</li>
    <li><b>A range instead of one number.</b> Reporting a prediction interval would show visitors that waits in that corner vary from about 37 to 70 min.</li></ul>`;
  document.querySelector("#train-info tbody").innerHTML = [
    ["Architecture", `${M.features.length} inputs → ${M.hidden.join(" → ")} → 1 output (ReLU)`],
    ["Split", `${M.n_train} train, ${M.n_val} validation, ${M.n_test} test`],
    ["Tuning", "5 architectures × 2 regularization strengths, chosen on validation error only"],
    ["Chosen", `hidden layers ${M.hidden.join(" and ")}, L2 ${M.alpha}`],
    ["Test set use", "Scored once, after choosing the model"],
    ["Baseline", `Median training wait, ${M.baseline_wait} min`],
  ].map(([k, v]) => `<tr><th scope="row">${k}</th><td>${v}</td></tr>`).join("");

  $("legend").innerHTML = [1, 2, 3, 4, 5].map((s) => `<button type="button" class="lg" data-s="${s}" aria-pressed="true"><i style="background:${STAFF_COLORS[s]}"></i>${s} staff</button>`).join("");
  $("legend").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    const s = +b.dataset.s;
    hidden.has(s) ? hidden.delete(s) : hidden.add(s);
    b.setAttribute("aria-pressed", !hidden.has(s));
    document.querySelectorAll("#scatter .pt").forEach((p) => p.classList.toggle("hide", hidden.has(+p.dataset.s)));
  });
}

function drawScatter() {
  const W = 400, H = 320, pad = { l: 40, r: 12, t: 12, b: 34 }, rows = M.test_rows;
  const max = Math.ceil(Math.max(...rows.map((r) => Math.max(r.wait_minutes, r.predicted))) / 50) * 50;
  const x = (v) => pad.l + (v / max) * (W - pad.l - pad.r), y = (v) => H - pad.b - (v / max) * (H - pad.t - pad.b);
  let s = `<line class="axis" x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}"/><line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H - pad.b}"/>`;
  for (let t = 0; t <= max; t += 50) s += `<text x="${x(t)}" y="${H - pad.b + 13}" text-anchor="middle">${t}</text><text x="${pad.l - 5}" y="${y(t) + 3}" text-anchor="end">${t}</text>`;
  s += `<line class="diag" x1="${x(0)}" y1="${y(0)}" x2="${x(max)}" y2="${y(max)}"/>`;
  s += `<text x="${(W + pad.l) / 2}" y="${H - 4}" text-anchor="middle">actual wait (min)</text><text transform="translate(10 ${H / 2}) rotate(-90)" text-anchor="middle">predicted wait (min)</text>`;
  rows.forEach((r, i) => {
    if (isWorst(r)) s += `<circle class="ring" cx="${x(r.wait_minutes)}" cy="${y(r.predicted)}" r="9"/>`;
    s += `<circle class="pt" data-i="${i}" data-s="${r.staff_on_shift}" cx="${x(r.wait_minutes)}" cy="${y(r.predicted)}" r="3.6" fill="${STAFF_COLORS[r.staff_on_shift]}" tabindex="0" aria-label="actual ${r.wait_minutes}, predicted ${r.predicted}"/>`;
  });
  $("scatter").innerHTML = s;
}
function showPoint(i) {
  const r = M.test_rows[i];
  document.querySelectorAll("#scatter .pt").forEach((e) => e.classList.toggle("sel", +e.dataset.i === i));
  $("point").innerHTML = `<b>${r.day_of_week} ${fmtHour(r.hour)}</b>, ${r.staff_on_shift} staff, ${r.patients_in_queue} waiting. Actual <b>${r.wait_minutes} min</b>, predicted <b>${r.predicted} min</b> (${r.error >= 0 ? "+" : ""}${r.error} min).`;
}
function setupScatter() {
  const svg = $("scatter");
  svg.addEventListener("click", (e) => { const t = e.target.closest(".pt"); if (t) showPoint(+t.dataset.i); });
  svg.addEventListener("keydown", (e) => { if (e.key === "Enter") { const t = e.target.closest(".pt"); if (t) showPoint(+t.dataset.i); } });
  svg.addEventListener("pointerover", (e) => {
    const t = e.target.closest(".pt"); if (!t) return;
    const r = M.test_rows[+t.dataset.i], box = svg.parentElement.getBoundingClientRect(), tb = t.getBoundingClientRect(), tip = $("tip");
    tip.hidden = false;
    tip.textContent = `${r.day_of_week} ${fmtHour(r.hour)} · actual ${r.wait_minutes} · predicted ${r.predicted}`;
    tip.style.left = tb.left + tb.width / 2 - box.left + "px";
    tip.style.top = tb.top - box.top + "px";
  });
  svg.addEventListener("pointerout", () => { $("tip").hidden = true; });
  $("show-worst").addEventListener("click", () => { showPoint(M.test_rows.findIndex(isWorst)); $("point").scrollIntoView({ block: "nearest" }); });
}

// held-out table
const tbl = { day: "all", staff: "all", sort: "abs_err", shown: 10 };
function filteredRows() {
  let r = M.test_rows.filter((x) => (tbl.day === "all" || x.day_of_week === tbl.day) && (tbl.staff === "all" || x.staff_on_shift === +tbl.staff));
  const key = tbl.sort;
  return r.slice().sort((a, b) => (key === "abs_err" ? Math.abs(b.error) - Math.abs(a.error) : b[key] - a[key]));
}
function renderTable() {
  const rows = filteredRows();
  document.querySelector("#rows tbody").innerHTML = rows.slice(0, tbl.shown).map((r) =>
    `<tr${r.error === M.worst.error && r.wait_minutes === M.worst.wait_minutes ? ' class="worst"' : ""}><td>${r.day_of_week}</td><td>${fmtHour(r.hour)}</td><td>${r.staff_on_shift}</td><td>${r.patients_in_queue}</td><td class="num">${r.wait_minutes}</td><td class="num">${r.predicted}</td><td class="num">${r.error > 0 ? "+" : ""}${r.error}</td></tr>`).join("");
  $("rows-count").textContent = `Showing ${Math.min(tbl.shown, rows.length)} of ${rows.length} visits`;
  $("more").hidden = tbl.shown >= rows.length;
}
function setupTable() {
  $("f-day").innerHTML = `<option value="all">All days</option>` + M.days.map((d) => `<option>${d}</option>`).join("");
  $("f-staff").innerHTML = `<option value="all">All staff levels</option>` + [1, 2, 3, 4, 5].map((s) => `<option value="${s}">${s}</option>`).join("");
  $("f-day").addEventListener("change", (e) => { tbl.day = e.target.value; tbl.shown = 10; renderTable(); });
  $("f-staff").addEventListener("change", (e) => { tbl.staff = e.target.value; tbl.shown = 10; renderTable(); });
  $("f-sort").addEventListener("change", (e) => { tbl.sort = e.target.value; tbl.shown = 10; renderTable(); });
  $("more").addEventListener("click", () => { tbl.shown += 20; renderTable(); });
  $("dl").addEventListener("click", () => {
    const rows = filteredRows();
    const csv = "day,hour,staff_on_shift,patients_in_queue,actual_wait,predicted_wait,error\n" + rows.map((r) => [r.day_of_week, r.hour, r.staff_on_shift, r.patients_in_queue, r.wait_minutes, r.predicted, r.error].join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "heldout-predictions.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

// ---------- 5. chatbot (WebLLM, opt-in) ----------
const MODEL_ID = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
let engine = null, guide = "", asking = false;
const history = [];
let QUESTIONS = ["Where did the data come from?", "How did the model compare with the baseline?", "How well did the model work on a dataset it was never tested on?"];

async function renderChatTests() {
  let tests = [];
  try { tests = await (await fetch("chatbot-tests.json")).json(); } catch { return; }
  QUESTIONS = tests.map((t) => t.question);
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  document.querySelector("#chat-tests tbody").innerHTML = tests.map((t) => {
    const ran = t.answer != null;
    return `<tr><td>${esc(t.type)}</td><td>${esc(t.question)}</td><td>${ran ? esc(t.answer) : '<span class="pending">Not yet run</span>'}</td><td>${ran ? (t.supported ? "Yes" : "No") + (t.note ? ": " + esc(t.note) : "") : "–"}</td></tr>`;
  }).join("");
}
function addMsg(cls, text) {
  const p = document.createElement("p");
  p.className = "msg " + cls; p.textContent = text;
  $("log").appendChild(p); $("log").scrollTop = $("log").scrollHeight;
  return p;
}
async function enableChat() {
  const btn = $("enable-chat");
  btn.disabled = true; $("chat-progress").hidden = false;
  try {
    const webllm = await import("https://esm.run/@mlc-ai/web-llm@0.2.79");
    engine = await webllm.CreateMLCEngine(MODEL_ID, {
      initProgressCallback: (p) => { $("chat-progress").value = p.progress; $("chat-load-text").textContent = p.text; },
    });
    $("chat-progress").hidden = true; $("chat-load-text").textContent = "";
    btn.hidden = true; $("chat").hidden = false;
    $("chat-status").textContent = "Chat is ready.";
    $("suggest").innerHTML = QUESTIONS.map((q) => `<button type="button">${q}</button>`).join("");
  } catch (e) {
    console.error(e);
    $("chat-progress").hidden = true; $("chat-load-text").textContent = "";
    $("chat-status").textContent = "Chat could not load (" + (e.message || e) + "). The rest of the exhibit still works, and the guide is below.";
    btn.disabled = false;
  }
}
async function ask(q) {
  if (asking || !engine) return;
  asking = true;
  addMsg("u", q);
  const out = addMsg("a", "Thinking…"); out.classList.add("typing");
  const sys = "You answer questions about one web exhibit using ONLY the guide below. " +
    "If the guide does not contain the answer, reply exactly: \"The guide doesn't say, so I can't answer that.\" " +
    "Never invent numbers, datasets, or results. Keep answers to 1 to 3 sentences.\n\n=== GUIDE ===\n" + guide;
  history.push({ role: "user", content: q });
  let text = "";
  try {
    const stream = await engine.chat.completions.create({ messages: [{ role: "system", content: sys }, ...history.slice(-6)], stream: true, temperature: 0.2, max_tokens: 200 });
    for await (const chunk of stream) {
      text += chunk.choices[0]?.delta?.content || "";
      out.classList.remove("typing"); out.textContent = text;
      $("log").scrollTop = $("log").scrollHeight;
    }
    history.push({ role: "assistant", content: text });
  } catch (e) { out.classList.remove("typing"); out.textContent = "Something went wrong: " + (e.message || e); }
  asking = false;
}
function setupChat() {
  if (!("gpu" in navigator)) {
    $("gpu-badge").textContent = "WebGPU not available"; $("gpu-badge").classList.add("bad");
    $("enable-chat").hidden = true;
    $("chat-status").textContent = "This browser cannot run the chat. Use recent desktop Chrome or Edge, or read the guide below.";
    return;
  }
  $("gpu-badge").textContent = "WebGPU available";
  $("chat-status").textContent = "The model downloads only after you select Enable chat.";
  $("enable-chat").addEventListener("click", enableChat);
  $("suggest").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) ask(b.textContent); });
  $("chat-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = $("chat-input").value.trim();
    if (!q) return;
    $("chat-input").value = ""; ask(q);
  });
}

// ---------- init ----------
async function init() {
  M = await (await fetch("model.json")).json();
  M.test_rows.forEach((r) => (r.abs_err = Math.abs(r.error)));
  try { guide = await (await fetch("guide.txt")).text(); $("guide-text").textContent = guide; } catch { $("guide-text").textContent = "The guide could not be loaded."; }
  setupControls(); render();
  $("guess").addEventListener("input", (e) => { $("guess-val").textContent = e.target.value; });
  $("guess-btn").addEventListener("click", guessBtn); newRound();
  renderResults(); drawScatter(); setupScatter(); setupTable(); renderTable();
  await renderChatTests();
  setupChat();
}
init().catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p>The exhibit could not load: ${e.message}</p>`); console.error(e); });
