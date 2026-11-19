// Clinic Wait-Time Simulator: the trained network runs in plain JS, no libraries.
const $ = (id) => document.getElementById(id);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const STAFF_COLORS = { 1: "#E5256F", 2: "#F27A0C", 3: "#B58900", 4: "#0FA89A", 5: "#5B3DF5" };
const PEOPLE_COLORS = ["#E5256F", "#F27A0C", "#FFC93C", "#2BA8FF", "#8A3CF0", "#0FA89A"];
let M; // model.json
const state = { hour: 11, day: "Mon", staff: 2, queue: 12 };
let shownPred = 0, prevQueue = 0, prevStaff = 0;

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
function tier(w) {
  if (w < 30) return { t: "⚡ Quick visit", c: "#0B7F74" };
  if (w < 70) return { t: "🙂 Pretty normal", c: "#8A6A00" };
  if (w < 130) return { t: "😬 A long wait", c: "#C25E00" };
  return { t: "🔥 Painful wait", c: "#C21858" };
}
const hm = (m) => { const r = Math.round(m); return r < 60 ? "" : `about ${Math.floor(r / 60)} h ${r % 60} min`; };

// ---------- simulator ----------
const personSVG = (c) => `<svg viewBox="0 0 24 32"><circle cx="12" cy="8" r="6" fill="${c}" stroke="#1B1B3A" stroke-width="1.5"/><path d="M2 31c0-9 4-14 10-14s10 5 10 14z" fill="${c}" stroke="#1B1B3A" stroke-width="1.5"/></svg>`;
const staffSVG = `<svg viewBox="0 0 24 28"><circle cx="12" cy="7" r="5" fill="#fff"/><path d="M2 28c0-8 4-13 10-13s10 5 10 13z" fill="#fff"/><path d="M12 17v7M8.5 20.5h7" stroke="#0FA89A" stroke-width="2.4" stroke-linecap="round"/></svg>`;

function setState(patch) { Object.assign(state, patch); render(); }

function render() {
  const c = state;
  $("o-hour").textContent = fmtHour(c.hour);
  $("o-queue").textContent = c.queue;
  $("hour").value = c.hour; $("queue").value = c.queue;
  document.querySelectorAll("#days button").forEach((b) => b.setAttribute("aria-checked", b.dataset.v === c.day));
  document.querySelectorAll("#staffseg button").forEach((b) => b.setAttribute("aria-checked", +b.dataset.v === c.staff));

  // scene
  $("staff-count").textContent = c.staff;
  $("queue-count").textContent = c.queue;
  $("desks").innerHTML = [1, 2, 3, 4, 5].map((n) =>
    `<button type="button" class="desk${n <= c.staff ? " on" : ""}${n <= c.staff && n > prevStaff ? " pop" : ""}" data-n="${n}" aria-label="Set staff to ${n}">${n <= c.staff ? staffSVG : "+"}</button>`).join("");
  $("people").innerHTML = c.queue === 0
    ? `<span class="empty-room">Nobody waiting 🎉</span>`
    : Array.from({ length: c.queue }, (_, i) => `<span class="person${i >= prevQueue ? " pop" : ""}">${personSVG(PEOPLE_COLORS[i % PEOPLE_COLORS.length])}</span>`).join("");
  prevQueue = c.queue; prevStaff = c.staff;

  // prediction
  const p = predict(c.hour, c.day, c.staff, c.queue);
  countTo(p);
  const tr = tier(p);
  $("tier").textContent = tr.t; $("tier").style.color = tr.c;
  $("marker").style.left = Math.min(p / 300, 1) * 100 + "%";
  $("hm").textContent = hm(p);
  const diff = p - M.baseline_wait;
  $("vs-base").textContent = `Typical clinic (median): ${M.baseline_wait} min. This is ${Math.abs(Math.round(diff))} min ${diff >= 0 ? "longer" : "shorter"}.`;

  const preds = [1, 2, 3, 4, 5].map((s) => predict(c.hour, c.day, s, c.queue));
  const btn = $("plus-one");
  if (c.staff < 5) {
    const save = p - preds[c.staff];
    btn.hidden = false;
    btn.textContent = `+1 staff saves ~${Math.max(0, Math.round(save))} min`;
  } else btn.hidden = true;

  const max = Math.max(...preds, 1);
  $("bars").innerHTML = preds.map((v, i) => {
    const s = i + 1;
    return `<button type="button" class="bar${s === c.staff ? " cur" : ""}" data-n="${s}" aria-label="Try ${s} staff: ${Math.round(v)} minutes"><span>${s} staff</span><span class="track"><span class="fill" style="display:block;width:${(v / max) * 100}%;background:${STAFF_COLORS[s]}"></span></span><span class="v">${Math.round(v)} min</span></button>`;
  }).join("");
  const gains = preds.slice(0, 4).map((v, i) => v - preds[i + 1]);
  $("insight").textContent = c.queue === 0
    ? "💡 With nobody waiting, staffing barely matters. Add patients to see the effect."
    : `💡 Going from 1 to 2 staff saves about ${Math.round(gains[0])} min here. Going from 4 to 5 saves only about ${Math.round(gains[3])} min. Each extra person helps less than the one before.`;
}

function countTo(target) {
  const el = $("pred");
  if (reduced) { el.textContent = Math.round(target); shownPred = target; return; }
  const from = shownPred, t0 = performance.now(), dur = 300;
  shownPred = target;
  cancelAnimationFrame(countTo.raf);
  const step = (now) => {
    const k = Math.min((now - t0) / dur, 1);
    el.textContent = Math.round(from + (target - from) * (1 - Math.pow(1 - k, 3)));
    if (k < 1) countTo.raf = requestAnimationFrame(step);
  };
  countTo.raf = requestAnimationFrame(step);
}

const PRESETS = [
  { n: "😰 Monday rush", s: { day: "Mon", hour: 9, staff: 2, queue: 18 } },
  { n: "😌 Quiet afternoon", s: { day: "Wed", hour: 15, staff: 3, queue: 3 } },
  { n: "💪 Fully staffed", s: { day: "Thu", hour: 11, staff: 5, queue: 12 } },
  { n: "🧍 Skeleton crew", s: { day: "Tue", hour: 12, staff: 1, queue: 14 } },
  { n: "🎲 Surprise me", s: null },
];
function setupControls() {
  $("presets").innerHTML = PRESETS.map((p, i) => `<button class="preset" type="button" data-i="${i}">${p.n}</button>`).join("");
  $("presets").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    const p = PRESETS[+b.dataset.i];
    setState(p.s || { day: M.days[Math.floor(Math.random() * M.days.length)], hour: 8 + Math.floor(Math.random() * 10), staff: 1 + Math.floor(Math.random() * 5), queue: Math.floor(Math.random() * 21) });
  });
  $("days").innerHTML = M.days.map((d) => `<button type="button" role="radio" data-v="${d}">${d}</button>`).join("");
  $("days").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState({ day: b.dataset.v }); });
  $("staffseg").innerHTML = [1, 2, 3, 4, 5].map((n) => `<button type="button" role="radio" data-v="${n}">${n}</button>`).join("");
  $("staffseg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState({ staff: +b.dataset.v }); });
  $("desks").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState({ staff: +b.dataset.n }); });
  $("bars").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setState({ staff: +b.dataset.n }); });
  $("hour").addEventListener("input", (e) => setState({ hour: +e.target.value }));
  $("queue").addEventListener("input", (e) => setState({ queue: +e.target.value }));
  $("q-minus").addEventListener("click", () => setState({ queue: Math.max(0, state.queue - 1) }));
  $("q-plus").addEventListener("click", () => setState({ queue: Math.min(20, state.queue + 1) }));
  $("plus-one").addEventListener("click", () => setState({ staff: Math.min(5, state.staff + 1) }));
}

// ---------- guessing game ----------
const game = { you: 0, ai: 0, cur: null, done: false };
function newRound() {
  game.cur = M.test_rows[Math.floor(Math.random() * M.test_rows.length)];
  game.done = false;
  const r = game.cur;
  $("case").innerHTML = [["Day", r.day_of_week], ["Time", fmtHour(r.hour)], ["Staff", r.staff_on_shift], ["Waiting", r.patients_in_queue]]
    .map(([k, v]) => `<div class="tag"><small>${k}</small>${v}</div>`).join("");
  $("reveal").innerHTML = "";
  $("guess").disabled = false;
  $("guess-btn").textContent = "Lock in my guess";
}
function guessBtn() {
  if (game.done) return newRound();
  const r = game.cur, g = +$("guess").value;
  const youErr = Math.abs(g - r.wait_minutes), aiErr = Math.abs(r.predicted - r.wait_minutes);
  const youWin = youErr < aiErr;
  if (youWin) game.you++; else game.ai++;
  $("s-you").textContent = game.you; $("s-ai").textContent = game.ai;
  game.done = true; $("guess").disabled = true;
  $("guess-btn").textContent = "Next visit →";
  $("reveal").innerHTML = `<div class="win">${youWin ? "🎉 You win this round!" : "🤖 The network wins this round."}</div>
    <div class="row"><div class="tag"><small>Actual wait</small>${r.wait_minutes} min</div><div class="tag"><small>You</small>${g} min (off by ${Math.round(youErr)})</div><div class="tag"><small>Network</small>${r.predicted} min (off by ${Math.round(aiErr)})</div></div>`;
}

// ---------- results + scatter ----------
const hidden = new Set();
function renderStats() {
  const m = M.metrics;
  $("hero-mae").textContent = m.model_mae;
  $("stats").innerHTML = `
    <div class="stat"><b>${m.model_mae} min</b><span>neural network: average miss on ${M.n_test} test visits</span></div>
    <div class="stat"><b>${m.baseline_mae} min</b><span>baseline: always guess the median (${M.baseline_wait} min)</span></div>
    <div class="stat"><b>${(m.baseline_mae / m.model_mae).toFixed(1)}× better</b><span>smaller error than the baseline</span></div>`;
  const w = M.worst;
  $("mistake").innerHTML = `<b>🔍 One mistake worth understanding.</b> A ${w.day_of_week} ${fmtHour(w.hour)} visit with ${w.staff_on_shift} staff and ${w.patients_in_queue} in the queue waited ${w.wait_minutes} min, but the model said ${w.predicted} min (off by ${Math.abs(w.error)}). Other Monday visits with 5 staff and 13 to 17 in the queue waited anywhere from about 37 to 70 min, so the model guessed near the middle of that spread and this visit was just unusually quick. Most of this miss is randomness in the data that these four inputs can't explain. More data in that corner, or an extra input like how many patients each staff member sees per hour, might help a little.`;
  $("legend").innerHTML = [1, 2, 3, 4, 5].map((s) => `<button type="button" class="lg" data-s="${s}" aria-pressed="true" style="background:${STAFF_COLORS[s]}">${s} staff</button>`).join("");
  $("legend").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    const s = +b.dataset.s;
    hidden.has(s) ? hidden.delete(s) : hidden.add(s);
    b.setAttribute("aria-pressed", !hidden.has(s));
    document.querySelectorAll("#scatter .pt").forEach((p) => p.classList.toggle("hide", hidden.has(+p.dataset.s)));
  });
}
const isWorst = (r) => r.wait_minutes === M.worst.wait_minutes && r.predicted === M.worst.predicted && r.hour === M.worst.hour && r.day_of_week === M.worst.day_of_week;
const geom = { W: 400, H: 320, pad: { l: 40, r: 12, t: 12, b: 34 } };
function drawScatter() {
  const { W, H, pad } = geom, rows = M.test_rows;
  const max = Math.ceil(Math.max(...rows.map((r) => Math.max(r.wait_minutes, r.predicted))) / 50) * 50;
  geom.x = (v) => pad.l + (v / max) * (W - pad.l - pad.r);
  geom.y = (v) => H - pad.b - (v / max) * (H - pad.t - pad.b);
  const { x, y } = geom;
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
  $("point").innerHTML = `<b>${r.day_of_week} ${fmtHour(r.hour)}</b>, ${r.staff_on_shift} staff, ${r.patients_in_queue} waiting. Actual wait <b>${r.wait_minutes} min</b>, model said <b>${r.predicted} min</b> (${r.error >= 0 ? "+" : ""}${r.error} min).`;
}
function setupScatter() {
  const wrap = $("scatter");
  wrap.addEventListener("click", (e) => { const t = e.target.closest(".pt"); if (t) showPoint(+t.dataset.i); });
  wrap.addEventListener("keydown", (e) => { if (e.key === "Enter") { const t = e.target.closest(".pt"); if (t) showPoint(+t.dataset.i); } });
  wrap.addEventListener("pointerover", (e) => {
    const t = e.target.closest(".pt"); if (!t) return;
    const r = M.test_rows[+t.dataset.i], box = wrap.parentElement.getBoundingClientRect(), tb = t.getBoundingClientRect();
    const tip = $("tip");
    tip.hidden = false;
    tip.textContent = `${r.day_of_week} ${fmtHour(r.hour)} · actual ${r.wait_minutes} · model ${r.predicted}`;
    tip.style.left = tb.left + tb.width / 2 - box.left + "px";
    tip.style.top = tb.top - box.top + "px";
  });
  wrap.addEventListener("pointerout", () => { $("tip").hidden = true; });
  $("show-worst").addEventListener("click", () => showPoint(M.test_rows.findIndex(isWorst)));
}

// ---------- chatbot (WebLLM, opt-in) ----------
const MODEL_ID = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
const QUESTIONS = ["Where did the data come from?", "How did the model compare with the baseline?", "How well did the model work on a dataset it was never tested on?", "What are the limits of this exhibit?"];
let engine = null, guide = "";
const history = [];
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
    $("chat-status").textContent = "Chat is ready. It answers only from the guide below and can still make mistakes.";
    $("suggest").innerHTML = QUESTIONS.map((q) => `<button type="button">${q}</button>`).join("");
  } catch (e) {
    console.error(e);
    $("chat-progress").hidden = true; $("chat-load-text").textContent = "";
    $("chat-status").textContent = "Chat couldn't load (" + (e.message || e) + "). The rest of the exhibit still works, and you can read the guide below.";
    btn.disabled = false;
  }
}
let asking = false;
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
  } catch (e) { out.classList.remove("typing"); out.textContent = "Sorry, something went wrong: " + (e.message || e); }
  asking = false;
}
function setupChat() {
  if (!("gpu" in navigator)) {
    $("enable-chat").hidden = true;
    $("chat-status").textContent = "Chat needs WebGPU, which this browser doesn't have. Try recent desktop Chrome or Edge. Everything else works, and the guide below has the same answers.";
    return;
  }
  $("chat-status").textContent = "Chat is optional and loads only when you click the button.";
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
  try { guide = await (await fetch("guide.txt")).text(); $("guide-text").textContent = guide; } catch { $("guide-text").textContent = "Couldn't load the guide."; }
  setupControls();
  render();
  renderStats(); drawScatter(); setupScatter();
  $("guess").addEventListener("input", (e) => { $("guess-val").textContent = e.target.value; });
  $("guess-btn").addEventListener("click", guessBtn);
  newRound();
  setupChat();
}
init().catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p>Couldn't load the model: ${e.message}</p>`); });
