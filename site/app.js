// Clinic Wait-Time Simulator: runs the trained network in plain JS, no libraries.
const $ = (id) => document.getElementById(id);
let M; // model.json

const relu = (x) => (x > 0 ? x : 0);

function predict(hour, day, staff, queue) {
  const raw = { hour, staff_on_shift: staff, patients_in_queue: queue };
  for (const d of M.days) raw["day_" + d] = d === day ? 1 : 0;
  let a = M.features.map((f, i) => (raw[f] - M.mean[i]) / M.std[i]);
  M.layers.forEach((L, li) => {
    const out = L.b.slice();
    for (let j = 0; j < out.length; j++) {
      for (let i = 0; i < a.length; i++) out[j] += a[i] * L.W[i][j];
    }
    a = li < M.layers.length - 1 ? out.map(relu) : out;
  });
  return Math.max(0, a[0]);
}

function cur() {
  return { hour: +$("hour").value, day: $("day").value, staff: +$("staff").value, queue: +$("queue").value };
}

function render() {
  const c = cur();
  $("o-hour").textContent = c.hour + ":00";
  $("o-staff").textContent = c.staff;
  $("o-queue").textContent = c.queue;
  const p = predict(c.hour, c.day, c.staff, c.queue);
  $("pred").textContent = Math.round(p);
  const diff = p - M.baseline_wait;
  $("vs-base").textContent = `Typical clinic (median) wait is ${M.baseline_wait} min, so this is ${Math.abs(Math.round(diff))} min ${diff >= 0 ? "longer" : "shorter"}.`;

  const preds = [1, 2, 3, 4, 5].map((s) => predict(c.hour, c.day, s, c.queue));
  const max = Math.max(...preds, 1);
  $("bars").innerHTML = preds.map((v, i) => {
    const s = i + 1;
    return `<div class="bar${s === c.staff ? " cur" : ""}"><span>${s} staff</span><div class="track"><div class="fill" style="width:${(v / max) * 100}%"></div></div><span class="v">${Math.round(v)} min</span></div>`;
  }).join("");

  // One clear discovery: diminishing returns of adding staff
  const gains = preds.slice(0, 4).map((v, i) => v - preds[i + 1]);
  const first = gains[0], last = gains[3];
  $("insight").textContent = c.queue === 0
    ? "With nobody in the queue, staffing barely matters. Add patients to the queue to see it."
    : `Going from 1 to 2 staff saves about ${Math.round(first)} min here. Going from 4 to 5 saves only about ${Math.round(last)} min. Each extra person helps less than the one before.`;
}

// ---------- held-out gallery ----------
function drawScatter() {
  const W = 400, H = 320, pad = { l: 40, r: 12, t: 12, b: 34 };
  const rows = M.test_rows;
  const max = Math.ceil(Math.max(...rows.map((r) => Math.max(r.wait_minutes, r.predicted))) / 50) * 50;
  const x = (v) => pad.l + (v / max) * (W - pad.l - pad.r);
  const y = (v) => H - pad.b - (v / max) * (H - pad.t - pad.b);
  let s = `<line class="axis" x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}"/><line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H - pad.b}"/>`;
  for (let t = 0; t <= max; t += 50) {
    s += `<text x="${x(t)}" y="${H - pad.b + 13}" text-anchor="middle">${t}</text><text x="${pad.l - 5}" y="${y(t) + 3}" text-anchor="end">${t}</text>`;
  }
  s += `<line class="diag" x1="${x(0)}" y1="${y(0)}" x2="${x(max)}" y2="${y(max)}"/>`;
  s += `<text x="${(W + pad.l) / 2}" y="${H - 4}" text-anchor="middle">actual wait (min)</text>`;
  s += `<text transform="translate(10 ${H / 2}) rotate(-90)" text-anchor="middle">predicted wait (min)</text>`;
  rows.forEach((r, i) => {
    const w = r === M.worst || (r.hour === M.worst.hour && r.day_of_week === M.worst.day_of_week && r.error === M.worst.error && r.wait_minutes === M.worst.wait_minutes);
    s += `<circle class="pt${w ? " worst" : ""}" data-i="${i}" cx="${x(r.wait_minutes)}" cy="${y(r.predicted)}" r="${w ? 5 : 3.2}" tabindex="0"><title>actual ${r.wait_minutes}, predicted ${r.predicted}</title></circle>`;
  });
  $("scatter").innerHTML = s;
}

function showPoint(i) {
  const r = M.test_rows[i];
  document.querySelectorAll("#scatter .pt").forEach((e) => e.classList.toggle("sel", +e.dataset.i === i));
  const e = r.error;
  $("point").innerHTML = `<b>${r.day_of_week} ${r.hour}:00</b>, ${r.staff_on_shift} staff, ${r.patients_in_queue} in queue. Actual wait <b>${r.wait_minutes} min</b>, model said <b>${r.predicted} min</b> (${e >= 0 ? "+" : ""}${e} min).`;
}

function worstIndex() {
  const w = M.worst;
  return M.test_rows.findIndex((r) => r.wait_minutes === w.wait_minutes && r.predicted === w.predicted && r.hour === w.hour && r.day_of_week === w.day_of_week);
}

function renderStats() {
  const m = M.metrics;
  $("stats").innerHTML = `
    <div class="stat"><b>${m.model_mae} min</b><span>neural network, average miss on ${M.n_test} test visits</span></div>
    <div class="stat"><b>${m.baseline_mae} min</b><span>baseline (always guess the median, ${M.baseline_wait} min)</span></div>
    <div class="stat"><b>${(m.baseline_mae / m.model_mae).toFixed(1)}×</b><span>smaller error than the baseline</span></div>`;
  const w = M.worst;
  $("mistake").innerHTML = `<b>One mistake worth understanding.</b> A ${w.day_of_week} ${w.hour}:00 visit with ${w.staff_on_shift} staff and ${w.patients_in_queue} in the queue waited ${w.wait_minutes} min, but the model said ${w.predicted} min (off by ${Math.abs(w.error)}). Other Monday visits with 5 staff and 13 to 17 in the queue waited anywhere from about 37 to 70 min, so the model guessed near the middle of that spread and this visit was just unusually quick. Most of this miss is randomness in the data that these four inputs can't explain. More data in that corner, or an extra input like how many patients each staff member sees per hour, might help a little.`;
}

// ---------- chatbot (WebLLM, opt-in) ----------
const MODEL_ID = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
let engine = null, guide = "";
const history = [];

function addMsg(cls, text) {
  const p = document.createElement("p");
  p.className = "msg " + cls;
  p.textContent = text;
  $("log").appendChild(p);
  $("log").scrollTop = $("log").scrollHeight;
  return p;
}

async function enableChat() {
  const btn = $("enable-chat");
  btn.disabled = true;
  $("chat-progress").hidden = false;
  try {
    guide = await (await fetch("guide.txt")).text();
    const webllm = await import("https://esm.run/@mlc-ai/web-llm@0.2.79");
    engine = await webllm.CreateMLCEngine(MODEL_ID, {
      initProgressCallback: (p) => {
        $("chat-progress").value = p.progress;
        $("chat-load-text").textContent = p.text;
      },
    });
    $("chat-progress").hidden = true;
    $("chat-load-text").textContent = "";
    btn.hidden = true;
    $("chat").hidden = false;
    $("chat-status").textContent = "Chat is ready. It answers only from the exhibit guide and can still make mistakes.";
  } catch (e) {
    console.error(e);
    $("chat-progress").hidden = true;
    $("chat-load-text").textContent = "";
    $("chat-status").textContent = "Chat couldn't load (" + (e.message || e) + "). The rest of the exhibit still works.";
    btn.disabled = false;
  }
}

async function ask(q) {
  addMsg("u", q);
  const out = addMsg("a", "…");
  const sys = "You answer questions about one web exhibit using ONLY the guide below. " +
    "If the guide does not contain the answer, reply exactly: \"The guide doesn't say, so I can't answer that.\" " +
    "Never invent numbers, datasets, or results. Keep answers to 1 to 3 sentences.\n\n=== GUIDE ===\n" + guide;
  history.push({ role: "user", content: q });
  const msgs = [{ role: "system", content: sys }, ...history.slice(-6)];
  let text = "";
  try {
    const stream = await engine.chat.completions.create({ messages: msgs, stream: true, temperature: 0.2, max_tokens: 200 });
    for await (const chunk of stream) {
      text += chunk.choices[0]?.delta?.content || "";
      out.textContent = text;
      $("log").scrollTop = $("log").scrollHeight;
    }
    history.push({ role: "assistant", content: text });
  } catch (e) {
    out.textContent = "Sorry, something went wrong: " + (e.message || e);
  }
}

function setupChat() {
  if (!("gpu" in navigator)) {
    $("enable-chat").hidden = true;
    $("chat-status").textContent = "Chat needs WebGPU, which this browser doesn't have. Try a recent desktop Chrome or Edge. The rest of the exhibit works without it.";
    return;
  }
  $("chat-status").textContent = "Chat is optional and loads only when you click the button.";
  $("enable-chat").addEventListener("click", enableChat);
  $("chat-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = $("chat-input").value.trim();
    if (!q || !engine) return;
    $("chat-input").value = "";
    ask(q);
  });
}

// ---------- init ----------
async function init() {
  M = await (await fetch("model.json")).json();
  const sel = $("day");
  sel.innerHTML = M.days.map((d) => `<option>${d}</option>`).join("");
  sel.value = "Mon";
  ["hour", "day", "staff", "queue"].forEach((id) => $(id).addEventListener("input", render));
  render();
  renderStats();
  drawScatter();
  $("scatter").addEventListener("click", (e) => { const t = e.target.closest(".pt"); if (t) showPoint(+t.dataset.i); });
  $("scatter").addEventListener("keydown", (e) => { if (e.key === "Enter") { const t = e.target.closest(".pt"); if (t) showPoint(+t.dataset.i); } });
  $("show-worst").addEventListener("click", () => showPoint(worstIndex()));
  setupChat();
}
init().catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p>Couldn't load the model: ${e.message}</p>`); });
