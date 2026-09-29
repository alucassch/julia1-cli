// julia1-cli serve: the playground page (GET /), one self-contained HTML file (no external resources). It sends
// {"state", "questions"} to POST /v1/predict and draws the answers.
#pragma once

static const char PLAYGROUND_HTML[] = R"PLAYGROUND(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Julia-1 playground</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --text: #1b1f24; --muted: #5d6673; --line: #d9dde3; --accent: #2f6fdf;
  --accent-soft: #dbe7fb; --bar: #9db7e8; --win: #2f6fdf; --error-bg: #fdecec; --error: #a4262c; --input: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111418; --panel: #1a1e24; --text: #e6e9ee; --muted: #9aa3ae; --line: #2e343c; --accent: #6c9cf0;
    --accent-soft: #1f2d45; --bar: #3a5584; --win: #6c9cf0; --error-bg: #3a1c1e; --error: #f1a2a6; --input: #14171c;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 14px 20px; border-bottom: 1px solid var(--line); background: var(--panel); display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: baseline; }
header h1 { font-size: 17px; margin: 0; }
#info { color: var(--muted); font-size: 12.5px; }
main { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 16px; padding: 16px 20px; max-width: 1400px; margin: 0 auto; }
@media (max-width: 900px) { main { grid-template-columns: minmax(0, 1fr); padding: 12px 16px; } }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
h2 { font-size: 14px; margin: 0 0 10px; display: flex; gap: 8px; align-items: center; justify-content: space-between; flex-wrap: wrap; }
label { color: var(--muted); font-size: 12px; }
input, select, textarea { font: inherit; color: var(--text); background: var(--input); border: 1px solid var(--line); border-radius: 5px; padding: 5px 7px; }
textarea { width: 100%; min-height: 110px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
input:focus, select:focus, textarea:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
button { font: inherit; cursor: pointer; border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 5px; padding: 5px 10px; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; padding: 7px 16px; }
button.small { padding: 2px 8px; font-size: 12px; }
button.remove { color: var(--muted); padding: 2px 8px; }
.row { display: flex; gap: 6px; align-items: center; margin: 4px 0; }
.row > input.grow { flex: 1; min-width: 0; }
.question { border: 1px solid var(--line); border-radius: 6px; padding: 10px; margin: 10px 0; }
.question .head { display: grid; grid-template-columns: 130px 110px minmax(0, 1fr) auto; gap: 6px; align-items: center; }
@media (max-width: 600px) { .question .head { grid-template-columns: 1fr 1fr; } .question .head .instructions { grid-column: 1 / -1; } }
.criteria { margin-top: 8px; padding-left: 8px; border-left: 3px solid var(--accent-soft); }
.criteria .key { width: 130px; }
.criteria .idx { width: 22px; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.hint { color: var(--muted); font-size: 12px; margin: 4px 0; }
.actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
#latency { color: var(--muted); font-size: 12.5px; font-variant-numeric: tabular-nums; }
.error { background: var(--error-bg); color: var(--error); border-radius: 6px; padding: 8px 10px; margin: 8px 0; white-space: pre-wrap; }
.answer { border: 1px solid var(--line); border-radius: 6px; padding: 10px; margin-bottom: 10px; }
.answer .title { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.answer .id { font-weight: 600; }
.badge { font-size: 11px; color: var(--accent); background: var(--accent-soft); border-radius: 4px; padding: 1px 6px; }
.headline { font-size: 16px; margin: 6px 0 8px; }
.headline small { color: var(--muted); font-size: 12.5px; }
.bar { display: grid; grid-template-columns: minmax(0, 1fr) 64px; gap: 8px; align-items: center; margin: 3px 0; }
.bar .label { font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar .track { grid-column: 1 / -1; height: 7px; background: var(--accent-soft); border-radius: 4px; overflow: hidden; margin-top: -2px; }
.bar .fill { height: 100%; background: var(--bar); }
.bar.win .fill { background: var(--win); }
.bar.win .label { font-weight: 600; }
.bar .value { text-align: right; font-variant-numeric: tabular-nums; font-size: 12.5px; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--muted); font-size: 12.5px; }
pre { background: var(--bg); border: 1px solid var(--line); border-radius: 5px; padding: 8px; overflow: auto; font-size: 12px; max-height: 360px; }
#keybox { display: none; gap: 6px; align-items: center; }
.empty { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Julia-1 playground</h1>
  <span id="info">loading model info…</span>
  <span id="keybox"><label for="apikey">API key</label><input id="apikey" type="password" size="22" autocomplete="off"><button id="savekey" class="small">Use key</button></span>
</header>
<main>
  <section>
    <h2>State
      <span class="row">
        <label for="stateFormat">format</label>
        <select id="stateFormat"><option value="text">text</option><option value="json">JSON</option></select>
        <button class="small" data-example="readme">Example: billing / shipping / access</button>
        <button class="small" data-example="pt">Exemplo: roteamento (pt)</button>
      </span>
    </h2>
    <textarea id="state" spellcheck="false"></textarea>
    <h2 style="margin-top:14px">Questions <button id="addQuestion" class="small">+ question</button></h2>
    <div class="hint">choice: 2–20 ids with descriptions · score: an ordered rubric of 2–20 levels (the answer is the expected level index) · noul: optional false/true descriptions (empty: literal false/true)</div>
    <div id="questions"></div>
    <div class="actions">
      <button id="run" class="primary">Predict</button>
      <span id="latency"></span>
    </div>
  </section>
  <section>
    <h2>Answers</h2>
    <div id="error"></div>
    <div id="answers"><p class="empty">Press Predict (Ctrl/⌘ + Enter).</p></div>
    <details><summary>Request (POST /v1/predict)</summary><pre id="request"></pre></details>
    <details><summary>Response</summary><pre id="response"></pre></details>
  </section>
</main>
<script>
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") el.className = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v); else el[k] = v;
  }
  for (const c of children) el.append(c);
  return el;
}

let apiKey = "";
try { apiKey = sessionStorage.getItem("julia1-api-key") || ""; } catch (e) {}
function headers() {
  const out = { "Content-Type": "application/json" };
  if (apiKey) out["Authorization"] = "Bearer " + apiKey;
  return out;
}

const EXAMPLES = {
  readme: {
    format: "text",
    state: "I was charged twice for the same order.",
    questions: {
      team: { type: "choice", instructions: "Which team should handle this request?",
              criteria: { billing: "Billing and payment disputes", shipping: "Shipping and delivery", access: "Account access and login" } },
    },
  },
  pt: {
    format: "json",
    state: {
      canal: "chat",
      cliente: { plano: "premium", pedidos_anteriores: 7 },
      mensagem: "Olá! Comprei um fone de ouvido há 12 dias e o pedido ainda não chegou. O rastreio está parado em \"em trânsito\" desde segunda-feira e eu viajo na sexta.",
    },
    questions: {
      fila: { type: "choice", instructions: "Qual equipe deve atender esta mensagem?",
              criteria: { financeiro: "Cobrança duplicada, reembolso ou problema de pagamento", logistica: "Pedido atrasado, entrega ou rastreamento",
                          suporte_tecnico: "Produto com defeito ou problema técnico", conta: "Login, senha ou acesso à conta" } },
      urgencia: { type: "score", instructions: "Qual é a urgência do atendimento?",
                  criteria: ["Baixa: pode esperar alguns dias", "Média: responder ainda hoje", "Alta: responder imediatamente"] },
      insatisfeito: { type: "noul", instructions: "O cliente está insatisfeito?",
                      criteria: { false: "Não, o cliente está satisfeito", true: "Sim, o cliente está insatisfeito ou reclamando" } },
    },
  },
};

function criteriaRow(type, key = "", text = "") {
  const row = h("div", { class: "row crit" });
  if (type === "choice") row.append(h("input", { class: "key", placeholder: "id", value: key }));
  else row.append(h("span", { class: "idx" }));
  row.append(h("input", { class: "grow desc", placeholder: "description", value: text }),
             h("button", { class: "remove", title: "remove", textContent: "×", onclick: () => { const box = row.parentElement; row.remove(); renumber(box); } }));
  return row;
}
function renumber(box) { $$(".idx", box).forEach((el, i) => { el.textContent = i; }); }

function renderCriteria(card, criteria) {
  const type = $(".type", card).value, box = $(".criteria", card);
  box.replaceChildren();
  if (type === "noul") {
    const c = criteria && typeof criteria === "object" ? criteria : {};
    for (const k of ["false", "true"]) {
      box.append(h("div", { class: "row" }, h("span", { class: "key", textContent: k, style: "width:40px;color:var(--muted)" }),
                   h("input", { class: "grow noul-" + k, placeholder: "optional description (empty: literal " + k + ")", value: c[k] || "" })));
    }
    return;
  }
  const rows = h("div", { class: "rows" });
  const entries = type === "choice"
    ? Object.entries(criteria && !Array.isArray(criteria) ? criteria : { a: "", b: "" })
    : (Array.isArray(criteria) ? criteria : ["", ""]).map((t) => ["", t]);
  for (const [k, t] of entries) rows.append(criteriaRow(type, k, t));
  renumber(rows);
  box.append(rows, h("button", { class: "small", textContent: type === "choice" ? "+ option" : "+ level",
                                 onclick: () => { rows.append(criteriaRow(type)); renumber(rows); } }));
}

function addQuestion(id = "", q = { type: "choice", instructions: "" }) {
  const card = h("div", { class: "question" });
  const type = h("select", { class: "type" });
  for (const t of ["choice", "score", "noul"]) type.append(h("option", { value: t, textContent: t }));
  type.value = q.type;
  type.addEventListener("change", () => renderCriteria(card, null));
  card.append(h("div", { class: "head" },
                h("input", { class: "qid", placeholder: "question id", value: id }), type,
                h("input", { class: "instructions", placeholder: "instructions (the question)", value: q.instructions || "" }),
                h("button", { class: "remove", title: "remove question", textContent: "×", onclick: () => card.remove() })),
              h("div", { class: "criteria" }));
  renderCriteria(card, q.criteria);
  $("#questions").append(card);
}

function loadExample(name) {
  const ex = EXAMPLES[name];
  $("#stateFormat").value = ex.format;
  $("#state").value = ex.format === "json" ? JSON.stringify(ex.state, null, 2) : ex.state;
  $("#questions").replaceChildren();
  for (const [id, q] of Object.entries(ex.questions)) addQuestion(id, q);
  $("#answers").replaceChildren(h("p", { class: "empty", textContent: "Press Predict (Ctrl/⌘ + Enter)." }));
  $("#error").replaceChildren();
}

// The request body and, per question, the description of each answer key.
function collect() {
  let state = $("#state").value;
  if ($("#stateFormat").value === "json") {
    try { state = JSON.parse(state); } catch (e) { throw new Error("State is not valid JSON: " + e.message); }
  }
  const questions = {}, labels = {};
  for (const card of $$(".question")) {
    const id = $(".qid", card).value, type = $(".type", card).value;
    const q = { type, instructions: $(".instructions", card).value };
    const names = {};
    if (type === "choice") {
      q.criteria = {};
      for (const row of $$(".crit", card)) { const k = $(".key", row).value, t = $(".desc", row).value; q.criteria[k] = t; names[k] = t; }
    } else if (type === "score") {
      q.criteria = $$(".crit .desc", card).map((el) => el.value);
      q.criteria.forEach((t, i) => { names[String(i)] = t; });
    } else {
      const f = $(".noul-false", card).value, t = $(".noul-true", card).value;
      if (f || t) { q.criteria = { false: f, true: t }; names.false = f; names.true = t; }
    }
    questions[id] = q;
    labels[id] = names;
  }
  return { body: { state, questions }, labels };
}

function pct(p) { return (100 * p).toFixed(1) + "%"; }

function renderAnswers(result, labels) {
  const out = [];
  for (const [id, a] of Object.entries(result.answers)) {
    const names = labels[id] || {};
    const probs = Object.entries(a.probabilities);
    const best = probs.reduce((m, e) => (e[1] > m[1] ? e : m), probs[0])[0];
    let headline;
    if (a.type === "choice") headline = h("div", { class: "headline" }, a.choice, " ", h("small", { textContent: names[a.choice] ? "— " + names[a.choice] : "" }));
    else if (a.type === "score") headline = h("div", { class: "headline" }, "score " + a.score.toFixed(3), " ",
                                              h("small", { textContent: "(expected level index, 0–" + (probs.length - 1) + "; most likely " + best + (names[best] ? ": " + names[best] : "") + ")" }));
    else headline = h("div", { class: "headline" }, "P(true) = " + a.noul.toFixed(4), " ", h("small", { textContent: a.noul >= 0.5 ? "→ true" : "→ false" }));
    const card = h("div", { class: "answer" },
      h("div", { class: "title" }, h("span", { class: "id", textContent: id }), h("span", { class: "badge", textContent: a.type }),
        a.max_probability !== undefined ? h("span", { class: "hint", textContent: "max probability " + a.max_probability.toFixed(4) }) : ""),
      headline);
    for (const [k, p] of probs) {
      const fill = h("div", { class: "fill" });
      fill.style.width = (100 * p).toFixed(2) + "%";
      card.append(h("div", { class: "bar" + (k === best ? " win" : "") },
        h("span", { class: "label", textContent: k + (names[k] && names[k] !== k ? " — " + names[k] : ""), title: names[k] || k }),
        h("span", { class: "value", textContent: pct(p) }),
        h("div", { class: "track" }, fill)));
    }
    out.push(card);
  }
  $("#answers").replaceChildren(...out);
}

function showError(message) { $("#error").replaceChildren(h("div", { class: "error", textContent: message })); }

async function run() {
  $("#error").replaceChildren();
  let req;
  try { req = collect(); } catch (e) { showError(e.message); return; }
  $("#request").textContent = JSON.stringify(req.body, null, 2);
  $("#run").disabled = true;
  const t0 = performance.now();
  try {
    const res = await fetch("/v1/predict", { method: "POST", headers: headers(), body: JSON.stringify(req.body) });
    const text = await res.text();
    const ms = performance.now() - t0;
    $("#response").textContent = text;
    let timing = "";
    for (const m of (res.headers.get("Server-Timing") || "").split(",")) {
      const r = /^\s*(\w+);dur=([\d.]+)(?:;desc="([^"]*)")?/.exec(m);
      if (r) timing += " · " + r[1] + " " + Number(r[2]).toFixed(1) + " ms" + (r[3] ? " (" + r[3] + ")" : "");
    }
    $("#latency").textContent = "round trip " + ms.toFixed(1) + " ms" + (timing ? " · server:" + timing : "");
    if (res.status === 401) { $("#keybox").style.display = "flex"; }
    const body = JSON.parse(text);
    if (!res.ok) { showError("HTTP " + res.status + ": " + (body.error || text)); return; }
    renderAnswers(body, req.labels);
  } catch (e) {
    showError(String(e));
  } finally {
    $("#run").disabled = false;
  }
}

async function loadInfo() {
  try {
    const res = await fetch("/v1/model", { headers: headers() });
    if (res.status === 401) {
      $("#keybox").style.display = "flex";
      $("#info").textContent = "this server requires an API key";
      return;
    }
    const m = await res.json();
    $("#keybox").style.display = "none";
    $("#info").textContent = m.name + " · " + m.file_type + " · " + m.device + " (" + m.mode + ") · max_length " + m.max_length +
      ", head_length " + m.head_length + (m.strict ? ", strict" : "") + " · " + m.version;
  } catch (e) {
    $("#info").textContent = "model info unavailable: " + e;
  }
}

$("#savekey").addEventListener("click", () => {
  apiKey = $("#apikey").value;
  try { sessionStorage.setItem("julia1-api-key", apiKey); } catch (e) {}
  loadInfo();
});
$("#addQuestion").addEventListener("click", () => addQuestion("q" + ($$(".question").length + 1)));
$("#run").addEventListener("click", run);
for (const b of $$("[data-example]")) b.addEventListener("click", () => loadExample(b.dataset.example));
document.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
loadExample("readme");
loadInfo();
</script>
</body>
</html>
)PLAYGROUND";
