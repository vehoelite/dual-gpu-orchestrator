"use strict";

const $ = (id) => document.getElementById(id);
const log = $("log");

function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + cls;
}

function addEntry(cls, who, text) {
  const div = document.createElement("div");
  div.className = "entry " + cls;
  if (who) {
    const span = document.createElement("span");
    span.className = "who";
    span.textContent = who;
    div.appendChild(span);
  }
  div.appendChild(document.createTextNode(text));
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function renderPlan(ev) {
  $("plan-count").textContent = `(${ev.done}/${ev.total})`;
  const ol = $("plan");
  ol.innerHTML = "";
  for (const step of ev.steps) {
    const li = document.createElement("li");
    li.className = step.status;
    li.textContent = `${step.index}. ${step.description}`;
    ol.appendChild(li);
  }
}

function running(isRunning) {
  $("start").disabled = isRunning;
  $("kill").disabled = !isRunning;
}

const handlers = {
  run_started: (e) => { running(true); setStatus("running", "running"); addEntry("system", "", `▶ run started: ${e.goal}`); },
  plan: (e) => renderPlan(e),
  message: (e) => addEntry(e.agent, e.agent + ":", e.text),
  action: (e) => addEntry("action " + e.agent, e.agent + " ⇒ " + e.verb, JSON.stringify(e.args) + (e.body_preview ? "\n" + e.body_preview : "")),
  result: (e) => addEntry("result", "result " + e.status, e.message_preview),
  parse_error: (e) => addEntry("error", e.agent + " parse error", e.error),
  worker_started: (e) => addEntry("system", "", `→ delegating step ${e.step}: ${e.subtask}`),
  worker_finished: (e) => addEntry("system", "", `← worker finished step ${e.step} (${e.stopped_reason})`),
  no_progress: () => addEntry("system", "", "⚠ no progress — stopping"),
  run_finished: (e) => { running(false); setStatus("finished: " + e.stopped_reason, "done"); addEntry("system", "", "■ run finished: " + e.stopped_reason); },
  run_aborted: () => { running(false); setStatus("aborted", "aborted"); addEntry("system", "", "■ run aborted"); },
  error: (e) => { running(false); setStatus("error", "error"); addEntry("error", "error", e.message); },
};

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    (handlers[ev.type] || (() => {}))(ev);
  };
  ws.onclose = () => setTimeout(connectWs, 1000);
}

async function loadModels() {
  const resp = await fetch("/api/models");
  if (!resp.ok) { addEntry("error", "error", "could not reach LM Studio via /api/models"); return; }
  const data = await resp.json();
  for (const id of ["dominant", "worker"]) {
    const sel = $(id);
    sel.innerHTML = "";
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m; opt.textContent = m;
      sel.appendChild(opt);
    }
  }
  if (data.models.length > 1) $("worker").selectedIndex = 1;
  const cb = $("research");
  cb.disabled = !data.research_available;
  cb.checked = data.research_available;
  $("research-label").textContent = data.research_available ? "research (available)" : "research (unavailable)";
}

$("start").onclick = async () => {
  log.innerHTML = "";
  $("plan").innerHTML = "";
  const body = {
    dominant: $("dominant").value,
    worker: $("worker").value,
    project: $("project").value,
    goal: $("goal").value,
    enable_research: $("research").checked,
    debug: $("debug").checked,
  };
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (resp.status === 409) addEntry("error", "error", "a run is already active");
};

$("kill").onclick = () => fetch("/api/stop", { method: "POST" });

connectWs();
loadModels();
