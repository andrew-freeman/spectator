const sessionsEl = document.getElementById("sessions");
const runsEl = document.getElementById("runs");
const timelineEl = document.getElementById("timeline");
const perRoleEl = document.getElementById("per-role");
const toolsEl = document.getElementById("tools");
const finalResponseEl = document.getElementById("final-response");
const sanitizeWarningsEl = document.getElementById("sanitize-warnings");

let currentSession = null;
let currentRun = null;

function clearElement(el) {
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
}

function createButton(label, onClick, isActive) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `list__item${isActive ? " list__item--active" : ""}`;
  button.textContent = label;
  button.addEventListener("click", onClick);
  return button;
}

function formatTs(ts) {
  if (typeof ts !== "number") {
    return "";
  }
  return new Date(ts * 1000).toLocaleTimeString();
}

function renderTimeline(events) {
  clearElement(timelineEl);
  events.forEach((event) => {
    const div = document.createElement("div");
    div.className = "list__row";
    div.textContent = `${formatTs(event.ts)} • ${event.kind}${event.role ? ` (${event.role})` : ""}`;
    timelineEl.appendChild(div);
  });
}

function renderPerRole(eventsByRole) {
  clearElement(perRoleEl);
  Object.entries(eventsByRole).forEach(([role, events]) => {
    const section = document.createElement("section");
    section.className = "role";
    const header = document.createElement("h3");
    header.textContent = role;
    section.appendChild(header);
    const list = document.createElement("div");
    list.className = "list";
    events.forEach((event) => {
      const item = document.createElement("div");
      item.className = "list__row";
      item.textContent = `${formatTs(event.ts)} • ${event.kind}`;
      list.appendChild(item);
    });
    section.appendChild(list);
    perRoleEl.appendChild(section);
  });
}

function renderTools(toolCalls) {
  clearElement(toolsEl);
  toolCalls.forEach((tool) => {
    const row = document.createElement("tr");
    const cells = [
      tool.id,
      tool.tool,
      tool.role,
      tool.ok,
      tool.duration_ms ? tool.duration_ms.toFixed(1) : "",
      tool.error,
    ];
    cells.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value === null || value === undefined ? "" : String(value);
      row.appendChild(cell);
    });
    toolsEl.appendChild(row);
  });
}

function renderSanitizeWarnings(warnings) {
  clearElement(sanitizeWarningsEl);
  warnings.forEach((warning) => {
    const item = document.createElement("div");
    item.className = "list__row";
    item.textContent = warning.message || JSON.stringify(warning);
    sanitizeWarningsEl.appendChild(item);
  });
}

async function loadRun(sessionId, runId) {
  const response = await fetch(`/api/sessions/${sessionId}/runs/${runId}`);
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  currentRun = runId;
  finalResponseEl.textContent = payload.final_response || "";
  renderTimeline(payload.events || []);
  renderPerRole(payload.events_by_role || {});
  renderTools(payload.tool_calls || []);
  renderSanitizeWarnings(payload.sanitize_warnings || []);
}

async function loadRuns(sessionId) {
  const response = await fetch(`/api/sessions/${sessionId}/runs`);
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  clearElement(runsEl);
  payload.runs.forEach((run) => {
    const button = createButton(
      run.run_id,
      () => loadRun(sessionId, run.run_id),
      run.run_id === currentRun,
    );
    runsEl.appendChild(button);
  });
}

async function loadSessions(preselect) {
  const response = await fetch("/api/sessions");
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  clearElement(sessionsEl);
  payload.sessions.forEach((session) => {
    const sessionId = session.session_id;
    if (!sessionId) {
      return;
    }
    const button = createButton(
      sessionId,
      () => {
        currentSession = sessionId;
        loadRuns(sessionId);
      },
      sessionId === currentSession,
    );
    sessionsEl.appendChild(button);
  });
  if (preselect && !currentSession) {
    currentSession = preselect;
    loadRuns(preselect);
  }
}

function readQueryParams() {
  const params = new URLSearchParams(window.location.search);
  return {
    session: params.get("session"),
    run: params.get("run"),
  };
}

async function bootstrap() {
  const { session, run } = readQueryParams();
  if (session) {
    currentSession = session;
  }
  await loadSessions(session);
  if (session && run) {
    currentRun = run;
    await loadRuns(session);
    await loadRun(session, run);
  }
}

bootstrap();
