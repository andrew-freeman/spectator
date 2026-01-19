const sessionsEl = document.getElementById("sessions");
const runsEl = document.getElementById("runs");
const timelineEl = document.getElementById("timeline");
const perRoleEl = document.getElementById("per-role");
const toolsEl = document.getElementById("tools");
const finalResponseEl = document.getElementById("final-response");
const sanitizeWarningsEl = document.getElementById("sanitize-warnings");
const selectionStatusEl = document.getElementById("selection-status");
const refreshAllBtn = document.getElementById("refresh-all");
const refreshSessionsBtn = document.getElementById("refresh-sessions");
const refreshRunsBtn = document.getElementById("refresh-runs");
const loadLatestRunBtn = document.getElementById("load-latest-run");
const copyFinalBtn = document.getElementById("copy-final");
const timelineFilterInput = document.getElementById("timeline-filter");
const timelineKindSelect = document.getElementById("timeline-kind");
const openLoopsEl = document.getElementById("open-loops");
const openLoopForm = document.getElementById("open-loop-form");
const openLoopTitle = document.getElementById("open-loop-title");
const openLoopDetails = document.getElementById("open-loop-details");
const openLoopTags = document.getElementById("open-loop-tags");
const openLoopPriority = document.getElementById("open-loop-priority");
const tabInspectBtn = document.getElementById("tab-button-inspect");
const tabChatBtn = document.getElementById("tab-button-chat");
const tabInspectPanel = document.getElementById("tab-inspect");
const tabChatPanel = document.getElementById("tab-chat");
const chatStatusEl = document.getElementById("chat-status");
const chatLogEl = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatSessionInput = document.getElementById("chat-session");
const chatBackendInput = document.getElementById("chat-backend");
const chatMessageInput = document.getElementById("chat-message");
const chatClearBtn = document.getElementById("chat-clear");

let currentSession = null;
let currentRun = null;
let currentEvents = [];
let chatMessages = [];
let sessionInputTouched = false;
let backendInputTouched = false;

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

async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "absolute";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function createActionButton(label, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "button button--small";
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

function scrollToRole(role) {
  const target = document.getElementById(`role-${role}`);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderTimeline(events) {
  clearElement(timelineEl);
  const query = (timelineFilterInput?.value || "").toLowerCase().trim();
  const kindFilter = timelineKindSelect?.value || "";
  const filtered = events.filter((event) => {
    if (kindFilter && event.kind !== kindFilter) {
      return false;
    }
    if (!query) {
      return true;
    }
    const haystack = [
      event.kind,
      event.role,
      JSON.stringify(event.data || {}),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
  filtered.forEach((event) => {
    const label = `${formatTs(event.ts)} • ${event.kind}${event.role ? ` (${event.role})` : ""}`;
    if (event.kind === "llm_req" && event.role) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "list__row list__row--link";
      button.textContent = label;
      button.addEventListener("click", () => scrollToRole(event.role));
      timelineEl.appendChild(button);
      return;
    }
    const div = document.createElement("div");
    div.className = "list__row";
    div.textContent = label;
    timelineEl.appendChild(div);
  });
}

function createCopyButton(label, text) {
  const button = createActionButton(label, async () => {
    await copyTextToClipboard(text || "");
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = label;
    }, 1000);
  });
  return button;
}

function createDownloadButton(role, llmReq, llmDone) {
  return createActionButton("Download raw event JSON", () => {
    const payload = { role, llm_req: llmReq, llm_done: llmDone };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `trace-${role || "unknown"}.json`;
    link.click();
    URL.revokeObjectURL(url);
  });
}

function createCollapsible(title, previewText, contentNode, isOpen) {
  const details = document.createElement("details");
  details.className = "collapsible";
  details.open = Boolean(isOpen);
  const summary = document.createElement("summary");
  summary.className = "collapsible__summary";
  const summaryTitle = document.createElement("span");
  summaryTitle.textContent = title;
  summary.appendChild(summaryTitle);
  if (previewText) {
    const preview = document.createElement("span");
    preview.className = "collapsible__preview";
    preview.textContent = previewText;
    summary.appendChild(preview);
  }
  details.appendChild(summary);
  details.appendChild(contentNode);
  return details;
}

function createCodeBlock(text) {
  const pre = document.createElement("pre");
  pre.className = "code";
  pre.textContent = text || "";
  return pre;
}

function renderPromptSections(entry, rawPrompt) {
  const sections = entry.prompt_sections || {};
  const sectionNames = Object.keys(sections);
  if (sectionNames.length === 0) {
    return createCodeBlock(rawPrompt || "");
  }
  const container = document.createElement("div");
  container.className = "prompt";
  const preferredOrder = ["STATE", "HISTORY_JSON", "UPSTREAM", "USER"];
  const ordered = [
    ...preferredOrder.filter((name) => sectionNames.includes(name)),
    ...sectionNames.filter((name) => !preferredOrder.includes(name)),
  ];
  ordered.forEach((name) => {
    const section = document.createElement("section");
    section.className = "prompt__section";
    const header = document.createElement("h4");
    header.className = "prompt__title";
    header.textContent = name;
    section.appendChild(header);
    const content =
      name === "HISTORY_JSON" && entry.history_json_pretty
        ? entry.history_json_pretty
        : sections[name];
    section.appendChild(createCodeBlock(content));
    container.appendChild(section);
  });
  return container;
}

function renderPerRole(perRoles, eventsByRole) {
  clearElement(perRoleEl);
  if (!Array.isArray(perRoles) || perRoles.length === 0) {
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
    return;
  }
  perRoles.forEach((entry) => {
    const role = entry.role || "unknown";
    const section = document.createElement("section");
    section.className = "role";
    section.id = `role-${role}`;
    const header = document.createElement("div");
    header.className = "role__header";
    const title = document.createElement("h3");
    title.textContent = role;
    header.appendChild(title);
    header.appendChild(
      createDownloadButton(role, entry.llm_req || null, entry.llm_done || null),
    );
    section.appendChild(header);

    const llmReq = entry.llm_req || {};
    const llmReqData = llmReq.data || {};
    const llmDone = entry.llm_done || {};
    const llmDoneData = llmDone.data || {};

    const systemPrompt = typeof llmReqData.system_prompt === "string" ? llmReqData.system_prompt : "";
    const systemPromptPreview =
      systemPrompt.length > 200 ? `${systemPrompt.slice(0, 200)}…` : systemPrompt;
    const systemContent = document.createElement("div");
    systemContent.className = "role__content";
    const systemActions = document.createElement("div");
    systemActions.className = "role__actions";
    systemActions.appendChild(createCopyButton("Copy system_prompt", systemPrompt));
    systemContent.appendChild(systemActions);
    systemContent.appendChild(createCodeBlock(systemPrompt));
    section.appendChild(
      createCollapsible(
        "system_prompt",
        systemPromptPreview,
        systemContent,
        false,
      ),
    );

    const prompt = typeof llmReqData.prompt === "string" ? llmReqData.prompt : "";
    const promptContent = document.createElement("div");
    promptContent.className = "role__content";
    const promptActions = document.createElement("div");
    promptActions.className = "role__actions";
    promptActions.appendChild(createCopyButton("Copy prompt", prompt));
    promptContent.appendChild(promptActions);
    promptContent.appendChild(renderPromptSections(entry, prompt));
    section.appendChild(createCollapsible("prompt", null, promptContent, true));

    const response = typeof llmDoneData.response === "string" ? llmDoneData.response : "";
    const responseBlock = document.createElement("div");
    responseBlock.className = "role__content";
    const responseHeader = document.createElement("div");
    responseHeader.className = "role__actions";
    responseHeader.appendChild(createCopyButton("Copy response", response));
    responseBlock.appendChild(responseHeader);
    responseBlock.appendChild(createCodeBlock(response));
    const responseWrapper = document.createElement("div");
    responseWrapper.className = "role__response";
    const responseTitle = document.createElement("h4");
    responseTitle.textContent = "response";
    responseWrapper.appendChild(responseTitle);
    responseWrapper.appendChild(responseBlock);
    section.appendChild(responseWrapper);
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

function renderOpenLoops(openLoops) {
  clearElement(openLoopsEl);
  if (!Array.isArray(openLoops) || openLoops.length === 0) {
    const empty = document.createElement("div");
    empty.className = "list__row list__row--muted";
    empty.textContent = "No open loops.";
    openLoopsEl.appendChild(empty);
    return;
  }
  openLoops.forEach((loop) => {
    const row = document.createElement("div");
    row.className = "list__row list__row--card";
    const title = document.createElement("div");
    title.className = "loop__title";
    title.textContent = loop.title || loop.raw || "Untitled loop";
    row.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "loop__meta";
    const chips = [];
    if (loop.id) {
      chips.push(`id:${loop.id}`);
    }
    if (typeof loop.priority === "number") {
      chips.push(`priority:${loop.priority}`);
    }
    if (Array.isArray(loop.tags) && loop.tags.length) {
      chips.push(`tags:${loop.tags.join(", ")}`);
    }
    meta.textContent = chips.join(" • ");
    row.appendChild(meta);

    if (loop.details) {
      const details = document.createElement("div");
      details.className = "loop__details";
      details.textContent = loop.details;
      row.appendChild(details);
    }

    if (loop.id) {
      const closeBtn = createActionButton("Close", () => closeOpenLoop(loop.id));
      row.appendChild(closeBtn);
    }
    openLoopsEl.appendChild(row);
  });
}

function renderTimelineKindOptions(events) {
  const kinds = Array.from(new Set(events.map((event) => event.kind))).sort();
  clearElement(timelineKindSelect);
  const allOption = document.createElement("option");
  allOption.value = "";
  allOption.textContent = "All";
  timelineKindSelect.appendChild(allOption);
  kinds.forEach((kind) => {
    const option = document.createElement("option");
    option.value = kind;
    option.textContent = kind;
    timelineKindSelect.appendChild(option);
  });
}

function updateSelectionStatus() {
  if (!selectionStatusEl) {
    return;
  }
  const sessionText = currentSession ? `Session: ${currentSession}` : "No session selected";
  const runText = currentRun ? `Run: ${currentRun}` : "Run: —";
  const eventText = currentEvents.length ? `Events: ${currentEvents.length}` : "Events: —";
  selectionStatusEl.textContent = `${sessionText} • ${runText} • ${eventText}`;
}

function getNextSessionId(sessions) {
  let maxValue = null;
  sessions.forEach((session) => {
    const sessionId = session.session_id;
    if (typeof sessionId !== "string") {
      return;
    }
    let value = null;
    if (/^\d+$/.test(sessionId)) {
      value = Number(sessionId);
    } else {
      const match = sessionId.match(/(\d+)\s*$/);
      if (match) {
        value = Number(match[1]);
      }
    }
    if (Number.isFinite(value)) {
      if (maxValue === null || value > maxValue) {
        maxValue = value;
      }
    }
  });
  if (maxValue === null) {
    return "0";
  }
  return String(maxValue + 1);
}

function ensureChatDefaults(sessions) {
  if (chatBackendInput && !backendInputTouched && !chatBackendInput.value.trim()) {
    chatBackendInput.value = "llama";
  }
  if (chatSessionInput && !sessionInputTouched && !chatSessionInput.value.trim()) {
    chatSessionInput.value = getNextSessionId(sessions);
  }
}

function setActiveTab(tab) {
  const isInspect = tab === "inspect";
  tabInspectBtn.classList.toggle("tab--active", isInspect);
  tabChatBtn.classList.toggle("tab--active", !isInspect);
  tabInspectBtn.setAttribute("aria-selected", String(isInspect));
  tabChatBtn.setAttribute("aria-selected", String(!isInspect));
  tabInspectPanel.classList.toggle("tab-panel--active", isInspect);
  tabChatPanel.classList.toggle("tab-panel--active", !isInspect);
}

function updateChatStatus(text) {
  if (chatStatusEl) {
    chatStatusEl.textContent = text;
  }
}

function renderChatLog() {
  clearElement(chatLogEl);
  if (!chatMessages.length) {
    const empty = document.createElement("div");
    empty.className = "list__row list__row--muted";
    empty.textContent = "No messages yet.";
    chatLogEl.appendChild(empty);
    return;
  }
  chatMessages.forEach((message) => {
    const bubble = document.createElement("div");
    bubble.className = `chat__bubble chat__bubble--${message.role}`;
    const header = document.createElement("div");
    header.className = "chat__meta";
    header.textContent = `${message.role} • ${message.session}`;
    const body = document.createElement("div");
    body.className = "chat__text";
    body.textContent = message.text;
    bubble.appendChild(header);
    bubble.appendChild(body);
    chatLogEl.appendChild(bubble);
  });
  chatLogEl.scrollTop = chatLogEl.scrollHeight;
}

async function sendChatMessage() {
  const session = chatSessionInput.value.trim() || currentSession;
  const message = chatMessageInput.value.trim();
  const backend = chatBackendInput.value.trim();
  if (!session || !message) {
    updateChatStatus("Session and message required.");
    return;
  }
  updateChatStatus("Sending...");
  chatMessages.push({ role: "user", text: message, session });
  renderChatLog();
  chatMessageInput.value = "";
  const payload = { session_id: session, text: message };
  if (backend) {
    payload.backend = backend;
  }
  const response = await fetch("/api/run_turn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    updateChatStatus("Failed to send.");
    return;
  }
  const data = await response.json();
  chatMessages.push({
    role: "assistant",
    text: data.final_text || "",
    session,
  });
  renderChatLog();
  updateChatStatus(`Run: ${data.run_id || "unknown"}`);
  currentSession = session;
  chatSessionInput.value = session;
  await loadSessions(session);
  if (data.run_id) {
    currentRun = data.run_id;
    await loadRuns(session);
    await loadRun(session, data.run_id);
  }
}

async function loadOpenLoops(sessionId) {
  if (!sessionId) {
    renderOpenLoops([]);
    return;
  }
  const response = await fetch(`/api/sessions/${sessionId}/open_loops`);
  if (!response.ok) {
    renderOpenLoops([]);
    return;
  }
  const payload = await response.json();
  renderOpenLoops(payload.open_loops || []);
}

async function addOpenLoop(sessionId) {
  if (!sessionId) {
    return;
  }
  const title = openLoopTitle.value.trim();
  const details = openLoopDetails.value.trim();
  const tags = openLoopTags.value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
  const priorityValue = openLoopPriority.value.trim();
  const priority = priorityValue ? Number(priorityValue) : null;

  const payload = {
    title,
    details: details || null,
    tags: tags.length ? tags : null,
    priority: Number.isFinite(priority) ? priority : null,
  };

  const response = await fetch(`/api/sessions/${sessionId}/open_loops`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    return;
  }
  const data = await response.json();
  openLoopTitle.value = "";
  openLoopDetails.value = "";
  openLoopTags.value = "";
  openLoopPriority.value = "";
  renderOpenLoops(data.open_loops || []);
}

async function closeOpenLoop(loopId) {
  if (!currentSession || !loopId) {
    return;
  }
  const response = await fetch(
    `/api/sessions/${currentSession}/open_loops/${loopId}/close`,
    { method: "POST" },
  );
  if (!response.ok) {
    return;
  }
  const data = await response.json();
  renderOpenLoops(data.open_loops || []);
}

async function loadRun(sessionId, runId) {
  const response = await fetch(`/api/sessions/${sessionId}/runs/${runId}`);
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  currentRun = runId;
  finalResponseEl.textContent = payload.final_response || "";
  currentEvents = payload.events || [];
  renderTimelineKindOptions(currentEvents);
  renderTimeline(currentEvents);
  renderPerRole(payload.per_role || [], payload.events_by_role || {});
  renderTools(payload.tool_calls || []);
  renderSanitizeWarnings(payload.sanitize_warnings || []);
  updateSelectionStatus();
}

async function loadRuns(sessionId) {
  const response = await fetch(`/api/sessions/${sessionId}/runs`);
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  clearElement(runsEl);
  const runs = Array.isArray(payload.runs) ? payload.runs : [];
  if (runs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "list__row list__row--muted";
    empty.textContent = "No runs found.";
    runsEl.appendChild(empty);
  }
  runs.forEach((run) => {
    const button = createButton(
      run.run_id,
      () => loadRun(sessionId, run.run_id),
      run.run_id === currentRun,
    );
    runsEl.appendChild(button);
  });
  updateSelectionStatus();
}

async function loadSessions(preselect) {
  const response = await fetch("/api/sessions");
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  clearElement(sessionsEl);
  const sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
  if (sessions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "list__row list__row--muted";
    empty.textContent = "No sessions found.";
    sessionsEl.appendChild(empty);
  }
  sessions.forEach((session) => {
    const sessionId = session.session_id;
    if (!sessionId) {
      return;
    }
    const button = createButton(
      sessionId,
      () => {
        currentSession = sessionId;
        currentRun = null;
        currentEvents = [];
        finalResponseEl.textContent = "";
        if (chatSessionInput) {
          chatSessionInput.value = sessionId;
        }
        loadRuns(sessionId);
        loadOpenLoops(sessionId);
        updateSelectionStatus();
      },
      sessionId === currentSession,
    );
    sessionsEl.appendChild(button);
  });
  if (preselect && !currentSession) {
    currentSession = preselect;
    if (chatSessionInput) {
      chatSessionInput.value = preselect;
    }
    loadRuns(preselect);
    loadOpenLoops(preselect);
  }
  ensureChatDefaults(sessions);
  updateSelectionStatus();
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
    await loadOpenLoops(session);
  }
}

if (refreshAllBtn) {
  refreshAllBtn.addEventListener("click", () => {
    loadSessions(currentSession);
    if (currentSession) {
      loadRuns(currentSession);
      loadOpenLoops(currentSession);
    }
    if (currentSession && currentRun) {
      loadRun(currentSession, currentRun);
    }
  });
}

if (refreshSessionsBtn) {
  refreshSessionsBtn.addEventListener("click", () => loadSessions(currentSession));
}

if (refreshRunsBtn) {
  refreshRunsBtn.addEventListener("click", () => {
    if (currentSession) {
      loadRuns(currentSession);
    }
  });
}

if (loadLatestRunBtn) {
  loadLatestRunBtn.addEventListener("click", async () => {
    if (!currentSession) {
      return;
    }
    const response = await fetch(`/api/sessions/${currentSession}/runs`);
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    const runs = Array.isArray(payload.runs) ? payload.runs : [];
    if (runs.length === 0) {
      return;
    }
    const latest = runs[runs.length - 1];
    currentRun = latest.run_id;
    await loadRuns(currentSession);
    await loadRun(currentSession, latest.run_id);
  });
}

if (copyFinalBtn) {
  copyFinalBtn.addEventListener("click", () => {
    copyTextToClipboard(finalResponseEl.textContent || "");
  });
}

if (timelineFilterInput) {
  timelineFilterInput.addEventListener("input", () => renderTimeline(currentEvents));
}

if (timelineKindSelect) {
  timelineKindSelect.addEventListener("change", () => renderTimeline(currentEvents));
}

if (openLoopForm) {
  openLoopForm.addEventListener("submit", (event) => {
    event.preventDefault();
    addOpenLoop(currentSession);
  });
}

if (tabInspectBtn && tabChatBtn) {
  tabInspectBtn.addEventListener("click", () => setActiveTab("inspect"));
  tabChatBtn.addEventListener("click", () => setActiveTab("chat"));
}

if (chatForm) {
  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendChatMessage();
  });
}

if (chatSessionInput) {
  chatSessionInput.addEventListener("input", () => {
    sessionInputTouched = true;
  });
}

if (chatBackendInput) {
  chatBackendInput.addEventListener("input", () => {
    backendInputTouched = true;
  });
}

if (chatClearBtn) {
  chatClearBtn.addEventListener("click", () => {
    chatMessages = [];
    renderChatLog();
    updateChatStatus("Cleared.");
  });
}

bootstrap();
