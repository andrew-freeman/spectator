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
  events.forEach((event) => {
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

async function loadRun(sessionId, runId) {
  const response = await fetch(`/api/sessions/${sessionId}/runs/${runId}`);
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  currentRun = runId;
  finalResponseEl.textContent = payload.final_response || "";
  renderTimeline(payload.events || []);
  renderPerRole(payload.per_role || [], payload.events_by_role || {});
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
