/**
 * Knowledge Transfer Agent — web chat UI
 */

const STORAGE_KEY = "kta_ui_settings";
const WORKSPACE_KEY = "kta_current_workspace";
const CHAT_PREFIX = "kta_chat_";
const THEME_KEY = "kta_theme";

const INGEST_CHUNK_SIZE = 50;
const INGEST_SKIP_DIRS = new Set([
  ".git",
  "node_modules",
  "__pycache__",
  ".venv",
  "venv",
  "dist",
  "build",
  ".idea",
  ".pytest_cache",
  ".mypy_cache",
  ".tox",
  "target",
  ".next",
  ".nuxt",
  "coverage",
  ".eggs",
]);
const INGEST_EXTENSIONS = new Set([
  ".txt",
  ".md",
  ".rst",
  ".markdown",
  ".pdf",
  ".py",
  ".pyi",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".mjs",
  ".cjs",
  ".go",
  ".java",
  ".kt",
  ".kts",
  ".rs",
  ".c",
  ".h",
  ".cpp",
  ".hpp",
  ".cc",
  ".cxx",
  ".cs",
  ".rb",
  ".php",
  ".swift",
  ".sql",
  ".sh",
  ".bash",
  ".zsh",
  ".yaml",
  ".yml",
  ".json",
  ".toml",
  ".ini",
  ".cfg",
  ".conf",
  ".html",
  ".htm",
  ".css",
  ".scss",
  ".less",
  ".vue",
  ".svelte",
  ".xml",
  ".gradle",
  ".properties",
  ".ipynb",
]);

const SAMPLE_PROMPTS = [
  "How does the RAG pipeline work in this project?",
  "What sources can we ingest — Confluence, GitHub, files?",
  "Explain hybrid retrieval and citations [N].",
  "How is the agent graph structured (planner → retrieve → generate)?",
];

const $ = (sel) => document.querySelector(sel);

const els = {
  messages: $("#messages"),
  messagesWrap: $("#messagesWrap"),
  welcome: $("#welcome"),
  composerForm: $("#composerForm"),
  questionInput: $("#questionInput"),
  btnSend: $("#btnSend"),
  btnNewChat: $("#btnNewChat"),
  welcomePromptChips: $("#welcomePromptChips"),
  conversationsEmpty: $("#conversationsEmpty"),
  headerSubtitle: $("#headerSubtitle"),
  btnHeaderIngest: $("#btnHeaderIngest"),
  btnChatUpload: $("#btnChatUpload"),
  btnWelcomeUpload: $("#btnWelcomeUpload"),
  ingestDialogTitle: $("#ingestDialogTitle"),
  ingestContextProject: $("#ingestContextProject"),
  ingestContextChat: $("#ingestContextChat"),
  projectMenu: $("#projectMenu"),
  healthStatus: $("#healthStatus"),
  streamMode: $("#streamMode"),
  btnTheme: $("#btnTheme"),
  citationsPanel: $("#citationsPanel"),
  citationsList: $("#citationsList"),
  btnCloseCitations: $("#btnCloseCitations"),
  settingsDialog: $("#settingsDialog"),
  settingsForm: $("#settingsForm"),
  apiBaseUrl: $("#apiBaseUrl"),
  apiKey: $("#apiKey"),
  btnSettings: $("#btnSettings"),
  btnCloseSettings: $("#btnCloseSettings"),
  btnToggleSidebar: $("#btnToggleSidebar"),
  sidebar: $("#sidebar"),
  btnIngest: $("#btnIngest"),
  btnClearIndex: $("#btnClearIndex"),
  ingestDialog: $("#ingestDialog"),
  ingestForm: $("#ingestForm"),
  ingestFiles: $("#ingestFiles"),
  ingestFolder: $("#ingestFolder"),
  ingestSummary: $("#ingestSummary"),
  localIngestPath: $("#localIngestPath"),
  btnLocalIngest: $("#btnLocalIngest"),
  tabLocalPath: $("#tabLocalPath"),
  tabGitClone: $("#tabGitClone"),
  ingestPanelFolder: $("#ingestPanelFolder"),
  ingestPanelGit: $("#ingestPanelGit"),
  ingestPanelLocal: $("#ingestPanelLocal"),
  gitCloneUrl: $("#gitCloneUrl"),
  gitCloneBranch: $("#gitCloneBranch"),
  btnGitClone: $("#btnGitClone"),
  fileDrop: $("#fileDrop"),
  fileList: $("#fileList"),
  replaceIndex: $("#replaceIndex"),
  ingestStatus: $("#ingestStatus"),
  ingestProgressPanel: $("#ingestProgressPanel"),
  uploadProgressBlock: $("#uploadProgressBlock"),
  indexProgressBlock: $("#indexProgressBlock"),
  uploadProgressFill: $("#uploadProgressFill"),
  indexProgressFill: $("#indexProgressFill"),
  uploadPercent: $("#uploadPercent"),
  indexPercent: $("#indexPercent"),
  uploadProgressDetail: $("#uploadProgressDetail"),
  indexProgressDetail: $("#indexProgressDetail"),
  btnCloseIngest: $("#btnCloseIngest"),
  btnCancelIngest: $("#btnCancelIngest"),
  btnSubmitIngest: $("#btnSubmitIngest"),
  workspaceSelect: $("#workspaceSelect"),
  btnNewWorkspace: $("#btnNewWorkspace"),
  btnDeleteWorkspace: $("#btnDeleteWorkspace"),
  headerProjectName: $("#headerProjectName"),
  btnManageHistory: $("#btnManageHistory"),
  historyDialog: $("#historyDialog"),
  btnCloseHistory: $("#btnCloseHistory"),
  btnCloseHistoryDone: $("#btnCloseHistoryDone"),
  historyCurrentProjectTitle: $("#historyCurrentProjectTitle"),
  historyMsgList: $("#historyMsgList"),
  historyMsgEmpty: $("#historyMsgEmpty"),
  historyProjectList: $("#historyProjectList"),
  histMsgSelectAll: $("#histMsgSelectAll"),
  histMsgSelectNone: $("#histMsgSelectNone"),
  histMsgDeleteSelected: $("#histMsgDeleteSelected"),
  histMsgClearAll: $("#histMsgClearAll"),
  histProjSelectAll: $("#histProjSelectAll"),
  histProjSelectNone: $("#histProjSelectNone"),
  histProjDeleteSelected: $("#histProjDeleteSelected"),
  bootSkeleton: $("#bootSkeleton"),
  welcomeEmptyIndex: $("#welcomeEmptyIndex"),
  welcomePromptsBlock: $("#welcomePromptsBlock"),
  btnWelcomeIndex: $("#btnWelcomeIndex"),
  appVersionLabel: $("#appVersionLabel"),
  aboutVersion: $("#aboutVersion"),
  aboutEndpoint: $("#aboutEndpoint"),
  settingsConnStatus: $("#settingsConnStatus"),
  confirmDialog: $("#confirmDialog"),
  confirmTitle: $("#confirmTitle"),
  confirmMessage: $("#confirmMessage"),
  confirmOk: $("#confirmOk"),
  confirmCancel: $("#confirmCancel"),
  confirmForm: $("#confirmForm"),
  toastHost: $("#toastHost"),
  askProgress: $("#askProgress"),
  btnLibrary: $("#btnLibrary"),
  btnOpenLibrary: $("#btnOpenLibrary"),
  libraryDialog: $("#libraryDialog"),
  btnCloseLibrary: $("#btnCloseLibrary"),
  btnCloseLibraryDone: $("#btnCloseLibraryDone"),
  librarySubtitle: $("#librarySubtitle"),
  libraryDocStats: $("#libraryDocStats"),
  libraryDocSearch: $("#libraryDocSearch"),
  libraryDocList: $("#libraryDocList"),
  libraryDocEmpty: $("#libraryDocEmpty"),
  btnRefreshDocs: $("#btnRefreshDocs"),
  libraryAuditList: $("#libraryAuditList"),
  libraryAuditEmpty: $("#libraryAuditEmpty"),
  btnRefreshAudit: $("#btnRefreshAudit"),
  libraryMemoryList: $("#libraryMemoryList"),
  libraryMemoryEmpty: $("#libraryMemoryEmpty"),
  btnRefreshMemory: $("#btnRefreshMemory"),
  btnClearMemory: $("#btnClearMemory"),
};

let abortController = null;
let isLoading = false;
let selectedFiles = [];
let ingestSkippedCount = 0;
/** @type {"project"|"chat"} project = new sidebar chat; chat = link to current conversation */
let ingestUIMode = "project";
let currentWorkspaceId = "default";
let workspaces = [];
let apiFeatures = {};
let currentServerThreadId = null;
let lastQuestion = "";
let lastAnswer = "";
let lastCitations = [];
let appVersion = "";
let indexReady = false;
let confirmResolver = null;
let libraryDocsCache = [];
let libraryTab = "docs";

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveSettings(data) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

/**
 * API root URL (no /api/v1, no /app).
 * Fixes common mistake: saving http://host:8000/app as API URL → 404 on ingest.
 */
function normalizeBaseUrl(url) {
  let base = (url || window.location.origin).trim().replace(/\/+$/, "");
  if (base.endsWith("/app")) base = base.slice(0, -4);
  return base;
}

function getBaseUrl() {
  const saved = loadSettings().apiBaseUrl?.trim();
  return normalizeBaseUrl(saved || window.location.origin);
}

function parseApiError(data, fallback) {
  if (!data) return fallback;
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return fallback;
}

function showApiBanner(message) {
  const banner = document.getElementById("apiBanner");
  if (!banner) return;
  banner.innerHTML = message;
  banner.classList.remove("hidden");
  document.body.classList.add("has-api-banner");
}

function hideApiBanner() {
  const banner = document.getElementById("apiBanner");
  if (!banner) return;
  banner.classList.add("hidden");
  document.body.classList.remove("has-api-banner");
}

function showToast(message, type = "info", duration = 4200) {
  const host = els.toastHost || document.getElementById("toastHost");
  if (!host || !message) return;
  const el = document.createElement("div");
  el.className = `toast ${type === "ok" || type === "err" || type === "warn" ? type : ""}`;
  el.setAttribute("role", type === "err" ? "alert" : "status");
  el.textContent = message;
  host.appendChild(el);
  window.setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity 0.2s ease";
    window.setTimeout(() => el.remove(), 220);
  }, duration);
}

function confirmAction({
  title = "Confirm",
  message = "Are you sure?",
  confirmLabel = "Confirm",
  danger = false,
} = {}) {
  return new Promise((resolve) => {
    if (!els.confirmDialog) {
      resolve(window.confirm(message));
      return;
    }
    confirmResolver = resolve;
    if (els.confirmTitle) els.confirmTitle.textContent = title;
    if (els.confirmMessage) els.confirmMessage.textContent = message;
    if (els.confirmOk) {
      els.confirmOk.textContent = confirmLabel;
      els.confirmOk.classList.toggle("danger", Boolean(danger));
    }
    els.confirmDialog.showModal();
  });
}

function finishConfirm(result) {
  if (els.confirmDialog?.open) els.confirmDialog.close();
  if (confirmResolver) {
    const r = confirmResolver;
    confirmResolver = null;
    r(Boolean(result));
  }
}

function setBootReady() {
  els.bootSkeleton?.classList.add("hidden");
  const hasMsgs = Boolean(els.messages?.querySelector(".msg"));
  if (!hasMsgs) {
    els.welcome?.classList.remove("hidden");
  }
}

function updateWelcomeForIndex(indexed) {
  const empty = !indexed;
  els.welcomeEmptyIndex?.classList.toggle("hidden", !empty);
  els.welcomePromptsBlock?.classList.toggle("hidden", empty);
  els.btnWelcomeUpload?.classList.toggle("hidden", empty);
}

/** Returns true if API has workspaces + upload (latest code). */
async function probeApi(base) {
  try {
    const res = await fetch(`${base}/api/v1/meta`, {
      headers: { "X-Workspace-Id": "default" },
    });
    if (!res.ok) return false;
    const meta = await res.json();
    apiFeatures = meta?.features || {};
    updateSidebarMode();
    updateIngestTabsVisibility();
    return Boolean(meta?.features?.workspaces && meta?.features?.ingest_upload);
  } catch {
    return false;
  }
}

async function ensureApiReachable() {
  let base = getBaseUrl();
  if (await probeApi(base)) {
    hideApiBanner();
    return true;
  }
  const origin = normalizeBaseUrl(window.location.origin);
  if (origin !== base && (await probeApi(origin))) {
    saveSettings({ ...loadSettings(), apiBaseUrl: origin });
    hideApiBanner();
    return true;
  }
  showApiBanner(
    `API is outdated or wrong URL. Restart server, then open <code>${origin}/app/</code> — ` +
      `Settings → API URL = <code>${origin}</code>`
  );
  return false;
}

function getWorkspaceId() {
  return currentWorkspaceId;
}

function getHeaders(json = true) {
  const headers = {};
  if (json) headers["Content-Type"] = "application/json";
  const key = loadSettings().apiKey;
  if (key?.trim()) headers["X-API-Key"] = key.trim();
  headers["X-Workspace-Id"] = getWorkspaceId();
  return headers;
}

function chatStorageKey(ws = currentWorkspaceId) {
  return `${CHAT_PREFIX}${ws}`;
}

function parseStoredChat(raw, expectedWorkspaceId) {
  const data = JSON.parse(raw);
  // New format: { workspaceId, messages }
  if (data && typeof data === "object" && Array.isArray(data.messages)) {
    if (data.workspaceId && data.workspaceId !== expectedWorkspaceId) return [];
    return data.messages.filter((m) => m?.text?.trim());
  }
  // Legacy: plain array (only use if stored under this project's key)
  if (Array.isArray(data)) {
    return data.filter((m) => m?.text?.trim());
  }
  return [];
}

function getStoredMessages(wsId) {
  try {
    const raw = localStorage.getItem(chatStorageKey(wsId));
    if (!raw) return [];
    return parseStoredChat(raw, wsId);
  } catch {
    return [];
  }
}

function setStoredMessages(wsId, messages) {
  const trimmed = (messages || []).filter((m) => m?.text?.trim());
  if (!trimmed.length) {
    localStorage.removeItem(chatStorageKey(wsId));
    return;
  }
  localStorage.setItem(
    chatStorageKey(wsId),
    JSON.stringify({ workspaceId: wsId, messages: trimmed })
  );
}

function saveChatHistory() {
  if (!els.messages) return;
  const msgs = [];
  els.messages.querySelectorAll(".msg").forEach((msg) => {
    const bubble = msg.querySelector(".msg-bubble");
    if (!bubble) return;
    // Prefer raw markdown so lists/code survive a reload intact.
    const text = (bubble.dataset.raw || bubble.textContent || "").trim();
    if (!text) return;
    msgs.push({
      role: msg.classList.contains("user") ? "user" : "assistant",
      text,
    });
  });
  setStoredMessages(currentWorkspaceId, msgs);
}

/** Show only this project's chat — clears screen first, never mixes projects. */
function displayChatForWorkspace(wsId) {
  if (!els.messages) return;

  els.messages.innerHTML = "";
  els.welcome?.classList.remove("hidden");
  document.querySelector(".app-shell")?.classList.remove("citations-open");
  els.citationsPanel?.classList.add("hidden");
  if (els.citationsList) els.citationsList.innerHTML = "";

  try {
    const raw = localStorage.getItem(chatStorageKey(wsId));
    if (!raw) return;
    const msgs = parseStoredChat(raw, wsId);
    if (!msgs.length) return;
    hideWelcome();
    msgs.forEach((m) => {
      const { bubble } = createMessage(m.role, m.text, { persist: false });
      if (m.role === "assistant") renderRichAnswer(bubble, m.text, []);
    });
  } catch {
    localStorage.removeItem(chatStorageKey(wsId));
  }
  scrollToBottom();
}

function updateProjectHeader() {
  const ws = workspaces.find((w) => w.id === currentWorkspaceId);
  if (els.headerProjectName) {
    els.headerProjectName.textContent = ws ? ws.name : "Project";
  }
}

function updateIndexUI(indexed) {
  indexReady = Boolean(indexed);
  document.querySelectorAll("[data-index-badge]").forEach((el) => {
    el.className = `index-badge ${indexed ? "index-badge--ready" : "index-badge--empty"}`;
    el.textContent = indexed ? "Indexed" : "No index";
  });
  els.btnHeaderIngest?.classList.toggle("hidden", Boolean(indexed));
  if (els.headerSubtitle) {
    els.headerSubtitle.textContent = indexed
      ? "Grounded answers with citations from this project's documents"
      : "Add documents to this project to enable search and Q&A";
  }
  updateWelcomeForIndex(indexed);
}

function updateSidebarMode() {
  const serverThreads = Boolean(apiFeatures.chat_threads);
  els.btnManageHistory?.classList.toggle("hidden", serverThreads);
}

async function fetchWorkspaces() {
  const res = await fetch(`${getBaseUrl()}/api/v1/workspaces`, { headers: getHeaders() });
  if (!res.ok) throw new Error("Could not load projects");
  workspaces = await res.json();
  return workspaces;
}

function renderWorkspaceSelect() {
  if (!els.workspaceSelect) return;
  els.workspaceSelect.innerHTML = workspaces
    .map(
      (w) =>
        `<option value="${escapeHtml(w.id)}">${escapeHtml(w.name)}</option>`
    )
    .join("");
  els.workspaceSelect.value = currentWorkspaceId;
  updateWorkspaceActions();
}

function updateWorkspaceActions() {
  const isDefault = currentWorkspaceId === "default";
  if (els.btnDeleteWorkspace) {
    els.btnDeleteWorkspace.disabled = isDefault;
    els.btnDeleteWorkspace.title = isDefault
      ? "Default project cannot be deleted"
      : "Delete this project and all its data";
  }
}

async function switchWorkspace(wsId) {
  if (!wsId) return;
  if (wsId !== currentWorkspaceId) {
    saveChatHistory();
    currentWorkspaceId = wsId;
    localStorage.setItem(WORKSPACE_KEY, wsId);
  }
  renderWorkspaceSelect();
  currentServerThreadId = null;
  if (apiFeatures.chat_threads) {
    // Land on a fresh "new chat" view; past conversations stay in the sidebar.
    els.messages.innerHTML = "";
    els.welcome?.classList.remove("hidden");
    document.getElementById("followups")?.classList.add("hidden");
    await loadConversations();
  } else {
    displayChatForWorkspace(wsId);
  }
  updateProjectHeader();
  updateSidebarMode();
  checkHealth();
}

async function deleteCurrentWorkspace() {
  if (currentWorkspaceId === "default") {
    showToast("The default project cannot be deleted.", "warn");
    return;
  }
  if (!(await ensureApiReachable())) return;

  const ws = workspaces.find((w) => w.id === currentWorkspaceId);
  const label = ws?.name || currentWorkspaceId;
  const ok = await confirmAction({
    title: "Delete project",
    message: `Delete project "${label}"? This removes its documents, search index, and chat history. This cannot be undone.`,
    confirmLabel: "Delete project",
    danger: true,
  });
  if (!ok) return;

  els.projectMenu?.removeAttribute("open");
  const deletedId = currentWorkspaceId;
  try {
    const res = await fetch(
      `${getBaseUrl()}/api/v1/workspaces/${encodeURIComponent(deletedId)}`,
      { method: "DELETE", headers: getHeaders() }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(parseApiError(data, `Failed to delete project (HTTP ${res.status})`));
    }

    localStorage.removeItem(chatStorageKey(deletedId));
    await fetchWorkspaces();
    currentWorkspaceId = "default";
    localStorage.setItem(WORKSPACE_KEY, "default");
    renderWorkspaceSelect();
    currentServerThreadId = null;
    if (apiFeatures.chat_threads) {
      els.messages.innerHTML = "";
      els.welcome?.classList.remove("hidden");
      await loadConversations();
    } else {
      displayChatForWorkspace("default");
    }
    updateProjectHeader();
    checkHealth();
    showToast("Project deleted.", "ok");
  } catch (e) {
    showToast(e.message || "Could not delete project", "err");
  }
}

async function createWorkspace() {
  if (!(await ensureApiReachable())) return;

  const name = prompt("Project name (e.g. Payment API, Mobile App repo):");
  if (!name?.trim()) return;

  try {
    const res = await fetch(`${getBaseUrl()}/api/v1/workspaces`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ name: name.trim() }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(parseApiError(data, `Failed to create project (HTTP ${res.status})`));
    }

    currentWorkspaceId = data.id;
    localStorage.setItem(WORKSPACE_KEY, data.id);
    await fetchWorkspaces();
    renderWorkspaceSelect();
    localStorage.removeItem(chatStorageKey(data.id));
    currentServerThreadId = null;
    if (apiFeatures.chat_threads) {
      els.messages.innerHTML = "";
      els.welcome?.classList.remove("hidden");
      await loadConversations();
    } else {
      displayChatForWorkspace(data.id);
    }
    updateProjectHeader();
    checkHealth();
    showToast(`Project "${data.name}" created.`, "ok");
  } catch (e) {
    showToast(e.message || "Could not create project", "err");
  }
}

async function initWorkspaces() {
  const apiOk = await ensureApiReachable();
  if (apiOk) {
    try {
      await fetchWorkspaces();
    } catch {
      workspaces = [{ id: "default", name: "Default project", created_at: "" }];
    }
  } else {
    workspaces = [{ id: "default", name: "Default project", created_at: "" }];
  }
  const saved = localStorage.getItem(WORKSPACE_KEY) || "default";
  currentWorkspaceId = workspaces.some((w) => w.id === saved) ? saved : "default";
  renderWorkspaceSelect();
  if (apiOk && apiFeatures.chat_threads) {
    await loadConversations();
  } else {
    displayChatForWorkspace(currentWorkspaceId);
  }
  updateProjectHeader();
  purgeLegacyGlobalChat();
}

/** Remove old single-bucket chat key so it does not bleed into projects. */
function purgeLegacyGlobalChat() {
  localStorage.removeItem("kta_chat");
  localStorage.removeItem("kta_messages");
}

async function deleteServerThread(threadId) {
  if (!apiFeatures.chat_threads || !threadId) return;
  const ok = await confirmAction({
    title: "Delete conversation",
    message: "Delete this conversation? This cannot be undone.",
    confirmLabel: "Delete",
    danger: true,
  });
  if (!ok) return;
  try {
    const res = await fetch(`${getBaseUrl()}/api/v1/chats/${encodeURIComponent(threadId)}`, {
      method: "DELETE",
      headers: getHeaders(),
    });
    if (!res.ok) return;
    if (currentServerThreadId === threadId) {
      currentServerThreadId = null;
      els.messages.innerHTML = "";
      els.welcome?.classList.remove("hidden");
      document.getElementById("followups")?.classList.add("hidden");
    }
    await loadConversations();
    showToast("Conversation deleted.", "ok");
  } catch {
    /* optional */
  }
}

async function loadConversations() {
  const list = document.getElementById("conversationList");
  if (!list || !apiFeatures.chat_threads) return;

  try {
    const res = await fetch(
      `${getBaseUrl()}/api/v1/chats?workspace_id=${encodeURIComponent(getWorkspaceId())}`,
      { headers: getHeaders() }
    );
    if (!res.ok) return;
    const threads = await res.json();
    els.conversationsEmpty?.classList.toggle("hidden", threads.length > 0);
    list.innerHTML = threads
      .map((t) => {
        const active = t.id === currentServerThreadId ? " active" : "";
        return `<li class="conversation-item">
          <button type="button" class="conversation-select${active}" data-id="${escapeHtml(t.id)}">${escapeHtml(t.title)}</button>
          <button type="button" class="conversation-delete" data-delete-id="${escapeHtml(t.id)}" aria-label="Delete conversation" title="Delete">×</button>
        </li>`;
      })
      .join("");

  } catch {
    /* optional feature */
  }
}

function gitRepoDisplayName(url) {
  if (!url?.trim()) return "repository";
  return (
    url
      .trim()
      .replace(/\/$/, "")
      .replace(/\.git$/i, "")
      .split("/")
      .filter(Boolean)
      .pop() || "repository"
  );
}

function getIngestChatTitle(fileCount = null) {
  const gitUrl = els.gitCloneUrl?.value?.trim();
  if (gitUrl) {
    return `Codebase: ${gitRepoDisplayName(gitUrl)}`;
  }
  const localPath = els.localIngestPath?.value?.trim();
  if (localPath) {
    const parts = localPath.replace(/\\/g, "/").split("/").filter(Boolean);
    const name = parts[parts.length - 1] || "project";
    return `Codebase: ${name}`;
  }
  const n = fileCount ?? selectedFiles.length;
  if (n > 0) {
    const first = ingestRelativePath(selectedFiles[0]);
    const root = first.includes("/") ? first.split("/")[0] : first.replace(/\.[^.]+$/, "");
    if (n === 1) return root || "Uploaded file";
    return `${root || "Upload"} (${n} files)`;
  }
  const ws = workspaces.find((w) => w.id === currentWorkspaceId);
  return ws ? `${ws.name} — docs` : "Document upload";
}

function shouldStartIngestChat() {
  if (!apiFeatures.chat_threads) return false;
  return ingestUIMode === "project";
}

function getCurrentThreadDisplayTitle() {
  if (!currentServerThreadId) return "New conversation";
  const btn = document.querySelector(
    `.conversation-select[data-id="${CSS.escape(currentServerThreadId)}"]`
  );
  return btn?.textContent?.trim() || "Current conversation";
}

function getActiveProjectName() {
  const ws = workspaces.find((w) => w.id === currentWorkspaceId);
  return ws?.name || "Project";
}

function buildIngestSummaryMessage(fileCount, title, indexMessage) {
  const project = getActiveProjectName();
  const n = fileCount || 0;
  const base =
    indexMessage ||
    `Indexed ${n} file(s) into project “${project}”. Search is shared across this project; citations come from the indexed documents.`;

  if (ingestUIMode === "project") {
    return `${base}\n\nNew conversation: “${title}” — use this thread for questions about what you just uploaded.`;
  }
  return `${base}\n\nLinked to this conversation (“${title}”). Ask anything about the files you uploaded here.`;
}

function updateIngestContextUI() {
  const project = getActiveProjectName();
  if (els.ingestDialogTitle) {
    els.ingestDialogTitle.textContent =
      ingestUIMode === "chat"
        ? currentServerThreadId
          ? "Upload to this conversation"
          : "Upload to a new conversation"
        : "Add documents to project";
  }
  if (els.ingestContextProject) {
    els.ingestContextProject.textContent = `Project: ${project}`;
  }
  if (els.ingestContextChat) {
    if (ingestUIMode === "chat") {
      const chatTitle = getCurrentThreadDisplayTitle();
      els.ingestContextChat.textContent = currentServerThreadId
        ? `Files will be linked to: “${chatTitle}”`
        : "A conversation will be created for this upload, then linked here.";
    } else {
      els.ingestContextChat.textContent =
        "After indexing, a new conversation (named from your upload) appears in the sidebar.";
    }
  }
}

async function createServerThread(title = "New conversation", opts = {}) {
  const { silent = false } = opts;
  if (!apiFeatures.chat_threads) return null;
  const res = await fetch(`${getBaseUrl()}/api/v1/chats`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({
      workspace_id: getWorkspaceId(),
      title: (title || "New conversation").slice(0, 200),
    }),
  });
  const t = await res.json();
  if (!res.ok) return null;
  currentServerThreadId = t.id;
  if (!silent) {
    els.messages.innerHTML = "";
    els.welcome?.classList.remove("hidden");
    document.getElementById("followups")?.classList.add("hidden");
  }
  await loadConversations();
  return t;
}

/** ChatGPT-style titles: name the thread after its first user message. */
function titleFromQuestion(q) {
  const t = (q || "").replace(/\s+/g, " ").trim();
  if (!t) return "New conversation";
  return t.length > 48 ? `${t.slice(0, 48)}…` : t;
}

/**
 * Lazily create/rename the active thread when the user actually sends a message.
 * No thread exists until there is content — same behavior as ChatGPT.
 */
async function ensureThreadForMessage(question) {
  if (!apiFeatures.chat_threads) return;
  if (!currentServerThreadId) {
    await createServerThread(titleFromQuestion(question), { silent: true });
    return;
  }
  if (getCurrentThreadDisplayTitle() === "New conversation") {
    await updateServerThreadTitle(currentServerThreadId, titleFromQuestion(question));
  }
}

async function updateServerThreadTitle(threadId, title) {
  if (!apiFeatures.chat_threads || !threadId) return;
  await fetch(`${getBaseUrl()}/api/v1/chats/${encodeURIComponent(threadId)}`, {
    method: "PATCH",
    headers: getHeaders(),
    body: JSON.stringify({ title: title.slice(0, 200) }),
  });
  await loadConversations();
}

async function afterIngestStartConversation(fileCount, indexMessage) {
  if (!apiFeatures.chat_threads) return;
  const title = getIngestChatTitle(fileCount);
  const summary = buildIngestSummaryMessage(fileCount, title, indexMessage);

  if (shouldStartIngestChat()) {
    await createServerThread(title);
    await saveServerMessage("assistant", summary);
    await loadServerThreadMessages(currentServerThreadId);
  } else if (currentServerThreadId) {
    const curTitle = getCurrentThreadDisplayTitle();
    if (curTitle === "New conversation" || !curTitle.trim()) {
      await updateServerThreadTitle(currentServerThreadId, title);
    }
    await saveServerMessage("assistant", summary);
    await loadServerThreadMessages(currentServerThreadId);
  } else {
    await createServerThread(title);
    await saveServerMessage("assistant", summary);
    await loadServerThreadMessages(currentServerThreadId);
  }
  await loadConversations();
  scrollToBottom();
}

async function loadServerThreadMessages(threadId) {
  if (!apiFeatures.chat_threads || !threadId) return;
  const res = await fetch(`${getBaseUrl()}/api/v1/chats/${threadId}/messages`, {
    headers: getHeaders(),
  });
  if (!res.ok) return;
  const msgs = await res.json();
  els.messages.innerHTML = "";
  if (!msgs.length) {
    els.welcome?.classList.remove("hidden");
    return;
  }
  hideWelcome();
  msgs.forEach((m) => {
    const { bubble } = createMessage(m.role, m.content, { persist: false });
    if (m.role === "assistant") renderRichAnswer(bubble, m.content, []);
  });
}

async function saveServerMessage(role, content, threadId = null) {
  const target = threadId || currentServerThreadId;
  if (!apiFeatures.chat_threads || !target || !content?.trim()) return;
  await fetch(`${getBaseUrl()}/api/v1/chats/${target}/messages`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ role, content: content.trim() }),
  });
}

function showFollowups(suggestions) {
  const box = document.getElementById("followups");
  const chips = document.getElementById("followupChips");
  if (!box || !chips || !suggestions?.length) return;
  chips.innerHTML = "";
  suggestions.forEach((q) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      els.questionInput.value = q;
      autoResizeTextarea();
      els.btnSend.disabled = false;
      box.classList.add("hidden");
    });
    chips.appendChild(btn);
  });
  box.classList.remove("hidden");
}

async function fetchFollowups(question, answer) {
  if (!apiFeatures.followups) return;
  try {
    const res = await fetch(`${getBaseUrl()}/api/v1/suggest-followups`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ question, answer }),
    });
    const data = await res.json();
    if (res.ok) showFollowups(data.suggestions);
  } catch {
    /* optional */
  }
}

async function sendFeedback(wasHelpful, btn) {
  if (!lastQuestion) return;
  try {
    await fetch(`${getBaseUrl()}/api/v1/feedback`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        thread_id: currentServerThreadId || "web",
        query: lastQuestion,
        was_helpful: wasHelpful,
        workspace_id: getWorkspaceId(),
      }),
    });
    if (btn) {
      btn.classList.add("active");
      btn.textContent = wasHelpful ? "Thanks" : "Noted";
    }
    showToast(wasHelpful ? "Thanks — feedback saved." : "Feedback noted.", "ok");
  } catch {
    showToast("Could not save feedback.", "err");
  }
}

function hideWelcome() {
  els.welcome?.classList.add("hidden");
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.messagesWrap.scrollTop = els.messagesWrap.scrollHeight;
  });
}

function autoResizeTextarea() {
  const ta = els.questionInput;
  ta.style.height = "auto";
  ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
}

function setLoading(loading) {
  isLoading = loading;
  els.btnSend.disabled = !loading && !els.questionInput.value.trim();
  els.btnSend.classList.toggle("loading", loading);
  els.btnSend.querySelector(".icon-send")?.classList.toggle("hidden", loading);
  els.btnSend.querySelector(".icon-stop")?.classList.toggle("hidden", !loading);
  els.questionInput.disabled = loading;
  els.askProgress?.classList.toggle("hidden", !loading);
}

function createMessage(role, text = "", opts = {}) {
  const { persist = false } = opts;
  hideWelcome();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "You" : "AI";

  const body = document.createElement("div");
  body.className = "msg-body";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  if (text) bubble.dataset.raw = text;

  body.appendChild(bubble);
  wrap.appendChild(avatar);
  wrap.appendChild(body);
  els.messages.appendChild(wrap);
  scrollToBottom();
  if (persist) saveChatHistory();
  return { wrap, bubble, body };
}

function addMeta(body, { confidence, reflection, citations, onSources, answerText, latencyMs }) {
  const meta = document.createElement("div");
  meta.className = "msg-meta";

  const actions = document.createElement("div");
  actions.className = "msg-actions";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "msg-action-btn";
  copyBtn.textContent = "Copy";
  copyBtn.addEventListener("click", async () => {
    const text = answerText || body.querySelector(".msg-bubble")?.textContent || "";
    try {
      await navigator.clipboard?.writeText(text);
      copyBtn.textContent = "Copied";
      copyBtn.classList.add("active");
      window.setTimeout(() => {
        copyBtn.textContent = "Copy";
        copyBtn.classList.remove("active");
      }, 1600);
    } catch {
      showToast("Copy failed.", "err");
    }
  });
  actions.appendChild(copyBtn);

  const up = document.createElement("button");
  up.type = "button";
  up.className = "msg-action-btn";
  up.textContent = "Helpful";
  up.addEventListener("click", () => sendFeedback(true, up));
  actions.appendChild(up);

  const down = document.createElement("button");
  down.type = "button";
  down.className = "msg-action-btn";
  down.textContent = "Not helpful";
  down.addEventListener("click", () => sendFeedback(false, down));
  actions.appendChild(down);

  body.appendChild(actions);

  if (confidence != null) {
    const pct = Math.round(confidence * 100);
    const badge = document.createElement("span");
    badge.className = "badge";
    if (pct >= 75) badge.classList.add("confidence-high");
    else if (pct >= 50) badge.classList.add("confidence-mid");
    else badge.classList.add("confidence-low");
    badge.textContent = `Confidence ${pct}%`;
    meta.appendChild(badge);
  }

  if (reflection) {
    const r = document.createElement("span");
    r.className = "badge";
    r.textContent = reflection;
    meta.appendChild(r);
  }

  if (latencyMs != null && Number.isFinite(latencyMs)) {
    const l = document.createElement("span");
    l.className = "badge";
    l.textContent = `${Math.round(latencyMs)} ms`;
    meta.appendChild(l);
  }

  if (citations?.length) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-link";
    btn.textContent = `View ${citations.length} source${citations.length > 1 ? "s" : ""}`;
    btn.addEventListener("click", () => onSources?.(citations));
    meta.appendChild(btn);
  }

  if (meta.children.length) body.appendChild(meta);
}

function renderCitations(citations, focusIdx = null) {
  els.citationsList.innerHTML = "";
  if (!citations?.length) {
    els.citationsList.innerHTML = '<p class="meta" style="padding:1rem;color:var(--text-muted)">No sources for this answer.</p>';
    return;
  }

  // [N] refers to the N-th retrieved context chunk, so cards are numbered by position.
  citations.forEach((c, i) => {
    const idx = i + 1;
    const card = document.createElement("article");
    card.className = "citation-card";
    card.dataset.cite = String(idx);
    const name = c.source?.split("/").pop() || c.source || "Unknown";
    const snippet = c.snippet
      ? `<div class="citation-snippet">${escapeHtml(c.snippet)}</div>`
      : "";
    card.innerHTML = `
      <span class="idx">[${idx}]</span>
      <div class="source">${escapeHtml(name)}</div>
      <div class="meta">${escapeHtml(c.source_type || "doc")}${c.page_number ? ` · page ${c.page_number}` : ""}</div>
      ${snippet}
    `;
    els.citationsList.appendChild(card);
  });

  document.querySelector(".app-shell")?.classList.add("citations-open");
  els.citationsPanel.classList.remove("hidden");

  if (focusIdx != null) {
    const target = els.citationsList.querySelector(`[data-cite="${focusIdx}"]`);
    if (target) {
      requestAnimationFrame(() => {
        target.scrollIntoView({ block: "nearest", behavior: "smooth" });
        target.classList.add("citation-card--flash");
        window.setTimeout(() => target.classList.remove("citation-card--flash"), 1400);
      });
    }
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/** Lightweight markdown for RAG answers (safe: escaped first). */
function formatAnswerHtml(text) {
  if (!text) return "";
  let s = escapeHtml(text);

  // Pull code blocks out first so [N]/bold/inline-code rules never touch code.
  const codeBlocks = [];
  s = s.replace(/```([\s\S]*?)```/g, (_, code) => {
    codeBlocks.push(`<pre><code>${code.trim()}</code></pre>`);
    return `\u0000CODE${codeBlocks.length - 1}\u0000`;
  });
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\[(\d+)\]/g, '<button type="button" class="cite-chip" data-cite="$1">[$1]</button>');

  const blocks = s.split(/\n{2,}/);
  let html = blocks
    .map((block) => {
      const lines = block.split("\n");
      if (lines.every((l) => /^\s*[-*]\s+/.test(l) || !l.trim())) {
        const items = lines
          .filter((l) => l.trim())
          .map((l) => `<li>${l.replace(/^\s*[-*]\s+/, "")}</li>`)
          .join("");
        return items ? `<ul>${items}</ul>` : "";
      }
      if (lines.every((l) => /^\s*\d+\.\s+/.test(l) || !l.trim())) {
        const items = lines
          .filter((l) => l.trim())
          .map((l) => `<li>${l.replace(/^\s*\d+\.\s+/, "")}</li>`)
          .join("");
        return items ? `<ol>${items}</ol>` : "";
      }
      if (/^\u0000CODE\d+\u0000$/.test(block.trim())) return block.trim();
      return `<p>${block.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");

  html = html.replace(/\u0000CODE(\d+)\u0000/g, (_, i) => codeBlocks[Number(i)] || "");
  return html;
}

function renderRichAnswer(bubble, text, citations) {
  if (!bubble) return;
  bubble.classList.add("rich");
  bubble.dataset.raw = text || "";
  bubble.innerHTML = formatAnswerHtml(text);
  bubble.querySelectorAll(".cite-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (citations?.length) renderCitations(citations, Number(btn.dataset.cite));
      else showToast("Sources are shown right after an answer is generated.", "info");
    });
  });
}

function showError(message) {
  const { bubble } = createMessage("assistant", message);
  bubble.classList.add("msg-error");
  showToast(message, "err");
}

async function checkHealth() {
  const pill = els.healthStatus;
  const text = pill?.querySelector(".status-text");
  if (!pill || !text) return;

  try {
    const res = await fetch(`${getBaseUrl()}/api/v1/health`, { headers: getHeaders() });
    const data = await res.json();
    const indexed = Boolean(res.ok && data.vector_store_loaded);
    if (data.version) {
      appVersion = String(data.version);
      if (els.appVersionLabel) els.appVersionLabel.textContent = `v${appVersion}`;
      if (els.aboutVersion) els.aboutVersion.textContent = appVersion;
    }
    updateIndexUI(indexed);
    if (res.ok) hideApiBanner();
    if (res.ok && indexed) {
      pill.className = "status-pill ok";
      text.textContent = "API connected · index ready";
    } else if (res.ok) {
      pill.className = "status-pill warn";
      text.textContent = "API connected · index empty";
    } else {
      throw new Error(data.detail || res.statusText);
    }
  } catch {
    updateIndexUI(false);
    pill.className = "status-pill err";
    text.textContent = "API unreachable";
  }
}

async function askFull(question) {
  const res = await fetch(`${getBaseUrl()}/api/v1/ask`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ question, workspace_id: getWorkspaceId() }),
    signal: abortController?.signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

async function askStream(question, bubble) {
  const res = await fetch(`${getBaseUrl()}/api/v1/ask/stream`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ question, workspace_id: getWorkspaceId() }),
    signal: abortController?.signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Stream failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";
  let citations = [];

  bubble.classList.add("typing-cursor");

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const event = JSON.parse(line.slice(6));
        if (event.type === "token" && event.text) {
          fullText += event.text;
          bubble.textContent = fullText;
          scrollToBottom();
        } else if (event.type === "done") {
          citations = event.citations || [];
        } else if (event.type === "error") {
          throw new Error(event.error || "Stream error");
        }
      } catch (e) {
        if (e instanceof SyntaxError) continue;
        throw e;
      }
    }
  }

  bubble.classList.remove("typing-cursor");
  return { answer: fullText, citations, reflection_status: "streamed", confidence_score: null };
}

async function submitQuestion(question) {
  const q = question.trim();
  if (!q || isLoading) return;

  lastQuestion = q;
  document.getElementById("followups")?.classList.add("hidden");
  createMessage("user", q);
  try {
    await ensureThreadForMessage(q);
  } catch {
    /* thread bookkeeping is best-effort; the ask itself still proceeds */
  }
  // Pin this Q&A to the thread active at ask time, so the reply lands in the
  // right conversation even if the user starts a new chat while it streams.
  const askThreadId = currentServerThreadId;
  await saveServerMessage("user", q, askThreadId);
  els.questionInput.value = "";
  autoResizeTextarea();
  setLoading(true);
  abortController = new AbortController();

  const { bubble, body, wrap } = createMessage("assistant", "");
  const useStream = els.streamMode ? els.streamMode.checked : true;

  try {
    let result;
    if (useStream) {
      result = await askStream(q, bubble);
    } else {
      result = await askFull(q);
    }

    lastAnswer = result.answer || "";
    lastCitations = result.citations || [];
    renderRichAnswer(bubble, lastAnswer || "No answer returned.", lastCitations);
    await saveServerMessage("assistant", lastAnswer, askThreadId);
    addMeta(body, {
      confidence: result.confidence_score,
      reflection: result.reflection_status,
      citations: result.citations,
      onSources: renderCitations,
      answerText: lastAnswer,
      latencyMs: result.latency_ms,
    });
    saveChatHistory();
    await fetchFollowups(lastQuestion, lastAnswer);
    await loadConversations();
  } catch (e) {
    bubble.classList.remove("typing-cursor");
    if (e.name === "AbortError") {
      if (bubble.textContent.trim()) {
        bubble.dataset.raw = bubble.textContent;
        saveChatHistory();
      } else {
        wrap.remove();
        showToast("Stopped.", "info");
      }
    } else {
      wrap.remove();
      showError(typeof e.message === "string" ? e.message : "Something went wrong. Check API settings.");
    }
  } finally {
    setLoading(false);
    abortController = null;
    scrollToBottom();
  }
}

function useSamplePrompt(text) {
  els.questionInput.value = text;
  autoResizeTextarea();
  els.btnSend.disabled = false;
  els.questionInput.focus();
  els.sidebar?.classList.remove("open");
}

function initPromptChips() {
  const container = els.welcomePromptChips;
  if (!container) return;
  container.innerHTML = "";
  SAMPLE_PROMPTS.forEach((text) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = text;
    btn.addEventListener("click", () => useSamplePrompt(text));
    li.appendChild(btn);
    container.appendChild(li);
  });
}

async function resetChat(clearStorage = true) {
  if (!els.messages) return;
  document.getElementById("followups")?.classList.add("hidden");
  if (apiFeatures.chat_threads) {
    // ChatGPT-style: "New chat" just resets the view. The server thread is
    // created lazily when the first message or upload actually happens.
    currentServerThreadId = null;
    els.messages.innerHTML = "";
    els.welcome?.classList.remove("hidden");
    document.querySelector(".app-shell")?.classList.remove("citations-open");
    els.citationsPanel?.classList.add("hidden");
    await loadConversations();
  } else {
    if (clearStorage) localStorage.removeItem(chatStorageKey());
    displayChatForWorkspace(currentWorkspaceId);
  }
}

function openSettings() {
  const s = loadSettings();
  const base = getBaseUrl();
  if (els.apiBaseUrl) els.apiBaseUrl.value = s.apiBaseUrl || base;
  if (els.apiKey) els.apiKey.value = s.apiKey || "";
  if (els.streamMode) els.streamMode.checked = s.stream !== false;
  if (els.aboutEndpoint) els.aboutEndpoint.textContent = base;
  if (els.aboutVersion) els.aboutVersion.textContent = appVersion || "—";
  if (els.settingsConnStatus) {
    els.settingsConnStatus.className = "settings-conn";
    els.settingsConnStatus.textContent = "Checking connection…";
    fetch(`${base}/api/v1/health`, { headers: getHeaders() })
      .then((r) => {
        if (!els.settingsConnStatus) return;
        if (r.ok) {
          els.settingsConnStatus.className = "settings-conn ok";
          els.settingsConnStatus.textContent = "Connected to API";
        } else {
          els.settingsConnStatus.className = "settings-conn err";
          els.settingsConnStatus.textContent = `API returned ${r.status}`;
        }
      })
      .catch(() => {
        if (!els.settingsConnStatus) return;
        els.settingsConnStatus.className = "settings-conn err";
        els.settingsConnStatus.textContent = "Cannot reach API";
      });
  }
  els.settingsDialog.showModal();
}

function setLibraryTab(tab) {
  libraryTab = tab || "docs";
  document.querySelectorAll(".library-tab").forEach((btn) => {
    const active = btn.dataset.libraryTab === libraryTab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-library-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.libraryPanel !== libraryTab);
  });
}

async function openLibrary(tab = "docs") {
  els.projectMenu?.removeAttribute("open");
  const ws = workspaces.find((w) => w.id === currentWorkspaceId);
  if (els.librarySubtitle) {
    els.librarySubtitle.textContent = ws
      ? `${ws.name} · indexed sources, audit trail, shared memory`
      : "Indexed sources, audit, and shared memory";
  }
  setLibraryTab(tab);
  els.libraryDialog?.showModal();
  if (libraryTab === "docs") await loadLibraryDocs();
  else if (libraryTab === "audit") await loadLibraryAudit();
  else await loadLibraryMemory();
}

function renderLibraryDocs(filter = "") {
  const list = els.libraryDocList;
  if (!list) return;
  const q = filter.trim().toLowerCase();
  const items = libraryDocsCache.filter((s) => {
    if (!q) return true;
    return (
      (s.file_name || "").toLowerCase().includes(q) ||
      (s.source || "").toLowerCase().includes(q) ||
      (s.source_type || "").toLowerCase().includes(q)
    );
  });
  list.innerHTML = items
    .map(
      (s) => `
    <li class="library-item">
      <div class="library-item-title">${escapeHtml(s.file_name || "Untitled")}</div>
      <div class="library-item-meta">
        <span>${escapeHtml(s.source_type || "doc")}</span>
        <span>${s.chunk_count} chunk${s.chunk_count === 1 ? "" : "s"}</span>
      </div>
      <div class="library-item-path">${escapeHtml(s.source || "")}</div>
    </li>`
    )
    .join("");
  els.libraryDocEmpty?.classList.toggle("hidden", items.length > 0);
}

async function loadLibraryDocs() {
  if (els.libraryDocStats) els.libraryDocStats.textContent = "Loading index…";
  try {
    const res = await fetch(
      `${getBaseUrl()}/api/v1/documents?workspace_id=${encodeURIComponent(getWorkspaceId())}`,
      { headers: getHeaders() }
    );
    if (res.status === 404) {
      libraryDocsCache = [];
      if (els.libraryDocStats) els.libraryDocStats.textContent = "No index for this project yet";
      renderLibraryDocs(els.libraryDocSearch?.value || "");
      return;
    }
    const data = await res.json();
    if (!res.ok) throw new Error(parseApiError(data, "Could not load documents"));
    libraryDocsCache = data.sources || [];
    if (els.libraryDocStats) {
      els.libraryDocStats.textContent = `${data.source_count || 0} source${
        data.source_count === 1 ? "" : "s"
      } · ${data.total_chunks || 0} chunks indexed`;
    }
    renderLibraryDocs(els.libraryDocSearch?.value || "");
  } catch (e) {
    if (els.libraryDocStats) els.libraryDocStats.textContent = "Failed to load documents";
    showToast(e.message || "Could not load document library", "err");
  }
}

async function loadLibraryAudit() {
  const list = els.libraryAuditList;
  if (!list) return;
  list.innerHTML = `<li class="library-item"><div class="library-item-meta">Loading…</div></li>`;
  try {
    const res = await fetch(
      `${getBaseUrl()}/api/v1/audit/recent?limit=40&workspace_id=${encodeURIComponent(getWorkspaceId())}`,
      { headers: getHeaders() }
    );
    const rows = await res.json();
    if (!res.ok) throw new Error(parseApiError(rows, "Could not load audit"));
    if (!Array.isArray(rows) || !rows.length) {
      list.innerHTML = "";
      els.libraryAuditEmpty?.classList.remove("hidden");
      return;
    }
    els.libraryAuditEmpty?.classList.add("hidden");
    list.innerHTML = rows
      .map((r) => {
        const ok = Number(r.success) === 1;
        const conf =
          r.confidence_score != null ? `${Math.round(Number(r.confidence_score) * 100)}%` : "—";
        const lat = r.latency_ms != null ? `${Math.round(Number(r.latency_ms))} ms` : "—";
        return `
      <li class="library-item">
        <div class="library-item-title">${escapeHtml((r.question || "").slice(0, 160))}</div>
        <div class="library-item-meta">
          <span class="${ok ? "pill-ok" : "pill-err"}">${ok ? "ok" : "failed"}</span>
          <span>conf ${escapeHtml(String(conf))}</span>
          <span>${escapeHtml(String(r.citations_count ?? 0))} cites</span>
          <span>${escapeHtml(lat)}</span>
          <span>${escapeHtml((r.created_at || "").replace("T", " ").slice(0, 19))}</span>
        </div>
        ${
          r.answer
            ? `<div class="library-item-preview">${escapeHtml(String(r.answer).slice(0, 220))}</div>`
            : ""
        }
      </li>`;
      })
      .join("");
  } catch (e) {
    list.innerHTML = "";
    els.libraryAuditEmpty?.classList.remove("hidden");
    showToast(e.message || "Could not load audit", "err");
  }
}

async function loadLibraryMemory() {
  const list = els.libraryMemoryList;
  if (!list) return;
  if (apiFeatures.shared_memory === false) {
    list.innerHTML = "";
    if (els.libraryMemoryEmpty) {
      els.libraryMemoryEmpty.textContent = "Shared memory is disabled on this server.";
      els.libraryMemoryEmpty.classList.remove("hidden");
    }
    return;
  }
  list.innerHTML = `<li class="library-item"><div class="library-item-meta">Loading…</div></li>`;
  try {
    const res = await fetch(
      `${getBaseUrl()}/api/v1/memory?limit=50&workspace_id=${encodeURIComponent(getWorkspaceId())}`,
      { headers: getHeaders() }
    );
    const data = await res.json();
    if (!res.ok) throw new Error(parseApiError(data, "Could not load memory"));
    const items = data.items || [];
    if (!items.length) {
      list.innerHTML = "";
      if (els.libraryMemoryEmpty) {
        els.libraryMemoryEmpty.textContent = "No shared memories stored.";
        els.libraryMemoryEmpty.classList.remove("hidden");
      }
      return;
    }
    els.libraryMemoryEmpty?.classList.add("hidden");
    list.innerHTML = items
      .map(
        (m) => `
      <li class="library-item" data-memory-id="${escapeHtml(m.id)}">
        <div class="library-item-title">${escapeHtml(m.memory_type || "memory")}</div>
        <div class="library-item-meta">
          <span>${escapeHtml((m.updated_at || m.created_at || "").replace("T", " ").slice(0, 19))}</span>
        </div>
        <div class="library-item-preview">${escapeHtml((m.content || "").slice(0, 280))}</div>
        <div class="library-item-actions">
          <button type="button" class="msg-action-btn" data-delete-memory="${escapeHtml(m.id)}">Delete</button>
        </div>
      </li>`
      )
      .join("");
  } catch (e) {
    list.innerHTML = "";
    els.libraryMemoryEmpty?.classList.remove("hidden");
    showToast(e.message || "Could not load memory", "err");
  }
}

async function deleteMemory(memoryId) {
  const ok = await confirmAction({
    title: "Delete memory",
    message: "Remove this shared memory entry?",
    confirmLabel: "Delete",
    danger: true,
  });
  if (!ok) return;
  try {
    const res = await fetch(
      `${getBaseUrl()}/api/v1/memory/${encodeURIComponent(memoryId)}?workspace_id=${encodeURIComponent(getWorkspaceId())}`,
      { method: "DELETE", headers: getHeaders() }
    );
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(parseApiError(data, "Delete failed"));
    }
    showToast("Memory deleted.", "ok");
    await loadLibraryMemory();
  } catch (e) {
    showToast(e.message || "Could not delete memory", "err");
  }
}

async function clearAllMemory() {
  const ok = await confirmAction({
    title: "Clear shared memory",
    message: "Delete all shared memories for this project?",
    confirmLabel: "Clear all",
    danger: true,
  });
  if (!ok) return;
  try {
    const res = await fetch(
      `${getBaseUrl()}/api/v1/memory?workspace_id=${encodeURIComponent(getWorkspaceId())}`,
      { method: "DELETE", headers: getHeaders() }
    );
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(parseApiError(data, "Clear failed"));
    showToast(`Cleared ${data.deleted ?? 0} memories.`, "ok");
    await loadLibraryMemory();
  } catch (e) {
    showToast(e.message || "Could not clear memory", "err");
  }
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function ingestRelativePath(file) {
  return (file.webkitRelativePath || file.name || "").replace(/\\/g, "/");
}

function shouldIngestFile(file) {
  const rel = ingestRelativePath(file);
  if (!rel) return false;
  const parts = rel.split("/");
  for (const p of parts) {
    const low = p.toLowerCase();
    if (INGEST_SKIP_DIRS.has(low)) return false;
    if (low.endsWith(".egg-info")) return false;
  }
  const ext = rel.includes(".") ? `.${rel.split(".").pop().toLowerCase()}` : "";
  return INGEST_EXTENSIONS.has(ext);
}

function filterIngestFileList(fileList) {
  const accepted = [];
  let skipped = 0;
  for (const f of Array.from(fileList || [])) {
    if (!f.size && !f.name) continue;
    if (shouldIngestFile(f)) accepted.push(f);
    else skipped += 1;
  }
  return { accepted, skipped };
}

function updateIngestSummary() {
  if (!els.ingestSummary) return;
  if (!selectedFiles.length) {
    els.ingestSummary.textContent = "No files selected";
    return;
  }
  const parts = [`${selectedFiles.length} file(s) ready to index`];
  if (ingestSkippedCount > 0) {
    parts.push(`${ingestSkippedCount} skipped (unsupported or ignored folders)`);
  }
  els.ingestSummary.textContent = parts.join(" · ");
}

function renderFileList() {
  updateIngestSummary();
  if (!selectedFiles.length) {
    els.fileList?.classList.add("hidden");
    if (els.fileList) els.fileList.innerHTML = "";
    if (els.btnSubmitIngest) els.btnSubmitIngest.disabled = true;
    return;
  }
  els.fileList?.classList.remove("hidden");
  const preview = selectedFiles.slice(0, 8);
  const more = selectedFiles.length - preview.length;
  els.fileList.innerHTML =
    preview
      .map(
        (f) =>
          `<li><span>${escapeHtml(ingestRelativePath(f))}</span><span>${formatBytes(f.size)}</span></li>`
      )
      .join("") + (more > 0 ? `<li class="file-more">…and ${more} more</li>` : "");
  els.btnSubmitIngest.disabled = false;
}

function setIngestFiles(fileList) {
  const { accepted, skipped } = filterIngestFileList(fileList);
  selectedFiles = accepted;
  ingestSkippedCount = skipped;
  renderFileList();
  els.ingestStatus?.classList.add("hidden");
}

function newUploadBatchId() {
  const hex = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  return `batch_${hex}`;
}

function showIngestProgressPanel(show = true) {
  els.ingestProgressPanel?.classList.toggle("hidden", !show);
}

function setUploadProgress(percent, detail) {
  const pct = Math.min(100, Math.max(0, Math.round(percent)));
  if (els.uploadProgressFill) els.uploadProgressFill.style.width = `${pct}%`;
  if (els.uploadPercent) els.uploadPercent.textContent = `${pct}%`;
  if (els.uploadProgressDetail && detail) els.uploadProgressDetail.textContent = detail;
  const track = els.uploadProgressFill?.parentElement;
  if (track) track.setAttribute("aria-valuenow", String(pct));
}

function setIndexProgress(percent, detail, current = 0, total = 0) {
  els.indexProgressBlock?.classList.remove("hidden");
  const pct = Math.min(100, Math.max(0, Math.round(percent)));
  if (els.indexProgressFill) els.indexProgressFill.style.width = `${pct}%`;
  if (els.indexPercent) els.indexPercent.textContent = `${pct}%`;
  let text = detail || "Indexing…";
  if (total > 0) {
    text = `${detail || "Indexing"} · ${current} / ${total}`;
  }
  if (els.indexProgressDetail) els.indexProgressDetail.textContent = text;
  const track = els.indexProgressFill?.parentElement;
  if (track) track.setAttribute("aria-valuenow", String(pct));
}

function resetIngestProgress() {
  showIngestProgressPanel(false);
  els.uploadProgressBlock?.classList.remove("hidden");
  els.indexProgressBlock?.classList.add("hidden");
  setUploadProgress(0, "Waiting…");
  setIndexProgress(0, "Waiting to start…");
}

async function pollIngestJob(jobId) {
  showIngestProgressPanel(true);
  els.indexProgressBlock?.classList.remove("hidden");
  setIndexProgress(0, "Starting indexing job…");

  const poll = async () => {
    const jr = await fetch(`${getBaseUrl()}/api/v1/ingest/jobs/${jobId}`, {
      headers: getHeaders(),
    });
    const job = await jr.json();
    if (!jr.ok) throw new Error("Job status failed");

    const prog = job.result?.progress;
    if (prog) {
      setIndexProgress(
        prog.percent ?? 0,
        prog.message || job.message,
        prog.current ?? 0,
        prog.total ?? 0
      );
    } else if (job.message) {
      setIndexProgress(5, job.message);
    }

    if (job.status === "completed") {
      setIndexProgress(100, "Indexing complete", 1, 1);
      setIngestStatus("ok", job.message || "Indexing completed.");
      checkHealth();
      const fileTotal = job.result?.files_received || selectedFiles.length || 0;
      await afterIngestStartConversation(fileTotal, job.message);
      setTimeout(() => {
        els.ingestDialog.close();
        selectedFiles = [];
        ingestSkippedCount = 0;
        renderFileList();
        resetIngestProgress();
      }, 1200);
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.message || "Ingestion failed");
    }
    setTimeout(poll, 1000);
  };
  await poll();
}

async function verifyIngestEndpoint() {
  const url = `${getBaseUrl()}/api/v1/ingest/upload`;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: getHeaders(false),
      body: new FormData(),
    });
    // Empty body → 422 validation error means route exists; 404 means wrong server/URL
    return res.status !== 404;
  } catch {
    return false;
  }
}

async function openIngest(mode = "project") {
  if (!(await ensureApiReachable())) {
    showToast("API unreachable. Restart the server, then reload this page.", "err");
    return;
  }

  ingestUIMode = mode === "chat" ? "chat" : "project";
  els.projectMenu?.removeAttribute("open");
  selectedFiles = [];
  ingestSkippedCount = 0;
  if (els.ingestFiles) els.ingestFiles.value = "";
  if (els.ingestFolder) els.ingestFolder.value = "";
  if (els.replaceIndex) els.replaceIndex.checked = false;
  els.ingestStatus?.classList.add("hidden");
  resetIngestProgress();
  setIngestTab("folder");
  updateIngestTabsVisibility();
  updateIngestContextUI();
  renderFileList();
  els.ingestDialog.showModal();

  const ok = await verifyIngestEndpoint();
  if (!ok) {
    setIngestStatus(
      "err",
      `Upload API not found. Settings → API URL = ${getBaseUrl()} then restart server.`
    );
    els.btnSubmitIngest.disabled = true;
  } else {
    els.btnSubmitIngest.disabled = !selectedFiles.length;
  }
}

async function openIngestFromChat() {
  if (!(await ensureApiReachable())) return;
  // No eager thread creation: if there is no active chat, the conversation is
  // created (and titled from the upload) only after ingestion succeeds.
  await openIngest("chat");
}

function setIngestStatus(kind, message) {
  els.ingestStatus.classList.remove("hidden", "ok", "err", "loading");
  els.ingestStatus.classList.add(kind);
  els.ingestStatus.textContent = message;
}

function truncatePreview(text, max = 120) {
  const t = (text || "").trim();
  return t.length <= max ? t : `${t.slice(0, max)}…`;
}

function renderHistoryManager() {
  const ws = workspaces.find((w) => w.id === currentWorkspaceId);
  const projectName = ws ? ws.name : currentWorkspaceId;
  if (els.historyCurrentProjectTitle) {
    els.historyCurrentProjectTitle.textContent = `This project: ${projectName}`;
  }

  const messages = getStoredMessages(currentWorkspaceId);
  if (els.historyMsgList) {
    els.historyMsgList.innerHTML = messages
      .map(
        (m, idx) => `
      <li class="history-item">
        <input type="checkbox" class="hist-msg-cb" data-idx="${idx}" />
        <div class="history-item-body">
          <span class="history-item-role ${m.role}">${m.role === "user" ? "You" : "AI"}</span>
          <div class="history-item-preview">${escapeHtml(truncatePreview(m.text))}</div>
        </div>
      </li>`
      )
      .join("");
  }
  if (els.historyMsgEmpty) {
    els.historyMsgEmpty.classList.toggle("hidden", messages.length > 0);
  }

  if (els.historyProjectList) {
    els.historyProjectList.innerHTML = workspaces
      .map((w) => {
        const count = getStoredMessages(w.id).length;
        const isCurrent = w.id === currentWorkspaceId;
        return `
      <li class="history-item history-project-item">
        <input type="checkbox" class="hist-proj-cb" data-ws="${escapeHtml(w.id)}" aria-label="Select ${escapeHtml(w.name)}" />
        <div class="history-item-body">
          <div class="history-project-name">${escapeHtml(w.name)}${isCurrent ? ' <span class="history-current-tag">(current)</span>' : ""}</div>
          <div class="history-item-meta">${count} message${count === 1 ? "" : "s"}</div>
        </div>
      </li>`;
      })
      .join("");
  }
}

function openHistoryManager() {
  saveChatHistory();
  renderHistoryManager();
  els.historyDialog?.showModal();
}

function closeHistoryManager() {
  els.historyDialog?.close();
}

function getSelectedMessageIndexes() {
  return Array.from(document.querySelectorAll(".hist-msg-cb:checked"))
    .map((el) => parseInt(el.dataset.idx, 10))
    .filter((n) => !Number.isNaN(n))
    .sort((a, b) => b - a);
}

async function deleteSelectedMessages() {
  const indexes = getSelectedMessageIndexes();
  if (!indexes.length) {
    showToast("Select at least one message to delete.", "warn");
    return;
  }
  const ok = await confirmAction({
    title: "Delete messages",
    message: `Delete ${indexes.length} selected message(s) from this project?`,
    confirmLabel: "Delete",
    danger: true,
  });
  if (!ok) return;

  const msgs = getStoredMessages(currentWorkspaceId);
  indexes.forEach((i) => {
    if (i >= 0 && i < msgs.length) msgs.splice(i, 1);
  });
  setStoredMessages(currentWorkspaceId, msgs);
  displayChatForWorkspace(currentWorkspaceId);
  renderHistoryManager();
  showToast("Messages deleted.", "ok");
}

async function clearAllMessagesCurrentProject() {
  const ws = workspaces.find((w) => w.id === currentWorkspaceId);
  const label = ws ? ws.name : currentWorkspaceId;
  const ok = await confirmAction({
    title: "Clear chat history",
    message: `Clear all chat history for "${label}"?`,
    confirmLabel: "Clear all",
    danger: true,
  });
  if (!ok) return;
  localStorage.removeItem(chatStorageKey(currentWorkspaceId));
  displayChatForWorkspace(currentWorkspaceId);
  renderHistoryManager();
  showToast("Chat history cleared.", "ok");
}

async function deleteSelectedProjectChats() {
  const selected = Array.from(document.querySelectorAll(".hist-proj-cb:checked")).map(
    (el) => el.dataset.ws
  );
  if (!selected.length) {
    showToast("Select at least one project.", "warn");
    return;
  }
  const ok = await confirmAction({
    title: "Clear project chats",
    message: `Clear chat history for ${selected.length} project(s)?`,
    confirmLabel: "Clear",
    danger: true,
  });
  if (!ok) return;

  selected.forEach((wsId) => localStorage.removeItem(chatStorageKey(wsId)));
  displayChatForWorkspace(currentWorkspaceId);
  renderHistoryManager();
  showToast("Selected project chats cleared.", "ok");
}

function setAllCheckboxes(selector, checked) {
  document.querySelectorAll(selector).forEach((el) => {
    el.checked = checked;
  });
}

async function clearIndexedData() {
  if (!(await ensureApiReachable())) return;

  const ws = workspaces.find((w) => w.id === currentWorkspaceId);
  const label = ws ? ws.name : currentWorkspaceId;
  els.projectMenu?.removeAttribute("open");
  const ok = await confirmAction({
    title: "Clear search index",
    message: `Clear search index for "${label}"? Removes the FAISS index for this project only. Uploaded files remain — re-index to search again.`,
    confirmLabel: "Clear index",
    danger: true,
  });
  if (!ok) return;

  try {
    const res = await fetch(`${getBaseUrl()}/api/v1/index/reset`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        workspace_id: getWorkspaceId(),
        all_workspaces: false,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(parseApiError(data, `Reset failed (HTTP ${res.status})`));
    }
    showToast(data.message || "Index cleared.", "ok");
    resetChat(true);
    checkHealth();
  } catch (e) {
    showToast(e.message || "Could not clear index", "err");
  }
}

async function uploadFilesInChunks(files, replaceIndex) {
  const batchId = newUploadBatchId();
  const fileChunks = [];
  for (let i = 0; i < files.length; i += INGEST_CHUNK_SIZE) {
    fileChunks.push(files.slice(i, i + INGEST_CHUNK_SIZE));
  }

  const totalBytes = files.reduce((sum, f) => sum + (f.size || 0), 0);
  const useBytes = totalBytes > 0;
  let completedFiles = 0;
  let lastData = null;

  showIngestProgressPanel(true);
  els.indexProgressBlock?.classList.add("hidden");
  setUploadProgress(0, `Uploading 0 / ${files.length} files…`);
  setIngestStatus("loading", "Uploading files…");

  const uploadUrl = `${getBaseUrl()}/api/v1/ingest/upload`;
  const baseHeaders = getHeaders(false);

  for (let i = 0; i < fileChunks.length; i++) {
    const chunk = fileChunks[i];
    const form = new FormData();
    chunk.forEach((f) => {
      const rel = ingestRelativePath(f);
      form.append("files", f, rel);
    });
    form.append("upload_batch_id", batchId);
    form.append("start_ingest", i === fileChunks.length - 1 ? "true" : "false");
    form.append("replace_index", replaceIndex ? "true" : "false");
    form.append("workspace_id", getWorkspaceId());

    const xhr = new XMLHttpRequest();
    const { data, status } = await new Promise((resolve, reject) => {
      xhr.open("POST", uploadUrl);
      Object.entries(baseHeaders).forEach(([k, v]) => {
        if (v) xhr.setRequestHeader(k, v);
      });
      xhr.upload.onprogress = (e) => {
        if (useBytes && e.lengthComputable) {
          const doneBefore = files
            .slice(0, completedFiles)
            .reduce((s, f) => s + (f.size || 0), 0);
          const overall = doneBefore + e.loaded;
          const pct = Math.round((overall / totalBytes) * 100);
          setUploadProgress(
            pct,
            `Uploaded ${formatBytes(overall)} / ${formatBytes(totalBytes)}`
          );
        }
      };
      xhr.onload = () => {
        let parsed = {};
        try {
          parsed = JSON.parse(xhr.responseText || "{}");
        } catch {
          parsed = {};
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve({ data: parsed, status: xhr.status });
          return;
        }
        const err = new Error(parseApiError(parsed, `Upload failed (HTTP ${xhr.status})`));
        err.status = xhr.status;
        err.data = parsed;
        reject(err);
      };
      xhr.onerror = () => reject(new Error("Network error during upload"));
      xhr.send(form);
    }).catch((err) => {
      if (err.status === 404) {
        throw new Error(
          `Not Found — check Settings → API URL is ${getBaseUrl()} (not /app). Restart API.`
        );
      }
      throw err;
    });

    completedFiles += chunk.length;
    if (!useBytes) {
      const pct = Math.round((completedFiles / files.length) * 100);
      setUploadProgress(
        pct,
        `Uploaded ${completedFiles} / ${files.length} files (batch ${i + 1}/${fileChunks.length})`
      );
    }
    lastData = data;
  }

  setUploadProgress(100, `Uploaded ${files.length} file(s)`);
  return lastData;
}

async function submitGitCloneIngest() {
  const repoUrl = els.gitCloneUrl?.value?.trim();
  if (!repoUrl) {
    showToast("Enter a Git repository URL (e.g. https://github.com/org/repo.git).", "warn");
    return;
  }
  if (!(await ensureApiReachable())) return;

  els.btnGitClone.disabled = true;
  showIngestProgressPanel(true);
  els.uploadProgressBlock?.classList.add("hidden");
  els.indexProgressBlock?.classList.remove("hidden");
  setIndexProgress(0, "Cloning repository…");
  setIngestStatus("loading", "Cloning and indexing…");

  const branch = els.gitCloneBranch?.value?.trim();

  try {
    const res = await fetch(`${getBaseUrl()}/api/v1/ingest/git-clone`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        repo_url: repoUrl,
        branch: branch || null,
        workspace_id: getWorkspaceId(),
        replace_index: els.replaceIndex?.checked ?? false,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(parseApiError(data, `Git clone failed (HTTP ${res.status})`));
    }
    if (data.job_id) {
      await pollIngestJob(data.job_id);
      return;
    }
    setIngestStatus("ok", data.message || "Queued.");
    checkHealth();
  } catch (err) {
    setIngestStatus("err", err.message || "Git clone failed");
  } finally {
    els.btnGitClone.disabled = false;
  }
}

async function submitLocalPathIngest() {
  const path = els.localIngestPath?.value?.trim();
  if (!path) {
    showToast("Enter the full path to your project folder on this machine.", "warn");
    return;
  }
  if (!(await ensureApiReachable())) return;

  els.btnLocalIngest.disabled = true;
  showIngestProgressPanel(true);
  els.uploadProgressBlock?.classList.add("hidden");
  els.indexProgressBlock?.classList.remove("hidden");
  setIndexProgress(0, "Queuing folder ingest…");
  setIngestStatus("loading", "Indexing from local path…");

  try {
    const res = await fetch(`${getBaseUrl()}/api/v1/ingest/local-path`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        path,
        workspace_id: getWorkspaceId(),
        replace_index: els.replaceIndex?.checked ?? false,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(parseApiError(data, `Ingest failed (HTTP ${res.status})`));
    }
    if (data.job_id) {
      await pollIngestJob(data.job_id);
      return;
    }
    setIngestStatus("ok", data.message || "Queued.");
    checkHealth();
  } catch (err) {
    setIngestStatus("err", err.message || "Ingestion failed");
  } finally {
    els.btnLocalIngest.disabled = false;
  }
}

async function submitIngest(e) {
  e.preventDefault();
  if (!selectedFiles.length) return;
  if (!(await ensureApiReachable())) return;

  els.btnSubmitIngest.disabled = true;
  setIngestStatus("loading", "Preparing upload…");
  resetIngestProgress();

  try {
    const data = await uploadFilesInChunks(
      selectedFiles,
      els.replaceIndex?.checked ?? false
    );
    if (data?.job_id && apiFeatures.ingest_jobs) {
      setIngestStatus("loading", data.message || "Indexing…");
      await pollIngestJob(data.job_id);
      return;
    }
    setIngestStatus("ok", data?.message || "Upload complete.");
    checkHealth();
    await afterIngestStartConversation(
      data?.files_received || selectedFiles.length,
      data?.message
    );
    setTimeout(() => {
      els.ingestDialog.close();
      selectedFiles = [];
      ingestSkippedCount = 0;
      renderFileList();
    }, 1200);
  } catch (err) {
    setIngestStatus("err", err.message || "Ingestion failed");
    els.btnSubmitIngest.disabled = false;
  }
}

function setIngestTab(tab) {
  document.querySelectorAll(".ingest-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.ingestTab === tab);
  });
  els.ingestPanelFolder?.classList.toggle("hidden", tab !== "folder");
  els.ingestPanelGit?.classList.toggle("hidden", tab !== "git");
  els.ingestPanelLocal?.classList.toggle("hidden", tab !== "local");
  els.btnSubmitIngest?.classList.toggle("hidden", tab !== "folder");
}

function updateIngestTabsVisibility() {
  const showLocal = Boolean(apiFeatures.ingest_local_path);
  const showGit = Boolean(apiFeatures.ingest_git_clone);
  els.tabLocalPath?.classList.toggle("hidden", !showLocal);
  els.tabGitClone?.classList.toggle("hidden", !showGit);
  const active = document.querySelector(".ingest-tab.active")?.dataset.ingestTab;
  if (active === "local" && !showLocal) setIngestTab("folder");
  if (active === "git" && !showGit) setIngestTab("folder");
}

function getActiveIngestTab() {
  return document.querySelector(".ingest-tab.active")?.dataset.ingestTab || "folder";
}

function bindEvents() {
  if (!els.composerForm || !els.btnNewChat) {
    console.error("UI failed to bind: missing DOM elements");
    return;
  }

els.composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  if (isLoading) {
    abortController?.abort();
    return;
  }
  submitQuestion(els.questionInput.value);
});

els.questionInput.addEventListener("input", () => {
  autoResizeTextarea();
  if (!isLoading) els.btnSend.disabled = !els.questionInput.value.trim();
});

els.questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composerForm.requestSubmit();
  }
});

els.btnNewChat.addEventListener("click", () => {
  resetChat(true);
});
document.getElementById("conversationList")?.addEventListener("click", async (e) => {
  const del = e.target.closest("[data-delete-id]");
  if (del) {
    e.stopPropagation();
    await deleteServerThread(del.dataset.deleteId);
    return;
  }
  const btn = e.target.closest(".conversation-select[data-id]");
  if (!btn) return;
  saveChatHistory();
  currentServerThreadId = btn.dataset.id;
  await loadServerThreadMessages(currentServerThreadId);
  await loadConversations();
});
els.btnManageHistory?.addEventListener("click", openHistoryManager);
els.btnCloseHistory?.addEventListener("click", closeHistoryManager);
els.btnCloseHistoryDone?.addEventListener("click", closeHistoryManager);
els.histMsgSelectAll?.addEventListener("click", () => setAllCheckboxes(".hist-msg-cb", true));
els.histMsgSelectNone?.addEventListener("click", () => setAllCheckboxes(".hist-msg-cb", false));
els.histMsgDeleteSelected?.addEventListener("click", deleteSelectedMessages);
els.histMsgClearAll?.addEventListener("click", clearAllMessagesCurrentProject);
els.histProjSelectAll?.addEventListener("click", () => setAllCheckboxes(".hist-proj-cb", true));
els.histProjSelectNone?.addEventListener("click", () => setAllCheckboxes(".hist-proj-cb", false));
els.histProjDeleteSelected?.addEventListener("click", deleteSelectedProjectChats);
els.historyMsgList?.addEventListener("click", (e) => {
  const item = e.target.closest(".history-item");
  if (!item || e.target.matches('input[type="checkbox"]')) return;
  const cb = item.querySelector(".hist-msg-cb");
  if (cb) cb.checked = !cb.checked;
});
els.historyProjectList?.addEventListener("click", (e) => {
  const item = e.target.closest(".history-item");
  if (!item || e.target.matches('input[type="checkbox"]')) return;
  const cb = item.querySelector(".hist-proj-cb");
  if (cb) cb.checked = !cb.checked;
});
els.workspaceSelect?.addEventListener("change", (e) => switchWorkspace(e.target.value));
els.btnNewWorkspace?.addEventListener("click", createWorkspace);
els.btnDeleteWorkspace?.addEventListener("click", deleteCurrentWorkspace);
els.btnCloseCitations.addEventListener("click", () => {
  document.querySelector(".app-shell")?.classList.remove("citations-open");
  els.citationsPanel.classList.add("hidden");
});

els.btnIngest?.addEventListener("click", () => openIngest("project"));
els.btnHeaderIngest?.addEventListener("click", () => openIngest("project"));
els.btnChatUpload?.addEventListener("click", openIngestFromChat);
els.btnWelcomeUpload?.addEventListener("click", openIngestFromChat);
els.btnWelcomeIndex?.addEventListener("click", () => openIngest("project"));
els.btnClearIndex?.addEventListener("click", clearIndexedData);
els.btnCloseIngest.addEventListener("click", () => els.ingestDialog.close());
els.btnCancelIngest.addEventListener("click", () => els.ingestDialog.close());
els.ingestForm.addEventListener("submit", submitIngest);
els.ingestFiles?.addEventListener("change", (e) => {
  setIngestFiles(e.target.files);
  e.target.value = "";
});
els.ingestFolder?.addEventListener("change", (e) => {
  setIngestFiles(e.target.files);
  e.target.value = "";
});
els.btnLocalIngest?.addEventListener("click", submitLocalPathIngest);
els.btnGitClone?.addEventListener("click", submitGitCloneIngest);
document.querySelectorAll(".ingest-tab").forEach((btn) => {
  btn.addEventListener("click", () => setIngestTab(btn.dataset.ingestTab));
});

els.btnSettings.addEventListener("click", openSettings);
els.btnCloseSettings.addEventListener("click", () => els.settingsDialog.close());
els.btnLibrary?.addEventListener("click", () => openLibrary("docs"));
els.btnOpenLibrary?.addEventListener("click", () => openLibrary("docs"));
els.btnCloseLibrary?.addEventListener("click", () => els.libraryDialog?.close());
els.btnCloseLibraryDone?.addEventListener("click", () => els.libraryDialog?.close());
els.btnRefreshDocs?.addEventListener("click", () => loadLibraryDocs());
els.btnRefreshAudit?.addEventListener("click", () => loadLibraryAudit());
els.btnRefreshMemory?.addEventListener("click", () => loadLibraryMemory());
els.btnClearMemory?.addEventListener("click", () => clearAllMemory());
els.libraryDocSearch?.addEventListener("input", (e) => renderLibraryDocs(e.target.value));
document.querySelectorAll(".library-tab").forEach((btn) => {
  btn.addEventListener("click", async () => {
    setLibraryTab(btn.dataset.libraryTab);
    if (libraryTab === "docs") await loadLibraryDocs();
    else if (libraryTab === "audit") await loadLibraryAudit();
    else await loadLibraryMemory();
  });
});
els.libraryMemoryList?.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-delete-memory]");
  if (!btn) return;
  deleteMemory(btn.getAttribute("data-delete-memory"));
});

els.confirmCancel?.addEventListener("click", (e) => {
  e.preventDefault();
  finishConfirm(false);
});
els.confirmForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  finishConfirm(true);
});
els.confirmDialog?.addEventListener("cancel", (e) => {
  e.preventDefault();
  finishConfirm(false);
});

els.settingsForm.addEventListener("submit", (e) => {
  e.preventDefault();
  saveSettings({
    apiBaseUrl: els.apiBaseUrl?.value?.trim() || getBaseUrl(),
    apiKey: els.apiKey?.value || "",
    stream: els.streamMode?.checked !== false,
  });
  els.settingsDialog.close();
  showToast("Settings saved.", "ok");
  checkHealth();
});

document.addEventListener("click", (e) => {
  if (els.projectMenu?.open && !els.projectMenu.contains(e.target)) {
    els.projectMenu.removeAttribute("open");
  }
});

els.btnToggleSidebar?.addEventListener("click", () => {
  els.sidebar?.classList.toggle("open");
});

document.addEventListener("click", (e) => {
  if (
    window.innerWidth <= 900 &&
    els.sidebar?.classList.contains("open") &&
    !els.sidebar.contains(e.target) &&
    !els.btnToggleSidebar?.contains(e.target)
  ) {
    els.sidebar.classList.remove("open");
  }
});
}

function applyTheme(theme) {
  const t = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem(THEME_KEY, t);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", t === "light" ? "#e2e8f0" : "#0a0e17");
}

function initTheme() {
  els.btnTheme?.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(current === "dark" ? "light" : "dark");
  });
}

function boot() {
  const s = loadSettings();
  if (els.streamMode) els.streamMode.checked = s.stream !== false;
  initTheme();
  bindEvents();
  initPromptChips();
  autoResizeTextarea();
  updateSidebarMode();
  updateIngestTabsVisibility();
  initWorkspaces()
    .then(() => checkHealth())
    .finally(() => setBootReady());
  setInterval(checkHealth, 30000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
