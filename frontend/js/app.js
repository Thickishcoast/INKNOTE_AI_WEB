"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  noteList: $("#noteList"),
  noteFeed: $("#noteFeed"),
  aiCanvasPane: $("#aiCanvasPane"),
  conversationLog: $("#conversationLog"),
  noteTitle: $("#noteTitle"),
  saveStatus: $("#saveStatus"),
  newNoteButton: $("#newNoteButton"),
  deleteNoteButton: $("#deleteNoteButton"),
  clearConversationButton: $("#clearConversationButton"),
  connectionDot: $("#connectionDot"),
  connectionText: $("#connectionText"),
  modelSummary: $("#modelSummary"),
  handwritingCanvas: $("#handwritingCanvas"),
  handwritingWrap: $("#handwritingCanvasWrap"),
  processingOverlay: $("#processingOverlay"),
  processingText: $("#processingText"),
  canvasHint: $("#canvasHint"),
  sendCanvasButton: $("#sendCanvasButton"),
  handPenButton: $("#handPenButton"),
  handEraserButton: $("#handEraserButton"),
  handUndoButton: $("#handUndoButton"),
  handRedoButton: $("#handRedoButton"),
  handClearButton: $("#handClearButton"),
  canvasLayoutButton: $("#canvasLayoutButton"),
  canvasLayoutLabel: $("#canvasLayoutLabel"),
  inferenceStepsSelect: $("#inferenceStepsSelect"),
  handColorPalette: $("#handColorPalette"),
  customColorControl: $("#customColorControl"),
  customColorSwatch: $("#customColorSwatch"),
  customColorPopover: $("#customColorPopover"),
  customColorSvField: $("#customColorSvField"),
  customColorCursor: $("#customColorCursor"),
  customColorHue: $("#customColorHue"),
  customColorValue: $("#customColorValue"),
  toast: $("#toast"),
};

const state = {
  notes: [],
  note: null,
  saveTimer: null,
  toastTimer: null,
  busy: false,
};

const customColor = {
  hue: 225,
  saturation: 20 / 56,
  value: 56 / 255,
  dragging: false,
};

const CANVAS_LINES_STORAGE_KEY = "inknote-canvas-lines";
const INFERENCE_STEPS_STORAGE_KEY = "inknote-inference-steps";
const ALLOWED_INFERENCE_STEPS = new Set([10, 25, 50]);

function uniqueId(prefix = "block") {
  if (globalThis.crypto?.randomUUID) return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function showToast(message, duration = 4000) {
  clearTimeout(state.toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.remove("hidden");
  state.toastTimer = setTimeout(() => elements.toast.classList.add("hidden"), duration);
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = data?.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message || `${response.status} ${response.statusText}`);
  }
  return data;
}

class CanvasPad {
  constructor(canvas, wrap, options = {}) {
    this.canvas = canvas;
    this.wrap = wrap;
    this.context = canvas.getContext("2d");
    this.strokes = [];
    this.redoStack = [];
    this.currentStroke = null;
    this.pointerId = null;
    this.tool = "pen";
    this.color = options.color || "#242938";
    this.width = options.width || 4;
    this.onHistoryChange = options.onHistoryChange || (() => {});
    this.fadeAnimationFrame = null;
    this.fadeStartedAt = null;
    this.fadeDuration = 650;
    this.renderAlpha = 1;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(wrap);
    this.bindEvents();
    this.resize();
  }

  resize() {
    const rect = this.wrap.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    const physicalWidth = Math.round(width * dpr);
    const physicalHeight = Math.round(height * dpr);
    if (this.canvas.width === physicalWidth && this.canvas.height === physicalHeight) return;

    this.canvas.width = physicalWidth;
    this.canvas.height = physicalHeight;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.context.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.logicalWidth = width;
    this.logicalHeight = height;
    this.render();
  }

  bindEvents() {
    this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    this.canvas.addEventListener("pointerup", (event) => this.pointerUp(event));
    this.canvas.addEventListener("pointercancel", (event) => this.pointerUp(event));
    this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  }

  pointFromEvent(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      pressure: event.pressure > 0 ? event.pressure : 0.5,
      t: performance.now(),
    };
  }

  pointerDown(event) {
    if (state.busy || (event.pointerType === "mouse" && event.button !== 0)) return;
    event.preventDefault();
    this.cancelFade();
    this.pointerId = event.pointerId;
    this.canvas.setPointerCapture?.(event.pointerId);
    this.currentStroke = {
      tool: this.tool,
      color: this.color,
      width: this.width,
      pointerType: event.pointerType || "mouse",
      points: [this.pointFromEvent(event)],
    };
    this.drawDot(this.currentStroke, this.currentStroke.points[0]);
  }

  pointerMove(event) {
    if (!this.currentStroke || event.pointerId !== this.pointerId) return;
    event.preventDefault();
    const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
    for (const coalescedEvent of events) {
      const point = this.pointFromEvent(coalescedEvent);
      const previous = this.currentStroke.points.at(-1);
      this.currentStroke.points.push(point);
      this.drawSegment(this.currentStroke, previous, point);
    }
  }

  pointerUp(event) {
    if (!this.currentStroke || event.pointerId !== this.pointerId) return;
    event.preventDefault();
    this.strokes.push(this.currentStroke);
    this.redoStack = [];
    this.currentStroke = null;
    this.pointerId = null;
    this.updateState();
  }

  applyStrokeStyle(stroke, pressure = 0.5) {
    this.context.lineCap = "round";
    this.context.lineJoin = "round";
    this.context.globalCompositeOperation = stroke.tool === "eraser" ? "destination-out" : "source-over";
    this.context.strokeStyle = stroke.color;
    this.context.fillStyle = stroke.color;
    const pressureMultiplier = stroke.tool === "eraser" ? 1 : 0.55 + pressure * 0.9;
    this.context.lineWidth = stroke.width * pressureMultiplier;
  }

  drawDot(stroke, point) {
    this.context.save();
    this.applyStrokeStyle(stroke, point.pressure);
    this.context.beginPath();
    this.context.arc(point.x, point.y, Math.max(0.6, this.context.lineWidth / 2), 0, Math.PI * 2);
    this.context.fill();
    this.context.restore();
  }

  drawSegment(stroke, from, to) {
    if (!from || !to) return;
    this.context.save();
    this.applyStrokeStyle(stroke, (from.pressure + to.pressure) / 2);
    this.context.beginPath();
    this.context.moveTo(from.x, from.y);
    this.context.lineTo(to.x, to.y);
    this.context.stroke();
    this.context.restore();
  }

  render(alpha = this.renderAlpha) {
    this.context.save();
    this.context.setTransform(1, 0, 0, 1, 0, 0);
    this.context.clearRect(0, 0, this.canvas.width || 1, this.canvas.height || 1);
    this.context.restore();
    this.context.globalAlpha = Math.max(0, Math.min(1, alpha));

    for (const stroke of this.strokes) {
      if (stroke.points.length === 1) {
        this.drawDot(stroke, stroke.points[0]);
      } else {
        for (let index = 1; index < stroke.points.length; index += 1) {
          this.drawSegment(stroke, stroke.points[index - 1], stroke.points[index]);
        }
      }
    }

    this.context.globalAlpha = 1;
    this.updateState();
  }

  updateState() {
    this.wrap.classList.toggle("has-ink", this.hasInk());
    this.onHistoryChange({ canUndo: this.strokes.length > 0, canRedo: this.redoStack.length > 0 });
  }

  setTool(tool) {
    this.tool = tool;
    this.canvas.dataset.activeTool = tool;
    this.canvas.classList.toggle("eraser-active", tool === "eraser");
  }

  setColor(color) {
    this.color = color;
  }

  setWidth(width) {
    this.width = Number(width);
  }

  undo() {
    if (!this.strokes.length || state.busy) return;
    this.redoStack.push(this.strokes.pop());
    this.render();
  }

  redo() {
    if (!this.redoStack.length || state.busy) return;
    this.strokes.push(this.redoStack.pop());
    this.render();
  }

  clear() {
    if (state.busy) return;
    this.resetCanvas();
    this.strokes = [];
    this.redoStack = [];
    this.currentStroke = null;
    this.renderAlpha = 1;
    this.updateState();
  }

  resetCanvas() {
    if (this.fadeAnimationFrame) cancelAnimationFrame(this.fadeAnimationFrame);
    this.fadeAnimationFrame = null;
    this.fadeStartedAt = null;
    this.context.save();
    this.context.setTransform(1, 0, 0, 1, 0, 0);
    this.context.clearRect(0, 0, this.canvas.width || 1, this.canvas.height || 1);
    this.context.restore();
  }

  hasInk() {
    return this.strokes.some((stroke) => stroke.tool === "pen" && stroke.points.length > 0);
  }

  snapshot() {
    return structuredClone(this.strokes);
  }

  restore(strokes) {
    if (this.fadeAnimationFrame) cancelAnimationFrame(this.fadeAnimationFrame);
    this.fadeAnimationFrame = null;
    this.fadeStartedAt = null;
    this.strokes = structuredClone(strokes);
    this.redoStack = [];
    this.renderAlpha = 1;
    this.render();
  }

  exportStrokes() {
    return this.strokes.map((stroke) => ({
      tool: stroke.tool,
      color: stroke.color,
      width: stroke.width,
      pointerType: stroke.pointerType,
      points: stroke.points.map(({ x, y, pressure }) => ({ x, y, pressure })),
    }));
  }

  toDataURL() {
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = this.canvas.width;
    exportCanvas.height = this.canvas.height;
    const context = exportCanvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
    context.drawImage(this.canvas, 0, 0);
    return exportCanvas.toDataURL("image/png");
  }

  startFade() {
    if (this.fadeAnimationFrame) cancelAnimationFrame(this.fadeAnimationFrame);
    this.fadeStartedAt = performance.now();
    this.renderAlpha = 1;

    const animate = (now) => {
      const progress = Math.min(1, (now - this.fadeStartedAt) / this.fadeDuration);
      this.renderAlpha = 1 - smoothstep(progress);
      this.render(this.renderAlpha);
      if (progress < 1) {
        this.fadeAnimationFrame = requestAnimationFrame(animate);
      } else {
        this.fadeAnimationFrame = null;
        this.strokes = [];
        this.redoStack = [];
        this.renderAlpha = 1;
        this.resetCanvas();
        this.updateState();
      }
    };

    this.fadeAnimationFrame = requestAnimationFrame(animate);
  }

  cancelFade() {
    if (this.fadeAnimationFrame) cancelAnimationFrame(this.fadeAnimationFrame);
    this.fadeAnimationFrame = null;
    this.fadeStartedAt = null;
    this.renderAlpha = 1;
    this.render();
  }
}

function smoothstep(progress) {
  const value = Math.max(0, Math.min(1, progress));
  return value * value * (3 - 2 * value);
}

function diaryTokens(text) {
  return text.match(/\s+|[\p{L}\p{N}]+(?:[’'][\p{L}\p{N}]+)?|[^\s\p{L}\p{N}]/gu) || [];
}

function diaryTokenPause(token) {
  if (/^\s+$/.test(token)) return token.includes("\n") ? 90 : 18;
  if (/^[.!?]+$/.test(token)) return 230;
  if (/^[,;:]+$/.test(token)) return 130;
  return Math.min(115, 62 + token.length * 4);
}

function animateDiaryText(article, text) {
  const content = article.querySelector(".block-content");
  if (!content) return Promise.resolve();
  content.replaceChildren();
  const fragment = document.createDocumentFragment();
  const tokens = [];
  let delay = 50;

  for (const tokenText of diaryTokens(text)) {
    if (/^\s+$/.test(tokenText)) {
      fragment.append(document.createTextNode(tokenText));
      delay += diaryTokenPause(tokenText);
      continue;
    }
    const span = document.createElement("span");
    span.className = "diary-ink-token";
    span.textContent = tokenText;
    fragment.append(span);
    tokens.push({ element: span, delay });
    delay += diaryTokenPause(tokenText);
  }

  content.append(fragment);
  const startedAt = performance.now();
  const duration = 620;
  const finalDelay = tokens.at(-1)?.delay || 0;

  return new Promise((resolve) => {
    const animate = (now) => {
      const elapsed = now - startedAt;
      let complete = true;
      for (const token of tokens) {
        const local = (elapsed - token.delay) / duration;
        if (local <= 0) {
          complete = false;
          continue;
        }
        const progress = Math.min(1, local);
        token.element.style.opacity = String(smoothstep(progress));
        if (progress < 1) complete = false;
      }
      elements.aiCanvasPane.scrollTop = elements.aiCanvasPane.scrollHeight;
      if (!complete || elapsed < finalDelay + duration) {
        requestAnimationFrame(animate);
      } else {
        resolve();
      }
    };
    requestAnimationFrame(animate);
  });
}

async function animateImageReveal(article) {
  const image = article.querySelector("img");
  if (!image) return;

  article.classList.add("image-reveal");
  if (!image.complete) {
    await new Promise((resolve) => {
      image.addEventListener("load", resolve, { once: true });
      image.addEventListener("error", resolve, { once: true });
    });
  }

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    article.style.opacity = "1";
    return;
  }

  const startedAt = performance.now();
  const delay = 50;
  const duration = 620;
  await new Promise((resolve) => {
    const animate = (now) => {
      const progress = Math.max(0, Math.min(1, (now - startedAt - delay) / duration));
      article.style.opacity = String(smoothstep(progress));
      if (progress < 1) requestAnimationFrame(animate);
      else resolve();
    };
    requestAnimationFrame(animate);
  });
}

function fadeOutCurrentResponse() {
  if (!elements.noteFeed.children.length) return Promise.resolve();
  const startedAt = performance.now();
  const duration = 240;
  return new Promise((resolve) => {
    const animate = (now) => {
      const progress = Math.min(1, (now - startedAt) / duration);
      elements.noteFeed.style.opacity = String(1 - smoothstep(progress));
      if (progress < 1) requestAnimationFrame(animate);
      else resolve();
    };
    requestAnimationFrame(animate);
  });
}

function blockElement(block) {
  const article = document.createElement(block.type === "assistant_image" ? "figure" : "article");
  article.className = "note-block";
  article.dataset.blockId = block.id;

  const meta = document.createElement("div");
  meta.className = "block-meta";

  if (["assistant_text", "error"].includes(block.type)) {
    article.classList.add(block.type === "assistant_text" ? "assistant-block" : "error-block");
    meta.textContent = block.type === "assistant_text" ? "AI note" : "Error";
    const content = document.createElement("div");
    content.className = "text-block block-content";
    content.textContent = block.text || "";
    article.append(meta, content);
    return article;
  }

  if (block.type === "assistant_image") {
    article.classList.add("assistant-image-block");
    meta.textContent = "AI image";
    const frame = document.createElement("div");
    frame.className = "assistant-image-frame";
    const image = document.createElement("img");
    image.src = block.url;
    image.alt = block.prompt || "AI-generated image";
    image.loading = "eager";
    const download = document.createElement("a");
    download.className = "image-download-button";
    download.href = block.url;
    download.download = block.url.split("/").at(-1) || "inknote-image.png";
    download.title = "Download image";
    download.setAttribute("aria-label", "Download generated image");
    download.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 3v11m0 0 4-4m-4 4-4-4M5 17v3h14v-3"/>
      </svg>
    `;
    frame.append(image, download);
    article.append(meta, frame);
    return article;
  }

  return article;
}

async function replaceAIResponse(blocks) {
  await fadeOutCurrentResponse();
  elements.noteFeed.replaceChildren(...blocks.map(blockElement));
  elements.noteFeed.style.opacity = "1";
  elements.aiCanvasPane.scrollTop = 0;

  const animations = [];
  for (const block of blocks) {
    const article = elements.noteFeed.querySelector(`[data-block-id="${CSS.escape(block.id)}"]`);
    if (block.type === "assistant_text") animations.push(animateDiaryText(article, block.text));
    if (block.type === "assistant_image") animations.push(animateImageReveal(article));
  }
  await Promise.all(animations);
}

function latestResponseBlocks() {
  const responses = (state.note?.blocks || []).filter((block) =>
    ["assistant_text", "assistant_image", "error"].includes(block.type) && !block.pending,
  );
  const latest = responses.at(-1);
  if (!latest) return [];
  if (!latest.turn_id) return [latest];
  return responses.filter((block) => block.turn_id === latest.turn_id);
}

function renderAIResponse() {
  const blocks = latestResponseBlocks();
  elements.noteFeed.style.opacity = "1";
  if (!blocks.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state notebook-empty-state";
    empty.innerHTML = "<p>The agent’s response will appear here.</p>";
    elements.noteFeed.replaceChildren(empty);
    return;
  }
  elements.noteFeed.replaceChildren(...blocks.map(blockElement));
}

function conversationBlockElement(block) {
  const turn = document.createElement("article");
  const role = block.type.startsWith("user") ? "user" : block.type === "error" ? "error" : "assistant";
  turn.className = `conversation-turn ${role}`;
  turn.classList.toggle("pending", Boolean(block.pending));

  const label = document.createElement("span");
  label.className = "conversation-role";
  label.textContent = role === "user" ? "You" : role === "assistant" ? "AI" : "Error";
  turn.append(label);

  if (block.type === "assistant_image") {
    const image = document.createElement("img");
    image.className = "conversation-image";
    image.src = block.url;
    image.alt = block.prompt || "Generated image";
    turn.append(image);
  } else {
    const message = document.createElement("p");
    message.className = "conversation-message";
    message.textContent = block.text || (block.pending ? "Working on the canvas…" : "");
    turn.append(message);
  }
  return turn;
}

function renderConversationDock({ scroll = true } = {}) {
  const blocks = (state.note?.blocks || []).filter((block) =>
    ["user_canvas", "user_text", "assistant_text", "assistant_image", "error"].includes(block.type),
  );
  if (!blocks.length) {
    const empty = document.createElement("p");
    empty.className = "conversation-empty";
    empty.textContent = "Canvas requests and AI responses are recorded here.";
    elements.conversationLog.replaceChildren(empty);
    return;
  }

  elements.conversationLog.replaceChildren(...blocks.map(conversationBlockElement));
  if (scroll) requestAnimationFrame(() => {
    elements.conversationLog.scrollTop = elements.conversationLog.scrollHeight;
  });
}

function renderAll({ scroll = false } = {}) {
  renderAIResponse();
  renderConversationDock({ scroll });
}

function addBlock(type, extra = {}) {
  const block = {
    id: uniqueId(type),
    type,
    created_at: new Date().toISOString(),
    ...extra,
  };
  state.note.blocks.push(block);
  return block;
}

function removeBlock(block) {
  state.note.blocks = state.note.blocks.filter((candidate) => candidate.id !== block.id);
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function renderNoteList() {
  elements.noteList.replaceChildren();
  for (const note of state.notes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `note-list-item${state.note?.id === note.id ? " active" : ""}`;
    button.dataset.noteId = note.id;

    const title = document.createElement("span");
    title.className = "note-list-title";
    title.textContent = note.title;
    const time = document.createElement("span");
    time.className = "note-list-time";
    time.textContent = formatTime(note.updated_at);
    button.append(title, time);
    elements.noteList.append(button);
  }
}

function queueSave() {
  if (!state.note) return;
  elements.saveStatus.textContent = "Saving…";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveCurrentNote, 450);
}

async function saveCurrentNote() {
  if (!state.note) return;
  try {
    const saved = await apiJson(`/api/notes/${state.note.id}`, {
      method: "PUT",
      body: JSON.stringify({ title: state.note.title, blocks: state.note.blocks }),
    });
    state.note.updated_at = saved.updated_at;
    const index = state.notes.findIndex((note) => note.id === saved.id);
    if (index >= 0) state.notes[index] = saved;
    state.notes.sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    elements.saveStatus.textContent = "Saved";
    renderNoteList();
  } catch (error) {
    elements.saveStatus.textContent = "Save failed";
    showToast(error.message, 5000);
  }
}

async function loadNotes() {
  state.notes = await apiJson("/api/notes");
  if (!state.notes.length) {
    const note = await apiJson("/api/notes", {
      method: "POST",
      body: JSON.stringify({ title: "My first InkNote" }),
    });
    state.notes = [note];
  }
  await selectNote(state.notes[0].id);
}

async function selectNote(noteId) {
  if (state.note?.id === noteId) return;
  if (state.saveTimer) {
    clearTimeout(state.saveTimer);
    await saveCurrentNote();
  }
  state.note = await apiJson(`/api/notes/${noteId}`);
  handwritingPad.clear();
  elements.noteTitle.value = state.note.title;
  elements.saveStatus.textContent = "Saved";
  renderNoteList();
  renderAll({ scroll: true });
}

async function createNewNote() {
  const note = await apiJson("/api/notes", {
    method: "POST",
    body: JSON.stringify({ title: "Untitled note" }),
  });
  state.notes.unshift(note);
  state.note = null;
  await selectNote(note.id);
  elements.noteTitle.focus();
  elements.noteTitle.select();
}

function setBusy(busy) {
  state.busy = busy;
  if (busy) setCustomColorPopover(false);
  elements.sendCanvasButton.disabled = busy;
  elements.handPenButton.disabled = busy;
  elements.handEraserButton.disabled = busy;
  elements.handClearButton.disabled = busy;
  elements.canvasLayoutButton.disabled = busy;
  elements.inferenceStepsSelect.disabled = busy;
  elements.handColorPalette.classList.toggle("disabled", busy);
  updateHistoryButtons();
}

function setCanvasLineLayout(enabled, { persist = true } = {}) {
  elements.handwritingWrap.classList.toggle("plain-canvas", !enabled);
  elements.canvasLayoutButton.classList.toggle("active", enabled);
  elements.canvasLayoutButton.setAttribute("aria-pressed", String(enabled));
  elements.canvasLayoutButton.setAttribute(
    "aria-label",
    enabled ? "Turn notebook lines off" : "Turn notebook lines on",
  );
  elements.canvasLayoutLabel.textContent = enabled ? "Lines on" : "Lines off";
  if (persist) {
    try {
      localStorage.setItem(CANVAS_LINES_STORAGE_KEY, enabled ? "on" : "off");
    } catch {
      // The visual toggle still works when browser storage is unavailable.
    }
  }
}

function loadCanvasLineLayout() {
  try {
    return localStorage.getItem(CANVAS_LINES_STORAGE_KEY) !== "off";
  } catch {
    return true;
  }
}

function setInferenceSteps(value, { persist = true } = {}) {
  const requested = Number(value);
  const steps = ALLOWED_INFERENCE_STEPS.has(requested) ? requested : 10;
  elements.inferenceStepsSelect.value = String(steps);
  if (persist) {
    try {
      localStorage.setItem(INFERENCE_STEPS_STORAGE_KEY, String(steps));
    } catch {
      // The selector still works when browser storage is unavailable.
    }
  }
}

function loadInferenceSteps() {
  try {
    return Number(localStorage.getItem(INFERENCE_STEPS_STORAGE_KEY) || 10);
  } catch {
    return 10;
  }
}

function qualityLabelForSteps(steps) {
  return { 10: "Low", 25: "Medium", 50: "High" }[Number(steps)] || "Low";
}

function setHistoryButtonState(canUndo, canRedo) {
  elements.handUndoButton.disabled = state.busy || !canUndo;
  elements.handRedoButton.disabled = state.busy || !canRedo;
}

function updateHistoryButtons() {
  setHistoryButtonState(handwritingPad.strokes.length > 0, handwritingPad.redoStack.length > 0);
}

function activateCanvasTool(tool) {
  handwritingPad.setTool(tool);
  handwritingPad.setWidth(tool === "eraser" ? 28 : 4);
  const penActive = tool === "pen";
  elements.handPenButton.classList.toggle("active", penActive);
  elements.handEraserButton.classList.toggle("active", !penActive);
  elements.handPenButton.setAttribute("aria-pressed", String(penActive));
  elements.handEraserButton.setAttribute("aria-pressed", String(!penActive));
}

function hsvToHex(hue, saturation, value) {
  const chroma = value * saturation;
  const segment = ((hue % 360) + 360) % 360 / 60;
  const secondary = chroma * (1 - Math.abs((segment % 2) - 1));
  const match = value - chroma;
  let red = 0;
  let green = 0;
  let blue = 0;

  if (segment < 1) [red, green] = [chroma, secondary];
  else if (segment < 2) [red, green] = [secondary, chroma];
  else if (segment < 3) [green, blue] = [chroma, secondary];
  else if (segment < 4) [green, blue] = [secondary, chroma];
  else if (segment < 5) [red, blue] = [secondary, chroma];
  else [red, blue] = [chroma, secondary];

  return `#${[red, green, blue]
    .map((channel) => Math.round((channel + match) * 255).toString(16).padStart(2, "0"))
    .join("")}`;
}

function setCustomColorPopover(open) {
  elements.customColorPopover.classList.toggle("hidden", !open);
  elements.customColorSwatch.setAttribute("aria-expanded", String(open));
}

function applyCustomColor({ select = true } = {}) {
  const color = hsvToHex(customColor.hue, customColor.saturation, customColor.value);
  elements.customColorSvField.style.setProperty("--picker-hue", String(customColor.hue));
  elements.customColorHue.value = String(customColor.hue);
  elements.customColorCursor.style.left = `${customColor.saturation * 100}%`;
  elements.customColorCursor.style.top = `${(1 - customColor.value) * 100}%`;
  elements.customColorControl.style.setProperty("--custom-color", color);
  elements.customColorValue.value = color.toUpperCase();
  elements.customColorValue.textContent = color.toUpperCase();
  elements.customColorSvField.setAttribute(
    "aria-valuetext",
    `Saturation ${Math.round(customColor.saturation * 100)}%, value ${Math.round(customColor.value * 100)}%`,
  );
  if (select) selectColor(color, elements.customColorSwatch);
}

function setSaturationValueFromPointer(event) {
  const bounds = elements.customColorSvField.getBoundingClientRect();
  customColor.saturation = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
  customColor.value = Math.max(0, Math.min(1, 1 - (event.clientY - bounds.top) / bounds.height));
  applyCustomColor();
}

function selectColor(color, source = null) {
  handwritingPad.setColor(color);
  activateCanvasTool("pen");
  $$(".color-swatch").forEach((swatch) => swatch.classList.toggle("active", swatch === source));
  elements.customColorSwatch.classList.toggle("active", source === elements.customColorSwatch);
}

async function submitCanvas() {
  if (state.busy || !state.note) return;
  if (!handwritingPad.hasInk()) {
    showToast("Write or draw something before pressing Send.");
    return;
  }

  const turnId = uniqueId("turn");
  const snapshot = handwritingPad.snapshot();
  const pageDataUrl = handwritingPad.toDataURL();
  const strokes = handwritingPad.exportStrokes();
  const inferenceSteps = Number(elements.inferenceStepsSelect.value);
  const userBlock = addBlock("user_canvas", {
    turn_id: turnId,
    text: "Understanding canvas…",
    pending: true,
  });
  const pendingBlock = addBlock("assistant_text", {
    turn_id: turnId,
    text: "",
    pending: true,
  });

  setBusy(true);
  renderConversationDock({ scroll: true });
  queueSave();
  handwritingPad.startFade();
  elements.processingText.textContent = "Understanding and routing…";
  elements.processingOverlay.classList.remove("hidden");
  elements.sendCanvasButton.textContent = "Working…";
  const slowTimer = setTimeout(() => {
    elements.processingText.textContent = `Generating at ${qualityLabelForSteps(inferenceSteps)} quality…`;
  }, 9000);

  try {
    const result = await apiJson("/api/canvas/submit", {
      method: "POST",
      body: JSON.stringify({
        note_id: state.note.id,
        page_data_url: pageDataUrl,
        strokes,
        inference_steps: inferenceSteps,
      }),
    });

    userBlock.text = result.user_text || "Canvas submission";
    userBlock.intent = result.intent;
    userBlock.pending = false;

    const responseBlocks = [];
    if (result.text_response) {
      pendingBlock.text = result.text_response;
      pendingBlock.pending = false;
      responseBlocks.push(pendingBlock);
    } else {
      removeBlock(pendingBlock);
    }

    if (result.image_url) {
      responseBlocks.push(addBlock("assistant_image", {
        turn_id: turnId,
        url: result.image_url,
        prompt: result.image_prompt || "Generated from the canvas",
      }));
    }

    if (!responseBlocks.length) {
      pendingBlock.text = "I could not determine a response for this canvas.";
      pendingBlock.pending = false;
      if (!state.note.blocks.includes(pendingBlock)) state.note.blocks.push(pendingBlock);
      responseBlocks.push(pendingBlock);
    }

    renderConversationDock({ scroll: true });
    await replaceAIResponse(responseBlocks);
    const qualityNote = result.inference_steps
      ? ` Image quality: ${qualityLabelForSteps(result.inference_steps)}.`
      : "";
    elements.canvasHint.textContent = `Agent route: ${result.intent}.${qualityNote} Write or draw the next request.`;
  } catch (error) {
    userBlock.text = "Canvas submission";
    userBlock.pending = false;
    pendingBlock.type = "error";
    pendingBlock.text = error.message;
    pendingBlock.pending = false;
    handwritingPad.restore(snapshot);
    renderConversationDock({ scroll: true });
    await replaceAIResponse([pendingBlock]);
    showToast(error.message, 7000);
  } finally {
    clearTimeout(slowTimer);
    elements.processingOverlay.classList.add("hidden");
    elements.processingText.textContent = "Understanding canvas…";
    elements.sendCanvasButton.textContent = "Send";
    setBusy(false);
    queueSave();
  }
}

async function checkHealth() {
  try {
    const health = await apiJson("/api/health");
    elements.connectionDot.classList.add("online");
    elements.connectionDot.classList.remove("offline");
    elements.connectionText.textContent = "Desktop connected";
    const image = health.image_generation_enabled ? ` · Image: ${health.image_model}` : " · Image: disabled";
    elements.modelSummary.textContent = `Router: ${health.multimodal_model} · Chat: ${health.chat_model}${image}`;
  } catch (error) {
    elements.connectionDot.classList.add("offline");
    elements.connectionDot.classList.remove("online");
    elements.connectionText.textContent = "Desktop unavailable";
    elements.modelSummary.textContent = error.message;
  }
}

function bindUI() {
  elements.newNoteButton.addEventListener("click", () => {
    if (state.busy) return showToast("Wait for the current request to finish.");
    createNewNote().catch((error) => showToast(error.message));
  });

  elements.noteList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-note-id]");
    if (!button) return;
    if (state.busy) return showToast("Wait for the current request to finish.");
    selectNote(button.dataset.noteId).catch((error) => showToast(error.message));
  });

  elements.noteTitle.addEventListener("input", () => {
    if (!state.note) return;
    state.note.title = elements.noteTitle.value || "Untitled note";
    queueSave();
  });

  elements.deleteNoteButton.addEventListener("click", async () => {
    if (!state.note || state.busy || !confirm(`Delete “${state.note.title}”?`)) return;
    await apiJson(`/api/notes/${state.note.id}`, { method: "DELETE" });
    state.notes = state.notes.filter((note) => note.id !== state.note.id);
    state.note = null;
    if (!state.notes.length) await createNewNote();
    else await selectNote(state.notes[0].id);
  });

  elements.clearConversationButton.addEventListener("click", () => {
    if (state.busy || !state.note?.blocks.length || !confirm("Clear all responses and history from this note?")) return;
    state.note.blocks = [];
    renderAll();
    queueSave();
  });

  elements.handPenButton.addEventListener("click", () => activateCanvasTool("pen"));
  elements.handEraserButton.addEventListener("click", () => activateCanvasTool("eraser"));
  elements.handUndoButton.addEventListener("click", () => handwritingPad.undo());
  elements.handRedoButton.addEventListener("click", () => handwritingPad.redo());
  elements.handClearButton.addEventListener("click", () => handwritingPad.clear());
  elements.canvasLayoutButton.addEventListener("click", () => {
    const enabled = elements.canvasLayoutButton.getAttribute("aria-pressed") !== "true";
    setCanvasLineLayout(enabled);
  });
  elements.inferenceStepsSelect.addEventListener("change", (event) => {
    setInferenceSteps(event.target.value);
  });
  elements.sendCanvasButton.addEventListener("click", submitCanvas);

  elements.handColorPalette.addEventListener("click", (event) => {
    if (state.busy) return;
    const swatch = event.target.closest("[data-color]");
    if (swatch) {
      selectColor(swatch.dataset.color, swatch);
      setCustomColorPopover(false);
    }
  });

  elements.customColorSwatch.addEventListener("click", (event) => {
    event.stopPropagation();
    if (state.busy) return;
    const isOpen = elements.customColorSwatch.getAttribute("aria-expanded") === "true";
    setCustomColorPopover(!isOpen);
  });

  elements.customColorPopover.addEventListener("click", (event) => event.stopPropagation());

  elements.customColorSvField.addEventListener("pointerdown", (event) => {
    if (state.busy) return;
    customColor.dragging = true;
    elements.customColorSvField.setPointerCapture(event.pointerId);
    setSaturationValueFromPointer(event);
  });

  elements.customColorSvField.addEventListener("pointermove", (event) => {
    if (customColor.dragging) setSaturationValueFromPointer(event);
  });

  const stopCustomColorDrag = () => {
    customColor.dragging = false;
  };
  elements.customColorSvField.addEventListener("pointerup", stopCustomColorDrag);
  elements.customColorSvField.addEventListener("pointercancel", stopCustomColorDrag);

  elements.customColorSvField.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 0.1 : 0.02;
    if (event.key === "ArrowLeft") customColor.saturation -= step;
    else if (event.key === "ArrowRight") customColor.saturation += step;
    else if (event.key === "ArrowUp") customColor.value += step;
    else if (event.key === "ArrowDown") customColor.value -= step;
    else return;
    event.preventDefault();
    customColor.saturation = Math.max(0, Math.min(1, customColor.saturation));
    customColor.value = Math.max(0, Math.min(1, customColor.value));
    applyCustomColor();
  });

  elements.customColorHue.addEventListener("input", (event) => {
    customColor.hue = Number(event.target.value);
    applyCustomColor();
  });

  document.addEventListener("pointerdown", (event) => {
    if (!elements.customColorControl.contains(event.target)) setCustomColorPopover(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setCustomColorPopover(false);
  });

  document.addEventListener("keydown", (event) => {
    if (state.busy || event.target.matches("input, textarea")) return;
    if (!(event.ctrlKey || event.metaKey)) return;
    if (event.key.toLowerCase() === "z" && event.shiftKey) {
      event.preventDefault();
      handwritingPad.redo();
    } else if (event.key.toLowerCase() === "z") {
      event.preventDefault();
      handwritingPad.undo();
    } else if (event.key.toLowerCase() === "y") {
      event.preventDefault();
      handwritingPad.redo();
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden" && state.note) {
      fetch(`/api/notes/${state.note.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: state.note.title, blocks: state.note.blocks }),
        keepalive: true,
      }).catch(() => {});
    }
  });

  applyCustomColor({ select: false });
  setCanvasLineLayout(loadCanvasLineLayout(), { persist: false });
  setInferenceSteps(loadInferenceSteps(), { persist: false });
}

const handwritingPad = new CanvasPad(elements.handwritingCanvas, elements.handwritingWrap, {
  color: "#242938",
  width: 4,
  onHistoryChange: ({ canUndo, canRedo }) => setHistoryButtonState(canUndo, canRedo),
});

async function initialize() {
  bindUI();
  activateCanvasTool("pen");
  updateHistoryButtons();
  await Promise.allSettled([checkHealth(), loadNotes()]);
  if ("serviceWorker" in navigator && (window.isSecureContext || location.hostname === "localhost")) {
    navigator.serviceWorker.register("/service-worker.js").catch(() => {});
  }
}

initialize();
