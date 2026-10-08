"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {session: null, policy: "ppo", generator: "hybrid", pipeline: "voice", busy: false, dirty: false,
  pending: null, failed: null, selectedTurn: null, view: "conversation", status: null, recording: false, voiceDraft: null};
const scenarios = {
  family: {age: 35, budget: "low", profile: "FAMILY_ORIENTED", message: "I need health insurance for my family."},
  coverage: {age: 42, budget: "mid", profile: "LOYAL_TO_EXISTING_INSURER", message: "I already have a 5 lakh health policy. Do I need additional cover?"},
  price: {age: 29, budget: "low", profile: "PRICE_SENSITIVE", message: "I need family health insurance, but the premium is too expensive."},
  rejection: {age: 35, budget: "mid", profile: "HESITANT", message: "No, I am not interested. Please stop."},
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function icon(name) {
  const node = element("i");
  node.dataset.lucide = name;
  return node;
}
function icons() {
  if (window.lucide) window.lucide.createIcons({attrs: {"aria-hidden": "true"}});
}
function human(value) {
  if (!value || value === "unknown") return "Unknown";
  return String(value).toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
function percent(value, digits = 0) { return `${(Math.max(0, Math.min(1, Number(value) || 0)) * 100).toFixed(digits)}%`; }
function clock(value) { return new Date(value).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

async function api(path, data) {
  const response = await fetch(path, {method: data ? "POST" : "GET",
    headers: data ? {"Content-Type": "application/json"} : {}, body: data ? JSON.stringify(data) : undefined,
    signal: AbortSignal.timeout(path.endsWith("/messages") ? 90000 : 15000)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
  return result;
}
function errorMessage(error) {
  return error.name === "TimeoutError" ? "The request timed out. Retry or use template mode." :
    error instanceof TypeError ? "The demo server is unreachable. Check that it is still running." : error.message;
}
let toastTimer;
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 3500);
}
function showError(message, retry = false) {
  $("#error-banner").hidden = false;
  $("#error-text").textContent = message;
  $("#retry-message").hidden = !retry;
}
function hideError() { $("#error-banner").hidden = true; }

function settings() {
  return {age: Number($("#age").value), budget: $("#budget").value, profile: $("#profile").value,
    policy: state.policy, generator: state.generator, pipeline: state.pipeline};
}
function setSettings(values) {
  $("#age").value = values.age;
  $("#budget").value = values.budget;
  $("#profile").value = values.profile;
  state.policy = values.policy;
  state.generator = values.generator;
  state.pipeline = values.pipeline || "text";
  updateSettings();
}
function updateSettings() {
  for (const type of ["policy", "generator", "pipeline"]) {
    $$(`[data-${type}]`).forEach((button) => {
      const active = button.dataset[type] === state[type];
      button.classList.toggle("selected", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }
  const current = settings();
  state.dirty = Boolean(state.session && Object.keys(current).some((key) => current[key] !== state.session.settings[key]));
  $("#buyer-title").textContent = `${human(current.profile).replace("Family Oriented", "Family-oriented")} buyer`;
  $("#buyer-subtitle").textContent = `Age ${current.age || "--"} / ${current.budget === "mid" ? "Medium" : human(current.budget)} budget`;
  if ($("#empty-profile")) $("#empty-profile").textContent = `${human(current.profile)} buyer / Age ${current.age || "--"}`;
  const active = state.session?.settings || current;
  $("#engine-label").textContent = `${active.policy === "ppo" ? "PPO" : "Rules"} + ${{ollama: "Local LLM", hybrid: "Hybrid", template: "Templates"}[active.generator]}`;
  $("#pipeline-label").textContent = active.pipeline === "voice" ? "VOICE-TRAINED AGENT" : "TEXT-TRAINED AGENT";
  $("#checkpoint-name").textContent = `${active.pipeline === "voice" ? "Voice-trained" : "Text-trained"} ${active.policy === "ppo" ? "PPO" : "rule baseline"}`;
  $("#checkpoint-path").textContent = active.policy === "rule" ? "Rule policy / shared conversation guards" :
    active.pipeline === "voice" ? state.status?.voice.checkpoint || "Checking artifacts" : "results/checkpoints/ppo.npz";
  $("#experiment-label").textContent = active.pipeline.toUpperCase();
  controls();
}
function controls() {
  const closed = Boolean(state.session?.closed);
  const locked = state.busy || Boolean(state.recording);
  $$("#profile-form input, #profile-form select, #profile-form fieldset button, .scenario, #new-session").forEach((node) => {node.disabled = locked;});
  $("#apply-profile").disabled = locked || (!state.dirty && Boolean(state.session));
  $("#message").disabled = locked || closed || !state.session || state.dirty;
  $("#send").disabled = $("#message").disabled || !$("#message").value.trim();
  $$("[data-message]").forEach((node) => {node.disabled = locked || closed || !state.session || state.dirty;});
  const voice = state.status?.voice;
  const audioReady = voice?.asr_ready && state.session?.settings.pipeline === "voice";
  $("#upload-audio").disabled = $("#message").disabled || !audioReady || Boolean(state.voiceDraft);
  $("#record").disabled = state.recording === "requesting" || state.recording === "stopping" ||
    (!state.recording && ($("#message").disabled || Boolean(state.voiceDraft) || !audioReady || !navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode));
  $("#review-transcript").disabled = locked || Boolean(state.voiceDraft);
  $("#discard-transcript").disabled = locked;
  $("#transcript-review").hidden = !state.voiceDraft;
  $("#send").setAttribute("aria-label", state.voiceDraft ? "Confirm transcript and send" : "Send buyer message");
  $("#send").dataset.tooltip = state.voiceDraft ? "Confirm transcript and send" : "Send message";
  $("#auto-speak").disabled = !voice?.tts_available;
  $$("[data-speak]").forEach((node) => {node.disabled = locked || !voice?.tts_available;});
  $("#speech-status").textContent = !voice ? "Checking speech" : !voice.asr_ready ? "Speech model unavailable" :
    state.session?.settings.pipeline === "voice" ? "Local Vosk / English" : "Text-trained pipeline";
  $("#export").disabled = !state.session?.history.length;
  $("#conversation-closed").hidden = !closed;
  $("#response-status").classList.toggle("busy", state.busy);
  const status = state.recording ? "Microphone active" : state.busy ? "Processing input" : closed ? "Session closed" : state.dirty ? "Profile changes pending" : state.session ? "Ready" : "Connecting";
  $("#response-status").replaceChildren(element("span", "status-dot"), document.createTextNode(status));
  if (state.status) {
    if (!(state.pipeline === "voice" ? voice.ppo_ready : state.status.ppo_ready)) $("[data-policy='ppo']").disabled = true;
    if (!voice.nlp_ready) $("[data-pipeline='voice']").disabled = true;
  }
}

async function confirmReset() {
  if (!state.session?.history.length) return true;
  const dialog = $("#reset-dialog");
  dialog.showModal();
  return new Promise((resolve) => {
    const finish = (result) => {dialog.close(); resolve(result);};
    $("#confirm-reset").onclick = () => finish(true);
    $("#cancel-reset").onclick = () => finish(false);
    dialog.oncancel = (event) => {event.preventDefault(); finish(false);};
  });
}
async function newSession() {
  if (!$("#profile-form").reportValidity()) return false;
  state.busy = true;
  stopPlayback();
  controls();
  hideError();
  try {
    state.session = await api("/api/sessions", settings());
    state.selectedTurn = null;
    state.pending = null;
    state.failed = null;
    state.voiceDraft = null;
    state.dirty = false;
    $("#voice-feedback").hidden = true;
    sessionStorage.setItem("insurance-demo-session", state.session.id);
    $("#message").value = "";
    $("#session-label").textContent = `Session ${state.session.id.slice(0, 6).toUpperCase()}`;
    $("#character-count").textContent = "0 / 2000";
    renderMessages();
    renderInspector(null);
    updateSettings();
    if (state.view === "catalogue") await loadProducts();
    return true;
  } catch (error) {
    showError(errorMessage(error));
    return false;
  } finally {
    state.busy = false;
    controls();
  }
}

function sourceName(response) {
  if (response.source === "ollama") return "Local LLM";
  if (response.action === "RESPECT_REJECTION") return "Safety template";
  if (response.generation_route === "llm_fallback" || response.llm_error || response.llm_rejected) return "Template fallback";
  if (response.generation_route === "deterministic_task") return "Deterministic reply";
  return "Template";
}
function renderMessage(role, text, entry) {
  const message = element("article", `chat-message ${role}`);
  const avatar = element("span", "message-avatar");
  avatar.append(icon(role === "buyer" ? "user-round" : "shield-check"));
  const body = element("div", "message-body");
  const header = element("div", "message-header");
  header.append(element("strong", "", role === "buyer" ? "Buyer" : "Insurance agent"), element("span", "", clock(entry.timestamp)));
  if (role === "agent") {
    const source = sourceName(entry.response);
    header.append(element("span", `message-source ${source.includes("fallback") ? "fallback" : ""}`, source));
  }
  if (role === "buyer" && entry.voice) header.append(element("span", "message-source audio-source", entry.voice.corrected ? "Corrected voice transcript" : entry.voice.reviewed ? "Reviewed voice transcript" : "Voice transcript"));
  body.append(header, element("div", "message-text", text));
  if (role === "agent") {
    const footer = element("div", "message-footer");
    const check = element("span", "");
    check.title = "Basic numeric-claim, repetition and buyer-role checks; not full semantic validation.";
    check.append(icon(entry.response.validation.ok ? "shield-check" : "triangle-alert"),
      document.createTextNode(entry.response.validation.ok ? "Basic checks passed" : "Check warning"));
    footer.append(check, element("span", "", `${(entry.duration_ms / 1000).toFixed(1)}s`));
    const speak = element("button", "icon-button replay-button");
    speak.dataset.speak = entry.turn;
    speak.setAttribute("aria-label", `Play response ${entry.turn}`);
    speak.dataset.tooltip = "Play checked response";
    speak.append(icon("volume-2"));
    speak.onclick = () => playResponse(entry);
    footer.append(speak);
    const inspect = element("button", "message-inspect", `Turn ${String(entry.turn).padStart(2, "0")}`);
    inspect.append(icon("arrow-up-right"));
    inspect.setAttribute("aria-label", `Inspect turn ${entry.turn}`);
    inspect.onclick = () => {state.selectedTurn = entry.turn; renderInspector(entry); setPanel("inspector");};
    footer.append(inspect);
    body.append(footer);
  }
  message.append(avatar, body);
  return message;
}
function renderMessages() {
  const log = $("#chat-log");
  const empty = $("#empty-chat");
  if (empty) log.removeChild(empty);
  log.replaceChildren();
  if (!state.session?.history.length && !state.pending) {
    if (empty) log.append(empty);
    else {
      const start = element("div", "empty-chat");
      start.id = "empty-chat";
      const symbol = element("span", "empty-symbol"); symbol.append(icon("messages-square"));
      const profile = element("p", "", `${human(settings().profile)} buyer / Age ${settings().age}`); profile.id = "empty-profile";
      const starters = element("div", "starter-messages");
      ["I need health insurance for my family.", "What does the policy cover?"].forEach((text) => {
        const button = element("button", "", text); button.dataset.message = text; button.append(icon("arrow-up-right")); starters.append(button);
      });
      start.append(symbol, element("h2", "", "New conversation"), profile, starters);
      log.append(start);
    }
  }
  for (const entry of state.session?.history || []) {
    log.append(renderMessage("buyer", entry.customer, entry), renderMessage("agent", entry.response.text, entry));
  }
  if (state.pending) {
    log.append(renderMessage("buyer", state.pending.message || "Voice recording", {timestamp: state.pending.timestamp}));
    const pending = element("article", "chat-message pending");
    const avatar = element("span", "message-avatar"); avatar.append(icon("shield-check"));
    const body = element("div", "message-body");
    const text = element("div", "message-text");
    const label = element("span", "", state.pending.audio ? state.pending.review ? "Transcribing for review..." : "Transcribing and responding..." : "Generating response..."); label.id = "pending-label";
    text.append(icon("loader-circle"), label);
    body.append(text); pending.append(avatar, body); log.append(pending);
  }
  $("#turn-count").textContent = `${state.session?.history.length || 0} ${state.session?.history.length === 1 ? "turn" : "turns"}`;
  icons();
  log.scrollTop = log.scrollHeight;
}

function traceRow(label, value, confidence) {
  const row = element("div", "trace-row");
  const detail = element("div");
  detail.append(element("strong", "", human(value)));
  if (typeof confidence === "number" && Number.isFinite(confidence)) {
    const meter = element("div", "confidence");
    const track = element("span", "confidence-track");
    const fill = element("span", "confidence-fill"); fill.style.width = percent(confidence);
    track.append(fill); meter.append(track, element("small", "", percent(confidence))); detail.append(meter);
  }
  row.append(element("span", "", label), detail);
  return row;
}
function statePair(label, value) {
  const pair = element("div"); pair.append(element("dt", "", label), element("dd", "", human(value))); return pair;
}
function validationLine(text, valid) {
  const line = element("div", `validation-line ${valid ? "" : "warning"}`);
  line.append(icon(valid ? "circle-check" : "triangle-alert"), document.createTextNode(text)); return line;
}
function renderInspector(entry) {
  $("#trace-turn").textContent = entry ? `TURN ${String(entry.turn).padStart(2, "0")}` : "NO TURNS";
  $("#action-name").textContent = entry ? human(entry.response.action) : "Awaiting buyer";
  $("#strategy-name").textContent = entry ? human(entry.response.strategy) : "No strategy selected";
  $("#understanding").replaceChildren();
  $("#buyer-state").replaceChildren();
  $("#validation").replaceChildren();
  $("#fact-list").replaceChildren();
  $("#runtime-trace").replaceChildren();
  $("#workflow-fields").replaceChildren();
  $("#workflow-section").hidden = !entry?.response.state.workflow?.active;
  if (!entry) {
    ["Intent", "Emotion", "Objection"].forEach((name) => $("#understanding").append(traceRow(name, "Not observed")));
    $("#buyer-state").append(statePair("Need", "unknown"), statePair("Budget", state.session?.settings.budget || settings().budget),
      statePair("Existing cover", "unknown"), statePair("Stage", "GREETING"));
    $("#validation").append(element("div", "validation-empty", "No response yet"));
    updateStage(null);
  } else {
    const response = entry.response, nlp = response.understanding, buyer = response.state;
    if (buyer.workflow?.active) {
      const names = {need: "Insurance need", age: "Buyer age", coverage: "Coverage target", existing: "Existing policy", budget: "Budget band"};
      for (const [field, label] of Object.entries(names)) {
        const done = buyer.workflow.completed_fields.includes(field);
        const row = element("li", done ? "complete" : buyer.workflow.pending_field === field ? "current" : "");
        row.append(icon(done ? "circle-check" : buyer.workflow.pending_field === field ? "circle-dot" : "circle"), document.createTextNode(label));
        $("#workflow-fields").append(row);
      }
      $("#workflow-phase").textContent = human(buyer.workflow.phase);
    }
    $("#runtime-trace").append(statePair("Input", entry.input_mode || "text"),
      statePair("Route", response.generation_route || response.source),
      statePair("LLM attempted", response.llm_attempted ? "Yes" : "No"),
      statePair("LLM time", `${Number(response.llm_seconds || 0).toFixed(2)} s`),
      statePair("Decision source", response.decision_source),
      statePair("Processing", `${(entry.duration_ms / 1000).toFixed(2)} s`));
    if (entry.voice) $("#runtime-trace").append(
      statePair("ASR confidence", percent(entry.voice.transcript.confidence, 1)),
      statePair("ASR time", `${entry.voice.transcript.asr_seconds.toFixed(2)} s`),
      statePair("Audio duration", `${entry.voice.transcript.audio_seconds.toFixed(1)} s`));
    if (entry.voice?.reviewed) $("#runtime-trace").append(statePair("Transcript", entry.voice.corrected ? "User corrected" : "User confirmed"));
    if (entry.voice?.corrected) $("#runtime-trace").append(statePair("ASR original", entry.voice.transcript.normalized_text));
    $("#understanding").append(traceRow("Intent", nlp.intent, nlp.intent_conf), traceRow("Emotion", nlp.emotion, nlp.emotion_conf),
      traceRow("Objection", nlp.objection, nlp.objection_conf));
    $("#buyer-state").append(statePair("Need", buyer.need), statePair("Budget", buyer.budget),
      statePair("Existing cover", buyer.existing_coverage), statePair("Stage", buyer.sales_stage));
    if (buyer.amount_context?.value != null) $("#buyer-state").append(statePair("Buyer amount", `${buyer.amount_context.value} (${human(buyer.amount_context.kind)})`));
    $("#validation").append(validationLine(response.validation.ok ? "Basic response checks passed" : "Response check warning", response.validation.ok));
    const fallback = sourceName(response).includes("fallback");
    $("#validation").append(validationLine(`Source: ${sourceName(response)}`, !fallback));
    if (response.llm_error) $("#validation").append(validationLine(`LLM error: ${response.llm_error_detail || response.llm_error}`, false));
    if (response.llm_rejected) $("#validation").append(validationLine(`LLM rejected: ${(response.llm_rejection_reasons || ["strategy or fact guard"]).map(human).join(", ")}`, false));
    if (response.llm_rejection_details?.unsupported?.length) $("#validation").append(validationLine(`Unverified numeric tokens: ${response.llm_rejection_details.unsupported.join(", ")}`, false));
    if (response.decision_reason) $("#validation").append(validationLine(`Dialogue constraint: ${human(response.decision_reason)}`, true));
    if (response.decision_source === "conversation_guard") $("#validation").append(validationLine(`Policy proposal: ${human(response.policy_action)}`, true));
    if (response.fallback_rephrased) $("#validation").append(validationLine("Repeated fallback replaced", true));
    if (response.validation.repeated) $("#validation").append(validationLine(response.validation.repeat_allowed ? "Question or summary repeated on request" : "Repeated response", Boolean(response.validation.repeat_allowed)));
    if (response.closed_reason) $("#validation").append(validationLine(`Closed: ${human(response.closed_reason)}`, true));
    for (const product of response.facts) {
      const reference = element("div", "fact-reference");
      reference.append(element("strong", "", `${product.product_id} / ${product.sum_insured}`), element("div", "", product.coverage));
      $("#fact-list").append(reference);
    }
    if (!response.facts.length) $("#fact-list").append(element("div", "validation-empty", "No product facts used"));
    updateStage(response);
  }
  icons();
}
function updateStage(response) {
  const stage = response?.state.sales_stage || "GREETING";
  const active = response?.closed ? "closed" : stage === "GREETING" ? "greeting" :
    stage === "COMMITMENT" ? "commitment" : ["DISCOVERY", "NEED_IDENTIFICATION"].includes(stage) ? "discovery" : "evaluation";
  $$("[data-stage]").forEach((node) => {node.classList.toggle("current", node.dataset.stage === active);});
}

async function sendMessage(retry = null) {
  if (state.busy || state.recording || !state.session || state.session.closed || state.dirty) return;
  stopPlayback();
  const text = retry?.message || $("#message").value.trim();
  if (!text || text.length > 2000) return;
  hideError();
  state.pending = retry || {message: text, request_id: crypto.randomUUID(), timestamp: new Date().toISOString(), transcript_request_id: state.voiceDraft?.request_id};
  state.busy = true; controls(); renderMessages();
  const started = Date.now();
  const timer = setInterval(() => {
    const label = $("#pending-label");
    if (label) label.textContent = `Generating response... ${Math.floor((Date.now() - started) / 1000)}s`;
  }, 1000);
  try {
    const entry = await api(`/api/sessions/${state.session.id}/messages`, {message: text, request_id: state.pending.request_id, transcript_request_id: state.pending.transcript_request_id});
    state.session.history.push(entry);
    state.session.closed = entry.response.closed;
    state.selectedTurn = entry.turn;
    state.failed = null;
    state.voiceDraft = null;
    state.pending = null;
    $("#message").value = "";
    $("#character-count").textContent = "0 / 2000";
    renderMessages(); renderInspector(entry);
    if ($("#auto-speak").checked) void playResponse(entry);
  } catch (error) {
    state.failed = state.pending;
    state.pending = null;
    $("#message").value = text;
    $("#character-count").textContent = `${text.length} / 2000`;
    renderMessages();
    showError(errorMessage(error), true);
  } finally {
    clearInterval(timer); state.busy = false; controls();
    if (!state.session.closed) $("#message").focus();
  }
}

let recorder = null, recordTimer = null, playback = null, playbackURL = null, playbackRequest = null;
let playbackVersion = 0;
function stopPlayback() {
  playbackVersion += 1;
  playbackRequest?.abort(); playbackRequest = null;
  if (playback) { playback.pause(); playback.removeAttribute("src"); playback.load(); playback = null; }
  if (playbackURL) URL.revokeObjectURL(playbackURL);
  playbackURL = null;
  $("#stop-playback").hidden = true;
}
async function playResponse(entry) {
  if (state.recording || !state.status?.voice.tts_available) return;
  stopPlayback();
  const version = playbackVersion, sessionId = state.session.id;
  playbackRequest = new AbortController();
  $("#stop-playback").hidden = false;
  const timeout = setTimeout(() => playbackRequest?.abort(), 65000);
  try {
    const response = await fetch(`/api/sessions/${sessionId}/speech`, {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({turn: entry.turn}), signal: playbackRequest.signal});
    if (!response.ok) throw new Error((await response.json()).error);
    const blob = await response.blob();
    if (version !== playbackVersion || state.session.id !== sessionId || state.recording) return;
    playbackURL = URL.createObjectURL(blob);
    playback = new Audio(playbackURL);
    playback.onended = stopPlayback;
    playback.onerror = () => {stopPlayback(); toast("Audio playback failed. The text response is still available.");};
    await playback.play();
  } catch (error) {
    if (version === playbackVersion) {
      stopPlayback();
      if (error.name !== "AbortError") showError(error.name === "NotAllowedError" ? "Playback was blocked by the browser. Use the response speaker button." : errorMessage(error));
    }
  } finally { clearTimeout(timeout); }
}
function recordingUI() {
  $("#recording-strip").hidden = !state.recording;
  const active = Boolean(state.recording);
  $("#record").classList.toggle("recording", active);
  $("#record").setAttribute("aria-label", active ? "Stop recording and send" : "Record buyer voice");
  $("#record").dataset.tooltip = active ? "Stop recording and send" : "Record buyer voice";
  $("#record").replaceChildren(icon(active ? "square" : "mic"));
  icons(); controls();
}
async function startRecording() {
  if (state.busy || state.recording || $("#record").disabled) return;
  stopPlayback(); hideError(); $("#voice-feedback").hidden = true;
  state.recording = "requesting";
  $("#recording-status").textContent = "Awaiting microphone permission";
  recorder = new LocalRecorder((level) => {$("#mic-level").value = level;});
  const current = recorder;
  recordingUI();
  try {
    if (!await current.start() || current !== recorder) return;
    state.recording = "active";
    const started = Date.now();
    $("#recording-status").textContent = "Recording 0 / 10 s";
    recordTimer = setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      $("#recording-status").textContent = `Recording ${seconds} / 10 s`;
      if (seconds >= 10) void finishRecording();
    }, 200);
    recordingUI();
  } catch (error) {
    if (current !== recorder) return;
    recorder = null; state.recording = false; recordingUI();
    showError(error.name === "NotAllowedError" ? "Microphone permission denied. Allow access in the browser or upload a WAV file." :
      error.name === "NotFoundError" ? "No microphone found. Connect a microphone or upload a WAV file." : errorMessage(error));
  }
}
async function finishRecording(discard = false) {
  if (!recorder || state.recording === "stopping") return;
  const current = recorder;
  clearInterval(recordTimer);
  state.recording = "stopping"; recordingUI();
  try {
    const blob = await current.stop(discard);
    recorder = null; state.recording = false; recordingUI();
    if (blob) await sendAudio(blob);
  } catch (error) {
    recorder = null; state.recording = false; recordingUI(); showError(errorMessage(error));
  }
}
async function sendAudio(blob, retry = null) {
  if (state.busy || state.recording || !state.session || state.session.closed || state.dirty) return;
  if (!blob.size || blob.size > 964096) {showError("Upload a 16 kHz mono PCM WAV no longer than 30 seconds."); return;}
  stopPlayback(); hideError(); $("#voice-feedback").hidden = true;
  state.pending = retry || {audio: blob, request_id: crypto.randomUUID(), timestamp: new Date().toISOString(), review: $("#review-transcript").checked};
  state.busy = true; controls(); renderMessages();
  try {
    const response = await fetch(`/api/sessions/${state.session.id}/audio${state.pending.review ? "?review=1" : ""}`, {method: "POST", body: blob,
      headers: {"Content-Type": "audio/wav", "X-Request-ID": state.pending.request_id}, signal: AbortSignal.timeout(90000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    state.pending = null; state.failed = null;
    if (result.review && result.transcript) {
      state.voiceDraft = result;
      $("#message").value = result.transcript.normalized_text;
      $("#character-count").textContent = `${$("#message").value.length} / 2000`;
      $("#transcript-review-label").textContent = `${result.accepted ? "Transcript awaiting confirmation" : "Low-confidence transcript"} / ASR ${percent(result.transcript.confidence)}`;
    } else if (result.entry) {
      state.session.history.push(result.entry); state.session.closed = result.entry.response.closed;
      state.selectedTurn = result.entry.turn;
      renderInspector(result.entry);
      if ($("#auto-speak").checked) void playResponse(result.entry);
    } else {
      $("#voice-feedback").hidden = false;
      $("#voice-feedback").textContent = result.spoken_text;
    }
    renderMessages();
  } catch (error) {
    state.failed = state.pending; state.pending = null; renderMessages(); showError(errorMessage(error), true);
  } finally {state.busy = false; controls();}
}

function setPanel(panel) {
  if (panel !== "chat" && state.recording) void finishRecording(true);
  $("#workspace").dataset.panel = panel;
  $$("[data-panel]").filter((node) => node.tagName === "BUTTON").forEach((node) => {node.classList.toggle("active", node.dataset.panel === panel);});
}
async function setView(view) {
  if (view !== "conversation" && state.recording) await finishRecording(true);
  state.view = view;
  $("#workspace").dataset.view = view;
  $$(".nav-button").forEach((button) => {
    const active = button.dataset.view === view; button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  $$(".view").forEach((node) => {node.hidden = node.id !== `${view}-view`;});
  $(".mobile-panels").hidden = view !== "conversation";
  try {
    if (view === "catalogue") await loadProducts();
    if (view === "results") await loadResults();
  } catch (error) {toast(errorMessage(error));}
}

let productsRequest = 0;
async function loadProducts() {
  const token = ++productsRequest;
  const current = state.session?.settings || settings();
  const query = new URLSearchParams({age: current.age, budget: current.budget, need: $("#catalogue-need").value});
  const data = await api(`/api/products?${query}`);
  if (token !== productsRequest) return;
  const matches = data.products.filter((product) => product.eligible);
  const products = $("#eligible-only").checked ? matches : data.products;
  $("#product-count").textContent = `${products.length} products`;
  $("#catalogue-summary").textContent = `${matches.length} profile matches / Age ${current.age} / ${current.budget === "mid" ? "Medium" : human(current.budget)} budget`;
  $("#product-grid").replaceChildren();
  if (!products.length) $("#product-grid").append(element("p", "no-products", "No synthetic products match these constraints."));
  for (const product of products) {
    const article = element("article", `product ${product.eligible ? "eligible" : ""}`);
    const header = element("div", "product-header");
    const mark = element("span");
    mark.append(icon(product.product_type.includes("HEALTH") ? "heart-pulse" : product.product_type === "RETIREMENT" ? "landmark" : "shield"));
    header.append(mark, element("span", `product-match ${product.eligible ? "" : "no-match"}`, product.eligible ? "Profile match" : "Outside profile"));
    const details = element("dl");
    for (const [label, value] of [["Sum insured", product.sum_insured], ["Premium band", product.premium], ["Waiting period", product.waiting_period],
      ["Eligibility", product.eligibility], ["Exclusions", product.exclusions.join(", ") || "None listed"]]) details.append(statePair(label, value));
    article.append(header, element("h2", "", human(product.product_type)), element("div", "product-code", product.product_id), details);
    $("#product-grid").append(article);
  }
  icons();
}
let resultsRequest = 0;
async function loadResults() {
  const token = ++resultsRequest, pipeline = $("#results-pipeline").value;
  $("#results-error").hidden = true;
  $("#results-source").textContent = "Loading recorded experiment";
  $("#benchmark-summary").replaceChildren();
  $("#results-table").replaceChildren(); $("#results-columns").replaceChildren();
  $("#voice-metrics").replaceChildren(); $("#text-chart").hidden = true;
  $("#results-disclaimer").textContent = "";
  let data;
  try { data = await api(`/api/results?pipeline=${pipeline}`); }
  catch (error) {
    if (token !== resultsRequest) return;
    $("#results-source").textContent = "No experiment loaded";
    $("#results-error").textContent = errorMessage(error); $("#results-error").hidden = false; return;
  }
  if (token !== resultsRequest) return;
  const voice = pipeline === "voice", ppo = voice ? data.aggregate.voice_ppo : data.metrics.find((row) => row.agent === "ppo");
  $("#results-source").textContent = `${data.source}${voice ? ` / ${data.training.environment_version}` : ""}`;
  const stats = voice ? [["Voice PPO reward", ppo.reward.mean.toFixed(3), `Seed SD ${ppo.reward.seed_sd.toFixed(3)}`],
    ["Simulated conversion", percent(ppo.conversion.mean, 2), "Not real policy sales"],
    ["Training steps", (data.training.total_timesteps * data.training.seeds.length).toLocaleString(), `${data.training.seeds.length} seeds`]] :
    [["PPO mean reward", ppo.avg_reward.toFixed(2), "Held-out simulation"],
      ["Objection resolution", percent(ppo.objection_resolution, 1), "Applicable episodes"], ["Independent seeds", String(ppo.seeds), "Reported seed means"]];
  for (const [label, value, note] of stats) {
    const stat = element("div", "benchmark-stat");
    stat.append(element("span", "", label), element("strong", "", value), element("small", "", note));
    $("#benchmark-summary").append(stat);
  }
  const columns = voice ? ["Policy", "Reward (mean / SD)", "Conversion", "Violation turns / episode", "Missed stops / episode"] :
    ["Policy", "Reward", "Conversion", "Objection resolution", "Qualified lead"];
  columns.forEach((label) => $("#results-columns").append(element("th", "", label)));
  const rows = voice ? Object.entries(data.aggregate).map(([agent, values]) => ({agent, ...values})) : data.metrics;
  $("#results-caption").textContent = voice ? `${data.training.evaluation_episodes} episodes per policy per seed / ${data.training.seeds.length} seeds` : "Original text simulator / Seed means";
  for (const row of rows) {
    const tr = element("tr", row.agent === (voice ? "voice_ppo" : "ppo") ? "highlight" : "");
    const values = voice ? [human(row.agent), `${row.reward.mean.toFixed(3)} / ${row.reward.seed_sd.toFixed(3)}`,
      percent(row.conversion.mean, 2), row.violations.mean.toFixed(3), row.missed_stop_turns.mean.toFixed(3)] :
      [row.agent === "ppo" ? "PPO" : human(row.agent), row.avg_reward.toFixed(3), percent(row.conversion, 2),
        percent(row.objection_resolution, 1), percent(row.qualified_lead, 1)];
    for (const value of values) tr.append(element("td", "", value));
    $("#results-table").append(tr);
  }
  $("#text-chart").hidden = voice;
  if (voice && data.speech) {
    $("#voice-metrics").append(element("h2", "", "Synthetic speech recognition"));
    const list = element("dl");
    for (const [split, values] of Object.entries(data.speech)) list.append(statePair(`${human(split)} word error rate`, `${(values.wer * 100).toFixed(2)}% / ${values.examples} recordings`));
    $("#voice-metrics").append(list);
  }
  $("#results-disclaimer").textContent = voice ?
    "Recorded before the explicit-stop and guided-intake fixes. These are historical checkpoint results, not evaluation of the current runtime. Synthetic system-voice recordings and simulated customers only. Voice v2 changes the observable dialogue: comparison with the original text experiment cannot isolate retraining benefits. The prior voice checkpoint matches the new checkpoint's conversion in this environment. LLM wording is not evaluated by these scores. Violations and missed stop turns remain nonzero." :
    "Historical synthetic text experiment, before the current explicit-stop and guided-intake fixes. PPO has higher mean reward but lower conversion than the rule baseline. These scores do not measure the current runtime, voice pipeline or LLM wording.";
}
async function refreshStatus() {
  try {
    state.status = await api("/api/status");
    const ready = state.status.ollama.ready;
    $("#model-status").className = `connection ${ready ? "" : "offline"}`;
    $("#model-status").replaceChildren(element("span", "status-dot"), element("span", "", ready ? "Local model ready" : state.status.ollama.online ? "Model missing" : "Local model offline"));
    $("#model-status").title = state.status.ollama.model;
    if (!state.session) {
      if (!state.status.voice.nlp_ready) state.pipeline = "text";
      if (!(state.pipeline === "voice" ? state.status.voice.ppo_ready : state.status.ppo_ready)) state.policy = "rule";
    }
    updateSettings();
  } catch (_) {
    $("#model-status").className = "connection offline";
    $("#model-status").replaceChildren(element("span", "status-dot"), element("span", "", "Demo server offline"));
  }
}
function exportConversation() {
  if (!state.session?.history.length) return;
  const transcript = {...state.session, exported: new Date().toISOString(),
    notice: "Research demo. Synthetic products. NLP estimates and simulated benchmark results are not real customer outcomes."};
  const blob = new Blob([JSON.stringify(transcript, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const anchor = element("a"); anchor.href = url; anchor.download = `insurance-demo-${state.session.id.slice(0, 6)}.json`;
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("Conversation exported");
}

$("#message-form").addEventListener("submit", (event) => {event.preventDefault(); sendMessage(state.failed?.message === $("#message").value.trim() ? state.failed : null);});
$("#message").addEventListener("input", () => {$("#character-count").textContent = `${$("#message").value.length} / 2000`; controls();});
$("#message").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {event.preventDefault(); $("#message-form").requestSubmit();}
});
$("#profile-form").addEventListener("input", updateSettings);
$("#profile-form").addEventListener("change", updateSettings);
$("#profile-form").addEventListener("submit", async (event) => {event.preventDefault(); if (await confirmReset()) await newSession();});
for (const type of ["policy", "generator", "pipeline"]) $$(`[data-${type}]`).forEach((button) => {
  button.onclick = () => {state[type] = button.dataset[type]; updateSettings();};
});
$$(".nav-button").forEach((button) => {button.onclick = () => setView(button.dataset.view);});
$$(".mobile-panels button").forEach((button) => {button.onclick = () => setPanel(button.dataset.panel);});
$("#show-results").onclick = () => {$("#results-pipeline").value = state.session?.settings.pipeline || state.pipeline; setView("results");};
[$("#new-session"), $("#restart")].forEach((button) => {button.onclick = async () => {if (await confirmReset()) {await newSession(); await setView("conversation"); setPanel("chat");}};});
$$("[data-scenario]").forEach((button) => {button.onclick = async () => {
  if (!await confirmReset()) return;
  const scenario = scenarios[button.dataset.scenario];
  setSettings({...settings(), ...scenario});
  if (await newSession()) {
    $$("[data-scenario]").forEach((node) => {node.classList.toggle("selected", node === button);});
    await setView("conversation"); setPanel("chat");
    $("#message").value = scenario.message;
    $("#message").dispatchEvent(new Event("input")); $("#message").focus();
  }
};});
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-message]");
  if (!button || button.disabled || state.busy) return;
  $("#message").value = button.dataset.message;
  $("#message").dispatchEvent(new Event("input")); $("#message").focus();
});
$("#export").onclick = exportConversation;
$("#retry-message").onclick = () => state.failed?.audio ? sendAudio(state.failed.audio, state.failed) : sendMessage(state.failed);
$("#dismiss-error").onclick = hideError;
$("#discard-transcript").onclick = () => {
  state.voiceDraft = null; state.failed = null; $("#message").value = ""; $("#character-count").textContent = "0 / 2000";
  hideError(); controls();
};
$("#catalogue-need").onchange = () => loadProducts().catch((error) => toast(errorMessage(error)));
$("#eligible-only").onchange = () => loadProducts().catch((error) => toast(errorMessage(error)));
$("#results-pipeline").onchange = loadResults;
$("#record").onclick = () => state.recording ? finishRecording() : startRecording();
$("#cancel-recording").onclick = () => finishRecording(true);
$("#stop-playback").onclick = stopPlayback;
$("#auto-speak").onchange = () => {if (!$("#auto-speak").checked) stopPlayback();};
$("#upload-audio").onclick = () => $("#audio-file").click();
$("#audio-file").onchange = () => {
  const file = $("#audio-file").files[0]; $("#audio-file").value = "";
  if (file) void sendAudio(file);
};
document.addEventListener("visibilitychange", () => {if (document.hidden && state.recording) void finishRecording(true);});
window.addEventListener("pagehide", () => {stopPlayback(); if (recorder) void recorder.stop(true); clearInterval(recordTimer);});

async function initialize() {
  icons(); updateSettings();
  await refreshStatus();
  const saved = sessionStorage.getItem("insurance-demo-session");
  if (saved) {
    try {
      state.session = await api(`/api/sessions/${saved}`);
      setSettings(state.session.settings);
      $("#session-label").textContent = `Session ${saved.slice(0, 6).toUpperCase()}`;
      renderMessages(); renderInspector(state.session.history.at(-1) || null); controls();
    } catch (_) {sessionStorage.removeItem("insurance-demo-session"); await newSession();}
  } else await newSession();
  setInterval(refreshStatus, 20000);
}
initialize();
