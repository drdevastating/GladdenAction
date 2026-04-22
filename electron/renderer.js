'use strict';

const BACKEND_URL          = 'http://localhost:8000/execute';
const VOICE_TRANSCRIBE_URL = 'http://localhost:8000/voice/transcribe';
const VOICE_CHECK_URL      = 'http://localhost:8000/voice/check';
const APPROVE_URL          = 'http://localhost:8000/approve';
const REJECT_URL           = 'http://localhost:8000/reject';

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const app        = document.getElementById('app');
const bar        = document.getElementById('interactive-zone');
const input      = document.getElementById('cmd-input');
const logPanel   = document.getElementById('log-panel');
const logInner   = document.getElementById('log-inner');
const statusDot  = document.getElementById('status-dot');
const closeBtn   = document.getElementById('log-close-btn');
const micBtn     = document.getElementById('mic-btn');
const micRipple  = document.getElementById('mic-ripple');
const micWaves   = document.getElementById('mic-waves');

/* ── Constants ──────────────────────────────────────────────────────────── */
const BAR_HEIGHT       = 56;
const LOG_HEIGHT       = 380;
const APPROVAL_HEIGHT  = 460;
const PADDING          = 16;

/* ── State ──────────────────────────────────────────────────────────────── */
let isExecuting    = false;
let isRecording    = false;
let isTranscribing = false;
let logOpen        = false;
let voiceReady     = false;
let approvalPending = false;
let approvalTimer  = null;
let approvalCountdownEl = null;

/* ═══════════════════════════════════════════════════════════════════════════
   VOICE READINESS CHECK
   ═════════════════════════════════════════════════════════════════════════ */

async function checkVoiceReady() {
  try {
    const res = await fetch(VOICE_CHECK_URL);
    if (!res.ok) return;
    const data = await res.json();
    voiceReady = data.ready === true;
    if (voiceReady) {
      micBtn.classList.add('ready');
      micBtn.title = 'Voice command — click to start, click again to stop';
    } else {
      micBtn.classList.add('not-ready');
      micBtn.title = 'Voice unavailable — install: pip install faster-whisper sounddevice';
    }
  } catch (_) {
    micBtn.classList.add('not-ready');
    micBtn.title = 'Backend not reachable';
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOG PANEL OPEN / CLOSE
   ═════════════════════════════════════════════════════════════════════════ */

function openLog(height) {
  if (logOpen) return;
  logOpen = true;
  logPanel.classList.add('expanded');
  bar.classList.add('log-open');
  window.gladden.resizeWindow(BAR_HEIGHT + (height || LOG_HEIGHT) + PADDING);
}

function closeLog() {
  if (!logOpen) return;
  logOpen = false;
  logPanel.classList.remove('expanded');
  bar.classList.remove('log-open');
  window.gladden.resizeWindow(BAR_HEIGHT + PADDING);
  clearApprovalBanner();
}

closeBtn.addEventListener('click', () => closeLog());

/* ═══════════════════════════════════════════════════════════════════════════
   VISIBILITY
   ═════════════════════════════════════════════════════════════════════════ */

function showOverlay() {
  app.classList.remove('hidden');
  app.classList.add('visible');
  window.gladden.setIgnoreMouseEvents(false);
  requestAnimationFrame(() => input.focus());
}

function hideOverlay() {
  if (isRecording || isTranscribing) return;
  closeLog();
  app.classList.add('hidden');
  app.classList.remove('visible');
  window.gladden.hideWindow();
}

window.addEventListener('DOMContentLoaded', () => {
  showOverlay();
  window.gladden.resizeWindow(BAR_HEIGHT + PADDING);
  input.focus();
  checkVoiceReady();
});

window.gladden.onTriggerHide(() => hideOverlay());
window.gladden.onTriggerShow(() => { showOverlay(); input.focus(); });
window.gladden.setIgnoreMouseEvents(false);

/* ═══════════════════════════════════════════════════════════════════════════
   INPUT HANDLING
   ═════════════════════════════════════════════════════════════════════════ */

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    if (isRecording) { stopRecording(); return; }
    if (isTranscribing) return;
    if (!isExecuting) {
      const instruction = input.value.trim();
      if (instruction) executeInstruction(instruction);
    }
    return;
  }
  if (e.key === 'Escape') {
    if (isRecording || isTranscribing) { cancelRecording(); return; }
    if (approvalPending) { sendReject(); return; }
    if (logOpen) { closeLog(); return; }
    hideOverlay();
  }
});

window.addEventListener('keydown', (e) => {
  if (e.key === ' ' && document.activeElement !== input) {
    e.preventDefault();
    if (isRecording) stopRecording();
    else if (!isExecuting && !isTranscribing) startRecording();
  }
});

bar.addEventListener('click', (e) => {
  if (e.target !== input && e.target !== micBtn && !micBtn.contains(e.target)) {
    input.focus();
  }
});

micBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  if (isTranscribing || isExecuting) return;
  if (isRecording) stopRecording(); else startRecording();
});

/* ═══════════════════════════════════════════════════════════════════════════
   APPROVAL GATE UI
   ═════════════════════════════════════════════════════════════════════════ */

function showApprovalBanner(event) {
  approvalPending = true;
  clearApprovalBanner();

  const banner = document.createElement('div');
  banner.id = 'approval-banner';
  banner.className = 'approval-banner';

  const isDestructive = event.destructive;
  const delay = event.auto_approve_in || 8;

  let planHtml = '';
  if (event.plan && event.plan.length) {
    planHtml = '<div class="approval-plan">' + event.plan.map(step => {
      const icon = step.destructive ? '🔴' : '🔵';
      return `<div class="plan-step ${step.destructive ? 'destructive' : ''}">
        <span class="step-icon">${icon}</span>
        <span class="step-num">${step.step}</span>
        <span class="step-desc">${escapeHtml(step.description)}</span>
      </div>`;
    }).join('') + '</div>';
  }

  banner.innerHTML = `
    <div class="approval-header ${isDestructive ? 'destructive' : ''}">
      <span class="approval-icon">${isDestructive ? '⚠' : '▶'}</span>
      <span class="approval-title">${isDestructive ? 'Destructive Plan — Review Required' : 'Plan Ready'}</span>
      <span id="approval-countdown" class="approval-countdown">${delay}s</span>
    </div>
    <div class="approval-msg">${escapeHtml(event.message)}</div>
    ${planHtml}
    <div class="approval-actions">
      <button class="approval-btn reject-btn" id="reject-btn">✕ Cancel</button>
      <button class="approval-btn approve-btn" id="approve-btn">▶ Run Now</button>
    </div>
    <div class="approval-progress">
      <div class="approval-progress-bar" id="approval-progress" style="animation-duration: ${delay}s"></div>
    </div>
  `;

  logInner.insertBefore(banner, logInner.firstChild);
  approvalCountdownEl = document.getElementById('approval-countdown');

  document.getElementById('approve-btn').addEventListener('click', () => sendApprove());
  document.getElementById('reject-btn').addEventListener('click', () => sendReject());

  // Countdown timer
  let remaining = delay;
  approvalTimer = setInterval(() => {
    remaining--;
    if (approvalCountdownEl) approvalCountdownEl.textContent = remaining + 's';
    if (remaining <= 0) {
      clearApprovalBanner();
    }
  }, 1000);

  // Trigger progress bar animation
  requestAnimationFrame(() => {
    const bar = document.getElementById('approval-progress');
    if (bar) bar.classList.add('running');
  });

  openLog(APPROVAL_HEIGHT);
  scrollLogToBottom();
}

function clearApprovalBanner() {
  approvalPending = false;
  if (approvalTimer) { clearInterval(approvalTimer); approvalTimer = null; }
  const existing = document.getElementById('approval-banner');
  if (existing) existing.remove();
}

async function sendApprove() {
  clearApprovalBanner();
  try {
    await fetch(APPROVE_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  } catch (_) {}
}

async function sendReject() {
  clearApprovalBanner();
  try {
    await fetch(REJECT_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  } catch (_) {}
  appendLogEntry({ type: 'error', stage: 'plan_rejected', message: 'Plan cancelled by user.', className: 'result-err' });
}

/* ═══════════════════════════════════════════════════════════════════════════
   VOICE RECORDING
   ═════════════════════════════════════════════════════════════════════════ */

let recordingAbortController = null;

function setMicState(state) {
  micBtn.setAttribute('data-state', state);
  if (state === 'recording') {
    micBtn.classList.add('active');
    micRipple.classList.add('pulsing');
    micWaves.classList.add('animating');
  } else {
    micBtn.classList.remove('active');
    micRipple.classList.remove('pulsing');
    micWaves.classList.remove('animating');
  }
}

function startRecording() {
  if (!voiceReady) {
    appendLogEntry({ type: 'error', stage: 'voice_unavailable', message: 'Voice deps not installed.' });
    openLog();
    return;
  }
  if (isRecording || isTranscribing || isExecuting) return;
  isRecording = true;
  input.value = '';
  input.placeholder = 'Recording… press Enter or click mic to stop';
  input.classList.add('voice-active');
  setMicState('recording');
  setStatus('loading');
}

async function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  isTranscribing = true;
  input.placeholder = 'Transcribing…';
  setMicState('transcribing');
  recordingAbortController = new AbortController();
  try {
    const response = await fetch(VOICE_TRANSCRIBE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_size: 'base', max_duration: 30.0 }),
      signal: recordingAbortController.signal,
    });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const data = await response.json();
    if (data.success && data.transcript) {
      input.value = data.transcript;
      input.classList.remove('voice-active');
      input.placeholder = 'Ask Gladden…';
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
      setStatus('success');
      setTimeout(() => setStatus('idle'), 2000);
    } else {
      input.value = '';
      input.classList.remove('voice-active');
      input.placeholder = 'Ask Gladden…';
      appendLogEntry({ type: 'error', stage: 'transcription_failed', message: data.error || 'No speech detected.', className: 'result-err' });
      openLog();
      setStatus('error');
      setTimeout(() => setStatus('idle'), 3000);
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      appendLogEntry({ type: 'error', stage: 'network_error', message: err.message, className: 'result-err' });
      openLog();
      setStatus('error');
      setTimeout(() => setStatus('idle'), 3000);
    }
    input.classList.remove('voice-active');
    input.placeholder = 'Ask Gladden…';
  } finally {
    isTranscribing = false;
    isRecording = false;
    recordingAbortController = null;
    setMicState('idle');
    input.focus();
  }
}

function cancelRecording() {
  if (!isRecording && !isTranscribing) return;
  if (recordingAbortController) recordingAbortController.abort();
  isRecording = false;
  isTranscribing = false;
  input.value = '';
  input.classList.remove('voice-active');
  input.placeholder = 'Ask Gladden…';
  setMicState('idle');
  setStatus('idle');
  input.focus();
}

/* ═══════════════════════════════════════════════════════════════════════════
   TEXT EXECUTION
   ═════════════════════════════════════════════════════════════════════════ */

async function executeInstruction(instruction) {
  if (isExecuting) return;
  isExecuting = true;
  input.classList.add('executing');
  input.disabled = true;
  setStatus('loading');
  clearLog();
  openLog();
  appendLogEntry({ type: 'info', stage: 'agent_start', message: '→ ' + instruction });

  try {
    const response = await fetch(BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ instruction }),
    });
    if (!response.ok) throw new Error('HTTP ' + response.status + ' — ' + response.statusText);
    const data = await response.json();
    if (Array.isArray(data.events) && data.events.length > 0) {
      await streamEvents(data.events);
    }
    appendSeparator();
    if (data.success) {
      appendLogEntry({ type: 'status', stage: 'done', message: data.output || 'Completed.', className: 'result-ok' });
      setStatus('success');
    } else {
      appendLogEntry({ type: 'error', stage: 'failed', message: data.error || 'Unknown error.', className: 'result-err' });
      setStatus('error');
    }
  } catch (err) {
    const isConnErr = err.message.includes('Failed to fetch') || err.message.includes('ERR_CONNECTION_REFUSED');
    appendSeparator();
    appendLogEntry({
      type: 'error', stage: 'network_error',
      message: isConnErr ? 'Backend not reachable — is the Python server running on :8000?' : err.message,
      className: 'result-err',
    });
    setStatus('error');
  } finally {
    isExecuting = false;
    input.classList.remove('executing');
    input.disabled = false;
    input.value = '';
    input.focus();
    setTimeout(() => setStatus('idle'), 4000);
  }
}

async function streamEvents(events) {
  const delay = events.length > 40 ? 10 : events.length > 20 ? 20 : 35;
  for (const ev of events) {
    // Approval events get special treatment
    if (ev.type === 'approval_required') {
      showApprovalBanner(ev);
      await sleep(200);
      continue;
    }
    if (ev.type === 'approval_decision') {
      clearApprovalBanner();
      appendLogEntry({
        type: ev.stage === 'rejected' ? 'error' : 'status',
        stage: ev.stage,
        message: ev.message || '',
      });
      await sleep(200);
      continue;
    }
    // Terminal output lines — use monospace styling
    if (ev.stage === 'stdout_line' || ev.stage === 'stderr_line') {
      appendTerminalLine(ev);
      await sleep(8);   // fast scroll for terminal output
      continue;
    }
    appendLogEntry({
      type:    ev.type    || 'info',
      stage:   ev.stage   || '',
      message: ev.message || '',
      ts:      ev.timestamp,
    });
    await sleep(delay);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOG HELPERS
   ═════════════════════════════════════════════════════════════════════════ */

function clearLog() {
  logInner.innerHTML = '';
  clearApprovalBanner();
}

function appendSeparator() {
  const sep = document.createElement('div');
  sep.className = 'log-separator';
  logInner.appendChild(sep);
}

function appendLogEntry({ type, stage, message, ts, className }) {
  type      = type      || 'info';
  stage     = stage     || '';
  message   = message   || '';
  className = className || '';
  const entry = document.createElement('div');
  entry.classList.add('log-entry');
  if (className) entry.classList.add(className);
  entry.setAttribute('data-type', type);
  entry.innerHTML =
    '<span class="log-ts">'    + escapeHtml(formatTs(ts))  + '</span>' +
    '<span class="log-stage">' + escapeHtml(stage)          + '</span>' +
    '<span class="log-msg">'   + escapeHtml(message)        + '</span>';
  logInner.appendChild(entry);
  scrollLogToBottom();
}

function appendTerminalLine(ev) {
  const isErr = ev.stage === 'stderr_line';
  const entry = document.createElement('div');
  entry.classList.add('terminal-line');
  if (isErr) entry.classList.add('stderr');
  entry.textContent = ev.message || '';
  logInner.appendChild(entry);
  scrollLogToBottom();
}

function scrollLogToBottom() {
  requestAnimationFrame(() => { logInner.scrollTop = logInner.scrollHeight; });
}

function setStatus(state) {
  statusDot.className = 'status-dot ' + state;
  statusDot.title = state;
}

function sleep(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatTs(iso) {
  var d = iso ? new Date(iso) : new Date();
  try { return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()); }
  catch(e) { return '--:--:--'; }
}

function pad(n) { return String(n).padStart(2, '0'); }