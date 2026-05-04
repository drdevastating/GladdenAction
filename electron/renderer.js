'use strict';

const BACKEND_URL          = 'http://localhost:8000/execute';
const VOICE_START_URL      = 'http://localhost:8000/voice/start';
const VOICE_STOP_URL       = 'http://localhost:8000/voice/stop';
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

/* ── NL command card (inline, below bar, no log panel) ─────────────────── */
let nlCard = null;

/* ── Constants ──────────────────────────────────────────────────────────── */
const BAR_HEIGHT       = 56;
const LOG_HEIGHT       = 380;
const APPROVAL_HEIGHT  = 460;
const NL_CARD_HEIGHT   = 78;
const PADDING          = 16;

/* ── State ──────────────────────────────────────────────────────────────── */
let isExecuting        = false;
let isRecording        = false;      // mic is capturing audio on the backend
let isTranscribing     = false;      // waiting for stop+transcription result
let logOpen            = false;
let voiceReady         = false;
let approvalPending    = false;
let approvalTimer      = null;
let approvalCountdownEl = null;
let activeStopController = null;     // AbortController for the /voice/stop fetch

/* ── NL pattern detection ───────────────────────────────────────────────── */
const NL_PATTERNS = [
  /how do i .+ (terminal|cmd|command|powershell)/i,
  /give me the command/i,
  /what('s| is) the command/i,
  /run a command that/i,
  /find (all |files? that)/i,
  /check what('s| is) (on |listening on )?port/i,
  /which process is (on|using) port/i,
  /show (me )?(disk|memory|network)/i,
  /compress|zip|archive/i,
  /kill the process on port/i,
  /list all (running|open|env)/i,
  /count (lines|files|words)/i,
  /i always forget the command/i,
  /i don'?t remember how to .+ terminal/i,
  /what (git|npm|pip|docker) command/i,
];

function isNLCommand(instruction) {
  return NL_PATTERNS.some(p => p.test(instruction));
}

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
      micBtn.title = 'Voice command — click to start recording, click again to stop';
    } else {
      micBtn.classList.add('not-ready');
      micBtn.title = 'Voice unavailable — run: pip install faster-whisper sounddevice';
    }
  } catch (_) {
    micBtn.classList.add('not-ready');
    micBtn.title = 'Backend not reachable';
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   NL COMMAND CARD
   ═════════════════════════════════════════════════════════════════════════ */

function showNLCard(cmd) {
  dismissNLCard();

  const card = document.createElement('div');
  card.id = 'nl-card';
  card.className = 'nl-card';

  const body = document.createElement('div');
  body.className = 'nl-card-body';

  const icon = document.createElement('span');
  icon.className = 'nl-card-icon';
  icon.textContent = '⌘';

  const code = document.createElement('span');
  code.className = 'nl-card-code';
  code.textContent = cmd;

  body.appendChild(icon);
  body.appendChild(code);

  const actions = document.createElement('div');
  actions.className = 'nl-card-actions';

  const copyBtn = document.createElement('button');
  copyBtn.className = 'nl-copy-btn';
  copyBtn.type = 'button';
  copyBtn.title = 'Copy to clipboard';
  copyBtn.innerHTML = `
    <svg class="nl-copy-icon" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="7" y="7" width="10" height="12" rx="2" stroke="currentColor" stroke-width="1.5"/>
      <path d="M13 7V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    <span class="nl-copy-label">Copy</span>
  `;

  copyBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(cmd).then(() => {
      copyBtn.classList.add('copied');
      copyBtn.innerHTML = `
        <svg class="nl-copy-icon" viewBox="0 0 20 20" fill="none">
          <path d="M4 10l4 4 8-8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span class="nl-copy-label">Copied!</span>
      `;
      setTimeout(() => {
        if (!card.isConnected) return;
        copyBtn.classList.remove('copied');
        copyBtn.innerHTML = `
          <svg class="nl-copy-icon" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="7" y="7" width="10" height="12" rx="2" stroke="currentColor" stroke-width="1.5"/>
            <path d="M13 7V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          <span class="nl-copy-label">Copy</span>
        `;
      }, 2000);
    }).catch(() => {
      copyBtn.querySelector('.nl-copy-label').textContent = 'Failed';
      setTimeout(() => { copyBtn.querySelector('.nl-copy-label').textContent = 'Copy'; }, 1500);
    });
  });

  actions.appendChild(copyBtn);
  card.appendChild(body);
  card.appendChild(actions);
  card.addEventListener('click', (e) => e.stopPropagation());
  app.appendChild(card);
  nlCard = card;

  window.gladden.resizeWindow(BAR_HEIGHT + NL_CARD_HEIGHT + PADDING + 8);
}

function dismissNLCard() {
  if (nlCard && nlCard.isConnected) {
    nlCard.classList.add('nl-card-exit');
    setTimeout(() => { if (nlCard) { nlCard.remove(); nlCard = null; } }, 220);
  } else {
    nlCard = null;
  }
  if (!logOpen) {
    window.gladden.resizeWindow(BAR_HEIGHT + PADDING);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOG PANEL
   ═════════════════════════════════════════════════════════════════════════ */

function openLog(height) {
  if (logOpen) return;
  dismissNLCard();
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
   VISIBILITY / HAZE
   ═════════════════════════════════════════════════════════════════════════ */

function showOverlay() {
  app.classList.remove('hidden');
  app.classList.add('visible');
  window.gladden.setIgnoreMouseEvents(false);
  window.gladden.showWindow();
  requestAnimationFrame(() => input.focus());
}

function hideOverlay() {
  if (isRecording || isTranscribing) return;
  closeLog();
  dismissNLCard();
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

document.addEventListener('click', (e) => {
  if (nlCard && !nlCard.contains(e.target)) dismissNLCard();
});

/* ═══════════════════════════════════════════════════════════════════════════
   INPUT HANDLING
   ═════════════════════════════════════════════════════════════════════════ */

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    // If currently recording, Enter = stop recording
    if (isRecording) { stopRecording(); return; }
    if (isTranscribing) return;
    if (!isExecuting) {
      const instruction = input.value.trim();
      if (instruction) {
        dismissNLCard();
        executeInstruction(instruction);
      }
    }
    return;
  }
  if (e.key === 'Escape') {
    if (isRecording || isTranscribing) { cancelRecording(); return; }
    if (approvalPending) { sendReject(); return; }
    if (nlCard) { dismissNLCard(); return; }
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
  if (isExecuting) return;
  if (isTranscribing) return;  // prevent double-click during transcription
  if (isRecording) stopRecording();
  else startRecording();
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

  let remaining = delay;
  approvalTimer = setInterval(() => {
    remaining--;
    if (approvalCountdownEl) approvalCountdownEl.textContent = remaining + 's';
    if (remaining <= 0) clearApprovalBanner();
  }, 1000);

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
   VOICE RECORDING  (fixed two-phase flow)

   Phase 1 — startRecording():
     POST /voice/start  →  backend starts recording thread, returns immediately
     UI shows "recording…" state

   Phase 2 — stopRecording():
     POST /voice/stop   →  backend signals stop, waits for transcription,
                           returns { success, transcript }
     UI shows "transcribing…" briefly then displays transcript
   ═════════════════════════════════════════════════════════════════════════ */

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

async function startRecording() {
  if (!voiceReady) {
    appendLogEntry({
      type: 'error',
      stage: 'voice_unavailable',
      message: 'Voice deps not installed. Run: pip install faster-whisper sounddevice',
    });
    openLog();
    return;
  }
  if (isRecording || isTranscribing || isExecuting) return;

  try {
    const res = await fetch(VOICE_START_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_size: 'large-v3', max_duration: 30.0 }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'HTTP ' + res.status);
    }

    const data = await res.json();
    if (!data.recording) {
      throw new Error(data.error || 'Backend did not start recording.');
    }

    // Recording is now active on the backend
    isRecording = true;
    input.value = '';
    input.placeholder = 'Recording… click mic or press Enter to stop';
    input.classList.add('voice-active');
    setMicState('recording');
    setStatus('loading');
    window.gladden.setHaze(true);

  } catch (err) {
    isRecording = false;
    input.placeholder = 'Ask Gladden…';
    input.classList.remove('voice-active');
    setMicState('idle');
    setStatus('error');
    appendLogEntry({ type: 'error', stage: 'voice_start_failed', message: err.message, className: 'result-err' });
    openLog();
    setTimeout(() => setStatus('idle'), 3000);
  }
}

async function stopRecording() {
  if (!isRecording) return;

  // Transition to "transcribing" state immediately for UX
  isRecording = false;
  isTranscribing = true;
  input.placeholder = 'Transcribing…';
  input.classList.remove('voice-active');
  input.classList.add('executing');
  setMicState('idle');
  setStatus('loading');

  activeStopController = new AbortController();

  try {
    // POST /voice/stop — this blocks until transcription is done (up to ~30s)
    const res = await fetch(VOICE_STOP_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
      signal: activeStopController.signal,
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'HTTP ' + res.status);
    }

    const data = await res.json();

    if (data.success && data.transcript) {
      // Put transcript into the input and let the user submit it
      input.value = data.transcript;
      input.classList.remove('executing');
      input.placeholder = 'Ask Gladden…';
      input.classList.add('voice-confirm');
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);

      // Remove confirm highlight after a moment
      setTimeout(() => input.classList.remove('voice-confirm'), 1500);

      setStatus('success');
      setTimeout(() => setStatus('idle'), 2000);
    } else {
      input.value = '';
      input.classList.remove('executing');
      input.placeholder = 'Ask Gladden…';
      const msg = data.error || 'No speech detected.';
      appendLogEntry({ type: 'error', stage: 'transcription_failed', message: msg, className: 'result-err' });
      openLog();
      setStatus('error');
      setTimeout(() => setStatus('idle'), 3000);
    }

  } catch (err) {
    if (err.name !== 'AbortError') {
      const isConn = err.message.includes('Failed to fetch') || err.message.includes('ERR_CONNECTION');
      appendLogEntry({
        type: 'error',
        stage: 'voice_stop_failed',
        message: isConn ? 'Backend not reachable.' : err.message,
        className: 'result-err',
      });
      openLog();
      setStatus('error');
      setTimeout(() => setStatus('idle'), 3000);
    }
    input.classList.remove('executing');
    input.placeholder = 'Ask Gladden…';
  } finally {
    isTranscribing = false;
    isRecording = false;
    activeStopController = null;
    setMicState('idle');
    input.focus();
  }
}

function cancelRecording() {
  if (!isRecording && !isTranscribing) return;

  // Abort the pending /voice/stop fetch if active
  if (activeStopController) {
    activeStopController.abort();
    activeStopController = null;
  }

  // Also signal the backend to stop capturing (fire-and-forget)
  if (isRecording) {
    fetch(VOICE_STOP_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
      .catch(() => {});
  }

  isRecording = false;
  isTranscribing = false;
  input.value = '';
  input.classList.remove('voice-active', 'executing', 'voice-confirm');
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

  const isNL = isNLCommand(instruction);

  isExecuting = true;
  input.classList.add('executing');
  input.disabled = true;
  setStatus('loading');

  if (!isNL) {
    clearLog();
    openLog();
    appendLogEntry({ type: 'info', stage: 'agent_start', message: '→ ' + instruction });
  }

  try {
    const response = await fetch(BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ instruction }),
    });
    if (!response.ok) throw new Error('HTTP ' + response.status + ' — ' + response.statusText);
    const data = await response.json();

    if (isNL) {
      let translatedCmd = null;

      if (Array.isArray(data.events)) {
        for (const ev of data.events) {
          if (ev.stage === 'command_ready') {
            translatedCmd = (ev.message || '').replace(/^Translated command:\s*/i, '').trim();
            break;
          }
        }
      }

      if (!translatedCmd && data.output) {
        translatedCmd = String(data.output).replace(/^Command \(not executed\):\s*/i, '').trim();
      }

      if (translatedCmd) {
        showNLCard(translatedCmd);
        setStatus('success');
      } else {
        openLog();
        appendLogEntry({ type: 'error', stage: 'failed', message: data.error || 'Could not translate command.', className: 'result-err' });
        setStatus('error');
      }
    } else {
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
    }
  } catch (err) {
    const isConnErr = err.message.includes('Failed to fetch') || err.message.includes('ERR_CONNECTION_REFUSED');
    if (!isNL) appendSeparator();
    else openLog();
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
    if (ev.stage === 'stdout_line' || ev.stage === 'stderr_line') {
      appendTerminalLine(ev);
      await sleep(8);
      continue;
    }
    if (ev.stage === 'command_ready') {
      await sleep(delay);
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

  const tsEl = document.createElement('span');
  tsEl.className = 'log-ts';
  tsEl.textContent = formatTs(ts);

  const stageEl = document.createElement('span');
  stageEl.className = 'log-stage';
  stageEl.textContent = stage;

  const msgEl = document.createElement('span');
  msgEl.className = 'log-msg';
  msgEl.textContent = message;

  entry.appendChild(tsEl);
  entry.appendChild(stageEl);
  entry.appendChild(msgEl);
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
  const d = iso ? new Date(iso) : new Date();
  try { return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()); }
  catch(e) { return '--:--:--'; }
}

function pad(n) { return String(n).padStart(2, '0'); }