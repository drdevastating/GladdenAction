'use strict';

const BACKEND_URL = 'http://localhost:8000/execute';

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const app        = document.getElementById('app');
const bar        = document.getElementById('interactive-zone');
const input      = document.getElementById('cmd-input');
const logPanel   = document.getElementById('log-panel');
const logInner   = document.getElementById('log-inner');
const statusDot  = document.getElementById('status-dot');
const closeBtn   = document.getElementById('log-close-btn');

/* ── Constants ──────────────────────────────────────────────────────────── */
const BAR_HEIGHT = 56;   // px — matches --bar-height in CSS
const LOG_HEIGHT = 340;  // px — matches --log-height in CSS
const PADDING    = 16;   // px — vertical padding

/* ── State ──────────────────────────────────────────────────────────────── */
let isExecuting = false;
let logOpen     = false;

/* ═══════════════════════════════════════════════════════════════════════════
   LOG PANEL OPEN / CLOSE
   ═════════════════════════════════════════════════════════════════════════ */

function openLog() {
  if (logOpen) return;
  logOpen = true;
  logPanel.classList.add('expanded');
  bar.classList.add('log-open');
  window.gladden.resizeWindow(BAR_HEIGHT + LOG_HEIGHT + PADDING);
}

function closeLog() {
  if (!logOpen) return;
  logOpen = false;
  logPanel.classList.remove('expanded');
  bar.classList.remove('log-open');
  window.gladden.resizeWindow(BAR_HEIGHT + PADDING);
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
  closeLog();
  app.classList.add('hidden');
  app.classList.remove('visible');
  window.gladden.hideWindow();
}

window.addEventListener('DOMContentLoaded', () => {
  showOverlay();
  window.gladden.resizeWindow(BAR_HEIGHT + PADDING);
  input.focus();
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
    if (!isExecuting) {
      const instruction = input.value.trim();
      if (instruction) executeInstruction(instruction);
    }
    return;
  }
  if (e.key === 'Escape') {
    if (logOpen) { closeLog(); return; }
    hideOverlay();
  }
});

bar.addEventListener('click', (e) => {
  if (e.target !== input) input.focus();
});

/* ═══════════════════════════════════════════════════════════════════════════
   EXECUTION
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
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ instruction }),
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
    const isConnErr = err.message.includes('Failed to fetch') ||
                      err.message.includes('ERR_CONNECTION_REFUSED') ||
                      err.message.includes('NetworkError');
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
  const delay = events.length > 30 ? 15 : events.length > 15 ? 25 : 40;
  for (const ev of events) {
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

function clearLog() { logInner.innerHTML = ''; }

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

function scrollLogToBottom() {
  requestAnimationFrame(() => { logInner.scrollTop = logInner.scrollHeight; });
}

/* ── Status dot ─────────────────────────────────────────────────────────── */
function setStatus(state) {
  statusDot.className = 'status-dot ' + state;
  statusDot.title = state;
}

/* ── Utilities ──────────────────────────────────────────────────────────── */
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