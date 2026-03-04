/**
 * renderer.js
 * Gladden AI Assistant — Renderer Process
 *
 * Responsibilities:
 *  - Show/hide the overlay with slide animation
 *  - Capture user input and POST to Python backend
 *  - Stream events into the log panel with animated entries
 *  - Handle click-through: bar captures mouse, transparent areas don't
 *  - Wire keyboard shortcuts (Escape, Ctrl+Space via main process signals)
 */

'use strict';

const BACKEND_URL = 'http://localhost:8000/execute';

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const app         = document.getElementById('app');
const bar         = document.getElementById('interactive-zone');
const input       = document.getElementById('cmd-input');
const logInner    = document.getElementById('log-inner');
const statusDot   = document.getElementById('status-dot');

/* ── State ──────────────────────────────────────────────────────────────── */
let isExecuting = false;

/* ═══════════════════════════════════════════════════════════════════════════
   VISIBILITY
   ═════════════════════════════════════════════════════════════════════════ */

function showOverlay() {
  app.classList.remove('hidden');
  app.classList.add('visible');
  // Always make bar clickable when visible
  window.gladden.setIgnoreMouseEvents(false);
  requestAnimationFrame(() => input.focus());
}

function hideOverlay() {
  app.classList.add('hidden');
  app.classList.remove('visible');
  window.gladden.hideWindow();
}

window.addEventListener('DOMContentLoaded', () => {
  showOverlay();
  input.focus();
});

window.gladden.onTriggerHide(() => hideOverlay());
window.gladden.onTriggerShow(() => { showOverlay(); input.focus(); });

/* ═══════════════════════════════════════════════════════════════════════════
   CLICK-THROUGH
   The window is NEVER fully click-through while visible — the bar must
   always be clickable. We simply keep ignore=false the entire time.
   ═════════════════════════════════════════════════════════════════════════ */

// Keep the bar always interactive — no toggling needed since the window
// is sized to exactly fit the bar anyway.
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
    hideOverlay();
  }
});

// Also handle click anywhere on bar to re-focus input
bar.addEventListener('click', (e) => {
  if (e.target !== input) {
    input.focus();
  }
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

  appendLogEntry({ type: 'info', stage: 'agent_start', message: `→ ${instruction}` });

  try {
    const response = await fetch(BACKEND_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ instruction }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status} — ${response.statusText}`);

    const data = await response.json();

    if (Array.isArray(data.events) && data.events.length > 0) {
      await streamEvents(data.events);
    }

    if (data.success) {
      appendLogEntry({ type: 'status', stage: 'done', message: data.output ?? 'Completed.', className: 'result-ok' });
      setStatus('success');
    } else {
      appendLogEntry({ type: 'error', stage: 'failed', message: data.error ?? 'Unknown error.', className: 'result-err' });
      setStatus('error');
    }

  } catch (err) {
    const isConnErr = err.message.includes('Failed to fetch') ||
                      err.message.includes('ERR_CONNECTION_REFUSED') ||
                      err.message.includes('NetworkError');
    appendLogEntry({
      type: 'error', stage: 'network_error',
      message: isConnErr
        ? 'Backend not reachable — is the Python server running on :8000?'
        : err.message,
      className: 'result-err',
    });
    setStatus('error');
  } finally {
    isExecuting = false;
    input.classList.remove('executing');
    input.disabled = false;
    input.value = '';
    input.focus();
    setTimeout(() => setStatus('idle'), 3000);
  }
}

async function streamEvents(events) {
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    appendLogEntry({ type: ev.type ?? 'info', stage: ev.stage ?? '', message: ev.message ?? '', ts: ev.timestamp });
    await sleep(events.length > 20 ? 20 : 40);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOG PANEL
   ═════════════════════════════════════════════════════════════════════════ */

function clearLog() { logInner.innerHTML = ''; }

function appendLogEntry({ type = 'info', stage = '', message = '', ts, className = '' }) {
  const entry = document.createElement('div');
  entry.classList.add('log-entry');
  if (className) entry.classList.add(className);
  entry.setAttribute('data-type', type);
  entry.innerHTML = `
    <span class="log-ts">${escapeHtml(formatTs(ts))}</span>
    <span class="log-stage">${escapeHtml(stage)}</span>
    <span class="log-msg">${escapeHtml(message)}</span>
  `;
  logInner.appendChild(entry);
  scrollLogToBottom();
}

function scrollLogToBottom() {
  requestAnimationFrame(() => { logInner.scrollTop = logInner.scrollHeight; });
}

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS DOT
   ═════════════════════════════════════════════════════════════════════════ */

function setStatus(state) {
  statusDot.className = `status-dot ${state}`;
  statusDot.title = state;
}

/* ═══════════════════════════════════════════════════════════════════════════
   UTILITIES
   ═════════════════════════════════════════════════════════════════════════ */

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatTs(iso) {
  const d = iso ? new Date(iso) : new Date();
  try { return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; }
  catch { return '--:--:--'; }
}

function pad(n) { return String(n).padStart(2, '0'); }