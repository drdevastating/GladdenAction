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

/* ── Backend endpoint ───────────────────────────────────────────────────── */
const BACKEND_URL = 'http://localhost:8000/execute';

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const app         = document.getElementById('app');
const bar         = document.getElementById('interactive-zone');
const input       = document.getElementById('cmd-input');
const logInner    = document.getElementById('log-inner');
const statusDot   = document.getElementById('status-dot');
const placeholder = logInner.querySelector('.log-placeholder');

/* ── State ──────────────────────────────────────────────────────────────── */
let isExecuting = false;

/* ═══════════════════════════════════════════════════════════════════════════
   VISIBILITY
   ═════════════════════════════════════════════════════════════════════════ */

function showOverlay() {
  app.classList.remove('hidden');
  app.classList.add('visible');
  requestAnimationFrame(() => input.focus());
}

function hideOverlay() {
  app.classList.add('hidden');
  app.classList.remove('visible');
  window.gladden.hideWindow();
}

// Initialise visible on load
window.addEventListener('DOMContentLoaded', () => {
  showOverlay();
  input.focus();
});

// Listen for hide/show signals from main process (global hotkeys)
window.gladden.onTriggerHide(() => hideOverlay());
window.gladden.onTriggerShow(() => {
  showOverlay();
  input.focus();
});

/* ═══════════════════════════════════════════════════════════════════════════
   CLICK-THROUGH — pass mouse events unless hovering over the bar
   ═════════════════════════════════════════════════════════════════════════ */

bar.addEventListener('mouseenter', () => {
  window.gladden.setIgnoreMouseEvents(false);
});

bar.addEventListener('mouseleave', () => {
  if (!isExecuting) {
    window.gladden.setIgnoreMouseEvents(true, { forward: true });
  }
});

/* ═══════════════════════════════════════════════════════════════════════════
   INPUT HANDLING
   ═════════════════════════════════════════════════════════════════════════ */

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !isExecuting) {
    const instruction = input.value.trim();
    if (instruction) executeInstruction(instruction);
  }
  // Escape inside input → hide overlay
  if (e.key === 'Escape') {
    hideOverlay();
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

  appendLogEntry({
    type:    'info',
    stage:   'agent_start',
    message: `→ ${instruction}`,
  });

  try {
    const response = await fetch(BACKEND_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ instruction }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} — ${response.statusText}`);
    }

    const data = await response.json();

    // ── Stream events with staggered animation ──────────────────── //
    if (Array.isArray(data.events) && data.events.length > 0) {
      await streamEvents(data.events);
    }

    // ── Final result ────────────────────────────────────────────── //
    if (data.success) {
      appendLogEntry({
        type:    'status',
        stage:   'done',
        message: data.output ?? 'Completed.',
        className: 'result-ok',
      });
      setStatus('success');
    } else {
      appendLogEntry({
        type:    'error',
        stage:   'failed',
        message: data.error ?? 'Unknown error.',
        className: 'result-err',
      });
      setStatus('error');
    }

  } catch (err) {
    // Network or JSON parse error
    const isConnectionRefused =
      err.message.includes('Failed to fetch') ||
      err.message.includes('ERR_CONNECTION_REFUSED') ||
      err.message.includes('NetworkError');

    appendLogEntry({
      type:    'error',
      stage:   'network_error',
      message: isConnectionRefused
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
    window.gladden.setIgnoreMouseEvents(false);

    // Reset status dot to idle after 3 seconds
    setTimeout(() => setStatus('idle'), 3000);
  }
}

/* ── Stream events with per-entry delay for visual effect ───────────────── */
async function streamEvents(events) {
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    appendLogEntry({
      type:    ev.type    ?? 'info',
      stage:   ev.stage   ?? '',
      message: ev.message ?? '',
      ts:      ev.timestamp,
    });
    // Stagger: 40ms per event (faster for many events)
    const delay = events.length > 20 ? 20 : 40;
    await sleep(delay);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOG PANEL
   ═════════════════════════════════════════════════════════════════════════ */

function clearLog() {
  // Keep a placeholder slot, replace content
  logInner.innerHTML = '';
}

/**
 * Append a single log entry row.
 * @param {object} opts
 * @param {string} opts.type      - 'info' | 'status' | 'error' | 'security' | 'warn'
 * @param {string} opts.stage     - stage name
 * @param {string} opts.message   - human-readable message
 * @param {string} [opts.ts]      - ISO timestamp (optional)
 * @param {string} [opts.className] - extra CSS class
 */
function appendLogEntry({ type = 'info', stage = '', message = '', ts, className = '' }) {
  const entry = document.createElement('div');
  entry.classList.add('log-entry');
  if (className) entry.classList.add(className);
  entry.setAttribute('data-type', type);

  const time = formatTs(ts);

  entry.innerHTML = `
    <span class="log-ts">${escapeHtml(time)}</span>
    <span class="log-stage">${escapeHtml(stage)}</span>
    <span class="log-msg">${escapeHtml(message)}</span>
  `;

  logInner.appendChild(entry);
  scrollLogToBottom();
}

function scrollLogToBottom() {
  requestAnimationFrame(() => {
    logInner.scrollTop = logInner.scrollHeight;
  });
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatTs(iso) {
  if (!iso) {
    // Use current time
    const now = new Date();
    return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  }
  try {
    const d = new Date(iso);
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  } catch {
    return '--:--:--';
  }
}

function pad(n) {
  return String(n).padStart(2, '0');
}