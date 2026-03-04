/**
 * preload.js
 * Gladden AI Assistant — Preload / IPC Bridge
 *
 * Exposes a minimal, typed API to the renderer via contextBridge.
 * No raw Node APIs are exposed — full contextIsolation is preserved.
 *
 * Exposed on window.gladden:
 *   setIgnoreMouseEvents(bool)   → forward to main process
 *   hideWindow()                 → hide the overlay
 *   showWindow()                 → show the overlay
 *   onTriggerHide(callback)      → listen for hide signal from main
 *   onTriggerShow(callback)      → listen for show signal from main
 */

'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gladden', {
  // ── Mouse event passthrough control ──────────────────────────────── //
  setIgnoreMouseEvents: (ignore) => {
    ipcRenderer.send('set-ignore-mouse-events', ignore);
  },

  // ── Window visibility ─────────────────────────────────────────────── //
  hideWindow: () => {
    ipcRenderer.send('hide-window');
  },

  showWindow: () => {
    ipcRenderer.send('show-window');
  },

  // ── Listen for main-process signals ──────────────────────────────── //
  onTriggerHide: (callback) => {
    ipcRenderer.on('trigger-hide', () => callback());
  },

  onTriggerShow: (callback) => {
    ipcRenderer.on('trigger-show', () => callback());
  },
});