'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gladden', {
  setIgnoreMouseEvents: (ignore, opts) =>
    ipcRenderer.send('set-ignore-mouse-events', ignore, opts),

  hideWindow: () =>
    ipcRenderer.send('hide-window'),

  showWindow: () =>
    ipcRenderer.send('show-window'),

  resizeWindow: (height) =>
    ipcRenderer.send('resize-window', height),

  onTriggerHide: (cb) =>
    ipcRenderer.on('trigger-hide', () => cb()),

  onTriggerShow: (cb) =>
    ipcRenderer.on('trigger-show', () => cb()),
});