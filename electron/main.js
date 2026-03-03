/**
 * main.js
 * Gladden AI Assistant — Electron Main Process
 *
 * Responsibilities:
 *  - Create a frameless, transparent, always-on-top overlay window
 *  - Register global hotkeys (Escape to hide, Ctrl+Space to toggle)
 *  - Prevent multiple instances via single-instance lock
 *  - Enable click-through on transparent areas
 *  - Handle IPC for backend communication proxy (optional)
 */

'use strict';

const {
  app,
  BrowserWindow,
  globalShortcut,
  ipcMain,
  screen,
  shell,
} = require('electron');

const path = require('path');

// ── Single-instance lock ────────────────────────────────────────────────── //
const gotLock = app.requestSingleInstanceLock();

if (!gotLock) {
  app.quit();
  process.exit(0);
}

// ── State ───────────────────────────────────────────────────────────────── //
let mainWindow = null;
let isVisible  = true;

// ── Window factory ──────────────────────────────────────────────────────── //
function createWindow() {
  const { width: screenWidth } = screen.getPrimaryDisplay().workAreaSize;

  mainWindow = new BrowserWindow({
    // ── Geometry ───────────────────────────────────────────────────── //
    width:  screenWidth,
    height: 110,
    x:      0,
    y:      0,

    // ── Chrome ─────────────────────────────────────────────────────── //
    frame:            false,
    transparent:      true,
    backgroundColor:  '#00000000',
    resizable:        false,
    movable:          false,
    minimizable:      false,
    maximizable:      false,
    fullscreenable:   false,
    skipTaskbar:      true,

    // ── Always on top — "screen-saver" level keeps it above fullscreen //
    alwaysOnTop:      true,
    visibleOnAllWorkspaces: true,

    // ── Security ───────────────────────────────────────────────────── //
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
      sandbox:          true,
    },
  });

  // Render the UI
  mainWindow.loadFile('index.html');

  // Set the window to be on the very top level
  mainWindow.setAlwaysOnTop(true, 'screen-saver');

  // ── Click-through on transparent areas ─────────────────────────── //
  // The renderer will send mouse-region updates; default: click-through
  mainWindow.setIgnoreMouseEvents(true, { forward: true });

  // ── Open DevTools in dev mode ───────────────────────────────────── //
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ── IPC: toggle mouse-capture region ───────────────────────────────────── //
// Renderer calls ipcRenderer.send('set-ignore-mouse-events', bool)
ipcMain.on('set-ignore-mouse-events', (event, ignore) => {
  if (mainWindow) {
    mainWindow.setIgnoreMouseEvents(ignore, { forward: true });
  }
});

// IPC: hide window
ipcMain.on('hide-window', () => {
  if (mainWindow) {
    mainWindow.hide();
    isVisible = false;
  }
});

// IPC: show window
ipcMain.on('show-window', () => {
  if (mainWindow) {
    mainWindow.show();
    isVisible = true;
  }
});

// ── App lifecycle ───────────────────────────────────────────────────────── //
app.whenReady().then(() => {
  createWindow();

  // ── Global shortcuts ──────────────────────────────────────────── //
  // Escape → hide overlay
  globalShortcut.register('Escape', () => {
    if (mainWindow && isVisible) {
      mainWindow.webContents.send('trigger-hide');
    }
  });

  // Ctrl+Space → toggle visibility
  globalShortcut.register('CommandOrControl+Space', () => {
    if (!mainWindow) return;
    if (isVisible) {
      mainWindow.webContents.send('trigger-hide');
    } else {
      mainWindow.show();
      isVisible = true;
      mainWindow.webContents.send('trigger-show');
      mainWindow.focus();
    }
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('second-instance', () => {
  // Focus existing window if user tries to open a second instance
  if (mainWindow) {
    if (!mainWindow.isVisible()) mainWindow.show();
    mainWindow.focus();
  }
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

// Keep the app alive even if all windows are closed (macOS behaviour)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});