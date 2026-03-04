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

/**
 * main.js
 * Gladden AI Assistant — Electron Main Process
 */

'use strict';

const {
  app,
  BrowserWindow,
  globalShortcut,
  ipcMain,
  screen,
} = require('electron');

const path = require('path');

// ── Single-instance lock ────────────────────────────────────────────────── //
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) { app.quit(); process.exit(0); }

// ── State ───────────────────────────────────────────────────────────────── //
let mainWindow = null;
let isVisible  = true;

// ── Window factory ──────────────────────────────────────────────────────── //
function createWindow() {
  const { width: screenWidth } = screen.getPrimaryDisplay().workAreaSize;

  // Narrow centered overlay
  const winWidth = 700;
  const winX     = Math.round((screenWidth - winWidth) / 2);

  mainWindow = new BrowserWindow({
    width:  winWidth,
    height: 110,
    x:      winX,
    y:      0,

    frame:            false,
    transparent:      true,
    backgroundColor:  '#00000000',
    resizable:        false,
    movable:          false,
    minimizable:      false,
    maximizable:      false,
    fullscreenable:   false,
    skipTaskbar:      true,
    alwaysOnTop:      true,
    visibleOnAllWorkspaces: true,

    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
      sandbox:          true,
    },
  });

  mainWindow.loadFile('index.html');
  mainWindow.setAlwaysOnTop(true, 'screen-saver');

  // Start with click-through OFF so the bar is immediately clickable
  mainWindow.setIgnoreMouseEvents(false);

  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── IPC: mouse-capture toggle ───────────────────────────────────────────── //
ipcMain.on('set-ignore-mouse-events', (event, ignore, opts) => {
  if (mainWindow) {
    mainWindow.setIgnoreMouseEvents(ignore, opts || {});
  }
});

ipcMain.on('hide-window', () => {
  if (mainWindow) { mainWindow.hide(); isVisible = false; }
});

ipcMain.on('show-window', () => {
  if (mainWindow) { mainWindow.show(); isVisible = true; }
});

// ── App lifecycle ───────────────────────────────────────────────────────── //
app.whenReady().then(() => {
  createWindow();

  globalShortcut.register('Escape', () => {
    if (mainWindow && isVisible) {
      mainWindow.webContents.send('trigger-hide');
    }
  });

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
  if (mainWindow) {
    if (!mainWindow.isVisible()) mainWindow.show();
    mainWindow.focus();
  }
});

app.on('will-quit', () => { globalShortcut.unregisterAll(); });

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});