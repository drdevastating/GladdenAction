'use strict';

const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const path = require('path');

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) { app.quit(); process.exit(0); }

let mainWindow = null;
let isVisible  = true;

function createWindow() {
  const { width: screenWidth } = screen.getPrimaryDisplay().workAreaSize;

  const winWidth = 700;
  const winX     = Math.round((screenWidth - winWidth) / 2);

  mainWindow = new BrowserWindow({
    width:  winWidth,
    height: 72,   // bar (56) + padding (16) — renderer will resize as needed
    x:      winX,
    y:      0,

    frame:           false,
    transparent:     true,
    backgroundColor: '#00000000',
    resizable:       false,
    movable:         false,
    minimizable:     false,
    maximizable:     false,
    fullscreenable:  false,
    skipTaskbar:     true,
    alwaysOnTop:     true,
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
  mainWindow.setIgnoreMouseEvents(false);

  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }

  mainWindow.on('closed', () => { mainWindow = null; });
}

/* ── IPC handlers ───────────────────────────────────────────────────────── */

ipcMain.on('set-ignore-mouse-events', (event, ignore, opts) => {
  if (mainWindow) mainWindow.setIgnoreMouseEvents(ignore, opts || {});
});

ipcMain.on('hide-window', () => {
  if (mainWindow) { mainWindow.hide(); isVisible = false; }
});

ipcMain.on('show-window', () => {
  if (mainWindow) { mainWindow.show(); isVisible = true; }
});

ipcMain.on('resize-window', (event, height) => {
  if (!mainWindow) return;
  const { width: screenWidth } = screen.getPrimaryDisplay().workAreaSize;
  const winWidth = 700;
  const winX     = Math.round((screenWidth - winWidth) / 2);
  // Clamp height to something sensible
  const h = Math.max(72, Math.min(height, 600));
  mainWindow.setBounds({ x: winX, y: 0, width: winWidth, height: h }, true); // true = animate
});

/* ── App lifecycle ──────────────────────────────────────────────────────── */

app.whenReady().then(() => {
  createWindow();

  globalShortcut.register('Escape', () => {
    if (mainWindow && isVisible) mainWindow.webContents.send('trigger-hide');
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
  if (mainWindow) { if (!mainWindow.isVisible()) mainWindow.show(); mainWindow.focus(); }
});

app.on('will-quit', () => { globalShortcut.unregisterAll(); });
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });