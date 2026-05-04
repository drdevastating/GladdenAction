'use strict';

const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const path = require('path');

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) { app.quit(); process.exit(0); }

let mainWindow  = null;
let hazeWindow  = null;
let isVisible   = true;

/* ── Haze overlay (full-screen, click-through, always-on-top) ─────────── */

function createHazeWindow() {
  const { width, height } = screen.getPrimaryDisplay().bounds;

  hazeWindow = new BrowserWindow({
    width,
    height,
    x: 0,
    y: 0,
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
    focusable:       false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  hazeWindow.setIgnoreMouseEvents(true, { forward: true });
  hazeWindow.setAlwaysOnTop(true, 'screen-saver', 1);
  hazeWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(`
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    width: 100vw; height: 100vh;
    background: transparent;
    overflow: hidden;
    pointer-events: none;
  }
  .haze {
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.6s cubic-bezier(0.16,1,0.3,1);
  }
  .haze.active { opacity: 1; }

  /* Four edge glows */
  .haze::before, .haze::after,
  .haze .edge-b, .haze .edge-l {
    content: '';
    position: absolute;
    pointer-events: none;
  }
  /* Top */
  .haze::before {
    top: 0; left: 0; right: 0;
    height: 90px;
    background: linear-gradient(180deg,
      rgba(56,217,245,0.55) 0%,
      rgba(56,217,245,0.30) 30%,
      rgba(56,217,245,0.08) 70%,
      transparent 100%
    );
    filter: blur(2px);
  }
  /* Bottom */
  .haze::after {
    bottom: 0; left: 0; right: 0;
    height: 90px;
    background: linear-gradient(0deg,
      rgba(56,217,245,0.55) 0%,
      rgba(56,217,245,0.30) 30%,
      rgba(56,217,245,0.08) 70%,
      transparent 100%
    );
    filter: blur(2px);
  }
  /* Left */
  .edge-l {
    top: 0; bottom: 0; left: 0;
    width: 90px;
    background: linear-gradient(90deg,
      rgba(56,217,245,0.55) 0%,
      rgba(56,217,245,0.30) 30%,
      rgba(56,217,245,0.08) 70%,
      transparent 100%
    );
    filter: blur(2px);
  }
  /* Right */
  .edge-r {
    top: 0; bottom: 0; right: 0;
    width: 90px;
    background: linear-gradient(270deg,
      rgba(56,217,245,0.55) 0%,
      rgba(56,217,245,0.30) 30%,
      rgba(56,217,245,0.08) 70%,
      transparent 100%
    );
    filter: blur(2px);
  }
  /* Corner glow blobs */
  .corner {
    position: absolute;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56,217,245,0.35) 0%, transparent 70%);
    filter: blur(8px);
    pointer-events: none;
  }
  .corner.tl { top: -60px;  left: -60px; }
  .corner.tr { top: -60px;  right: -60px; }
  .corner.bl { bottom: -60px; left: -60px; }
  .corner.br { bottom: -60px; right: -60px; }

  /* Subtle animated pulse */
  @keyframes hazePulse {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.72; }
  }
  .haze.active { animation: hazePulse 3s ease-in-out infinite; }
</style>
</head>
<body>
<div class="haze" id="haze">
  <div class="edge-l"></div>
  <div class="edge-r"></div>
  <div class="corner tl"></div>
  <div class="corner tr"></div>
  <div class="corner bl"></div>
  <div class="corner br"></div>
</div>
<script>
  // Listen for toggle messages via title hack (simplest IPC for data: URL)
  // We use a custom event via BrowserWindow.webContents.executeJavaScript instead
  window._hazeActive = false;
  window.setHaze = function(active) {
    window._hazeActive = active;
    document.getElementById('haze').classList.toggle('active', active);
  };
</script>
</body>
</html>
  `));

  hazeWindow.on('closed', () => { hazeWindow = null; });
}

function setHaze(active) {
  if (!hazeWindow) return;
  hazeWindow.webContents.executeJavaScript(`window.setHaze(${active})`).catch(() => {});
}

/* ── Main command bar window ────────────────────────────────────────────── */

function createWindow() {
  const { width: screenWidth } = screen.getPrimaryDisplay().workAreaSize;

  const winWidth = 700;
  const winX     = Math.round((screenWidth - winWidth) / 2);

  mainWindow = new BrowserWindow({
    width:  winWidth,
    height: 72,
    x:      winX,
    y:      6,

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
  mainWindow.setAlwaysOnTop(true, 'screen-saver', 2);
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
  setHaze(false);
});

ipcMain.on('show-window', () => {
  if (mainWindow) { mainWindow.show(); isVisible = true; }
  setHaze(true);
});

ipcMain.on('resize-window', (event, height) => {
  if (!mainWindow) return;
  const { width: screenWidth } = screen.getPrimaryDisplay().workAreaSize;
  const winWidth = 700;
  const winX     = Math.round((screenWidth - winWidth) / 2);
  const h = Math.max(72, Math.min(height, 600));
  mainWindow.setBounds({ x: winX, y: 6, width: winWidth, height: h }, true);
});

ipcMain.on('set-haze', (event, active) => {
  setHaze(active);
});

/* ── App lifecycle ──────────────────────────────────────────────────────── */

app.whenReady().then(() => {
  createHazeWindow();
  createWindow();

  // Show haze when visible
  setHaze(true);

  globalShortcut.register('Escape', () => {
    if (mainWindow && isVisible) mainWindow.webContents.send('trigger-hide');
  });

  globalShortcut.register('CommandOrControl+Space', () => {
    if (!mainWindow) return;
    if (isVisible) {
      mainWindow.webContents.send('trigger-hide');
      setHaze(false);
    } else {
      mainWindow.show();
      isVisible = true;
      mainWindow.webContents.send('trigger-show');
      mainWindow.focus();
      setHaze(true);
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