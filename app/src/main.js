const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')

// 1. Import the app (controls app lifecycle) and BrowserWindow (creates windows)
const { app, BrowserWindow, ipcMain, shell } = require('electron')
const { autoUpdater } = require('electron-updater')
const log = require('electron-log')

log.transports.file.level = 'info'
log.transports.file.maxSize = 10 * 1024 * 1024
try {
  const logFile = log.transports.file.getFile()
  log.info('[app] logging to', logFile ? logFile.path : 'unknown path')
} catch (err) {
  log.warn('[app] unable to determine log file path', err)
}
let workerProc = null

function resolvePython(workerDir) {
  // Allow override via env
  if (process.env.WORKER_PYTHON && fs.existsSync(process.env.WORKER_PYTHON)) {
    return { cmd: process.env.WORKER_PYTHON, args: [] }
  }
  const isWin = process.platform === 'win32'
  const venvPath = isWin
    ? path.join(workerDir, '.venv', 'Scripts', 'python.exe')
    : path.join(workerDir, '.venv', 'bin', 'python')
  if (fs.existsSync(venvPath)) return { cmd: venvPath, args: [] }
  // Fallbacks: try system python
  if (isWin) return { cmd: 'py', args: ['-3'] } // Windows launcher
  return { cmd: 'python3', args: [] }
}

function getWorkerDir() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'worker')
  }
  return path.join(__dirname, '..', '..', 'worker')
}

function startWorker() {
  const workerDir = getWorkerDir()

  if (!fs.existsSync(workerDir)) {
    log.error(`[worker] directory not found at ${workerDir}`)
    return
  }
  log.info(`[worker] starting from ${workerDir}`)
  const py = resolvePython(workerDir)
  const modelDir = path.join(workerDir, 'models', 'faster-whisper-base')
  const modelBin = path.join(modelDir, 'model.bin')
  const modelCfg = path.join(modelDir, 'config.json')
  const hasLocalModel = fs.existsSync(modelBin) && fs.existsSync(modelCfg)

  // Command: python -m uvicorn src.app:app --port 8000
  workerProc = spawn(
    py.cmd,
    [...py.args, '-m', 'uvicorn', 'src.app:app', '--port', '8000'],
    {
      cwd: workerDir,
      env: {
        ...process.env,
        // Only point to local model if the required files are present
        ...(hasLocalModel ? { WORKER_WHISPER_MODEL_DIR: modelDir } : {})
      },
      stdio: 'inherit' // pipe logs to Electron terminal
    }
  )

  workerProc.on('exit', (code, signal) => {
    log.error(`[worker] exited with code=${code} signal=${signal}`)
  })
}

let win = null

// 2. A function to create our window
function createWindow() {
  // Create a new BrowserWindow object with size
  win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, 'preload.js') // Load the preload script
    }
  })

  // Load a local HTML file into the window
  win.loadFile(path.join(__dirname, 'index.html'))
  win.webContents.once('did-finish-load', () => {
    let logFilePath = null
    try {
      const file = log.transports.file.getFile()
      logFilePath = file ? file.path : null
    } catch (err) {
      log.warn('[app] failed to fetch log file info', err)
    }
    win.webContents.send('app-info', {
      version: app.getVersion(),
      logPath: logFilePath
    })
  })
  return win
}

function setupAutoUpdater() {
  if (!app.isPackaged) {
    if (process.env.DEBUG_AUTO_UPDATE) {
      log.info('[auto-updater] skipped (development build)')
    }
    return
  }

  autoUpdater.logger = log
  autoUpdater.autoDownload = true
  autoUpdater.autoInstallOnAppQuit = true

  autoUpdater.on('update-available', info => {
    log.info('[auto-updater] Update available:', info?.version ?? 'unknown version')
  })
  autoUpdater.on('update-not-available', () => {
    log.info('[auto-updater] No update available')
  })
  autoUpdater.on('update-downloaded', info => {
    log.info('[auto-updater] Update downloaded:', info?.version ?? 'unknown version', '— installing on quit')
    try {
      autoUpdater.quitAndInstall(false, true)
    } catch (err) {
      log.error('[auto-updater] quitAndInstall failed', err)
    }
  })
  autoUpdater.on('error', err => {
    log.error('[auto-updater] Auto-update failed', err)
  })

  const checkForUpdates = () => {
    autoUpdater.checkForUpdatesAndNotify().catch(err => {
      log.error('[auto-updater] checkForUpdates failed', err)
    })
  }

  checkForUpdates()
  const interval = Number(process.env.AUTO_UPDATE_INTERVAL_MS ?? 60 * 60 * 1000)
  if (interval > 0) {
    setInterval(checkForUpdates, interval)
  }
}

// 3. Handle IPC messages from the renderer process
ipcMain.handle('echo', async (_event, msg) => {
  try {
    const res = await fetch('http://127.0.0.1:8000/echo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    })
    const data = await res.json()
    return data
  } catch (err) {
    log.error('[ipc] echo failed', err)
    return { error: err.message }
  }
})

ipcMain.handle('reveal-path', async (_evt, absPath) => {
  if (!absPath || typeof absPath !== 'string') return { ok: false, error: 'bad path' }
  try {
    // Show the file in Finder (macOS) / Explorer (Windows)
    shell.showItemInFolder(absPath)
    // Alternatively: await shell.openPath(absPath)  // opens the file directly
    return { ok: true }
  } catch (e) {
    log.error('[ipc] reveal-path failed', e)
    return { ok: false, error: String(e) }
  }
})

// 4. Wait until Electron is ready, then create the window
app.whenReady().then(() => {
  log.info(`[app] starting version ${app.getVersion()}`)
  startWorker()
  win = createWindow()
  setupAutoUpdater()

  // On macOS, re-create a window when clicking the dock icon if none are open
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

// 5. Quit the app when all windows are closed (except on macOS)
app.on('before-quit', () => {
  if (workerProc && !workerProc.killed) {
    try {
      workerProc.kill()
    } catch (err) {
      log.warn('[worker] failed to kill worker on quit', err)
    }
  }
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
