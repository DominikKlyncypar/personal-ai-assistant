const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')

// 1. Import the app (controls app lifecycle) and BrowserWindow (creates windows)
const { app, BrowserWindow, ipcMain, shell } = require('electron')
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

function startWorker() {
  // __dirname is .../app/src
  const repoRoot = path.join(__dirname, '..')          // .../app
  const workerDir = path.join(repoRoot, '..', 'worker')// .../worker

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
        ...(hasLocalModel ? { WORKER_WHISPER_MODEL_DIR: modelDir } : {}),
      },
      stdio: 'inherit',          // pipe logs to Electron terminal
    }
  )

  workerProc.on('exit', (code, signal) => {
    console.log(`[worker] exited with code=${code} signal=${signal}`)
  })
}

let win = null

// 2. A function to create our window
function createWindow() {
  // Create a new BrowserWindow object with size
  const win = new BrowserWindow({
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
    return { ok: false, error: String(e) }
  }
})

// 4. Wait until Electron is ready, then create the window
app.whenReady().then(() => {
  startWorker()
  createWindow()

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
    try { workerProc.kill() } catch (_) {}
  }
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
