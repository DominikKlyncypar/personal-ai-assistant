const { spawn } = require('child_process')
const path = require('path')

// 1. Import the app (controls app lifecycle) and BrowserWindow (creates windows)
const { app, BrowserWindow, ipcMain} = require('electron')
let workerProc = null

function startWorker() {
  // __dirname is .../app/src
  const repoRoot = path.join(__dirname, '..')          // .../app
  const workerDir = path.join(repoRoot, '..', 'worker')// .../worker

  // Mac/Linux venv python:
  const pythonPath = path.join(workerDir, '.venv', 'bin', 'python')

  // Command: python -m uvicorn src.main:app --port 8000
  workerProc = spawn(
    pythonPath,
    ['-m', 'uvicorn', 'src.main:app', '--port', '8000'],
    {
      cwd: workerDir,
      env: { ...process.env },   // inherit your env
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
