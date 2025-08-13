const path = require('path')

// 1. Import the app (controls app lifecycle) and BrowserWindow (creates windows)
const { app, BrowserWindow } = require('electron')

// 2. A function to create our window
function createWindow() {
  // Create a new BrowserWindow object with size
  const win = new BrowserWindow({
    width: 800,
    height: 600,
  })

  // Load a local HTML file into the window
  win.loadFile(path.join(__dirname, 'index.html'))
}

// 3. Wait until Electron is ready, then create the window
app.whenReady().then(() => {
  createWindow()

  // On macOS, re-create a window when clicking the dock icon if none are open
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

// 4. Quit the app when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
