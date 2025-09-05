// Expose a tiny, safe API into the page.
const { contextBridge, ipcRenderer } = require('electron')

function assertString(name, v) {
  if (typeof v !== 'string') throw new TypeError(`${name} must be a string`)
}

contextBridge.exposeInMainWorld('api', {
  echo: (msg) => { assertString('msg', msg); return ipcRenderer.invoke('echo', msg) },
  revealPath: (absPath) => { assertString('absPath', absPath); return ipcRenderer.invoke('reveal-path', absPath) },
})
