// Expose a tiny, safe API into the page.
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('api', {
  echo: (msg) => ipcRenderer.invoke('echo', msg)
})