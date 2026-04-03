import { app, BrowserWindow, ipcMain, dialog, Menu } from 'electron'
import path from 'path'
import { fileURLToPath, pathToFileURL } from 'url'
import isDev from 'electron-is-dev'
import fs from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

let mainWindow: BrowserWindow | null = null

const createWindow = () => {
  // Preload built from src/preload/preload.ts → dist-electron/preload/preload.js
  const preloadPath = path.join(__dirname, '../preload/preload.js')

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      preload: preloadPath,
      sandbox: true,
      enableRemoteModule: false,
      nodeIntegration: false,
      contextIsolation: true
    },
    icon: path.join(__dirname, '../../public/assets/icon.png')
  })

  const startUrl = isDev
    ? 'http://localhost:5173'
    : `file://${path.join(__dirname, '../../dist/index.html')}`

  mainWindow.loadURL(startUrl)

  if (isDev) {
    mainWindow.webContents.openDevTools()
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

// IPC Handlers for File Operations
ipcMain.handle('dialog:openFile', async (_event, options) => {
  if (!mainWindow) return { canceled: true, filePaths: [] }

  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Documents', extensions: ['pdf', 'txt', 'md', 'docx', 'doc'] },
      { name: 'PDF', extensions: ['pdf'] },
      { name: 'Text Files', extensions: ['txt', 'md', 'markdown'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    ...options
  })

  return { canceled, filePaths }
})

ipcMain.handle('dialog:selectDirectory', async (_event) => {
  if (!mainWindow) return { canceled: true }

  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  })

  return { canceled, filePath: filePaths[0] }
})

ipcMain.handle('file:readMetadata', async (_event, filePath: string) => {
  try {
    const stat = await fs.promises.stat(filePath)
    return {
      name: path.basename(filePath),
      size: stat.size,
      path: filePath,
      modified: stat.mtime
    }
  } catch (error) {
    throw new Error(`Failed to read file: ${error}`)
  }
})

// IPC Handlers for Chat/Model Operations
ipcMain.handle('chat:getAvailableModels', async () => {
  try {
    console.log('[IPC] chat:getAvailableModels called - fetching from backend...')
    // Fetch models dynamically from backend
    const response = await fetch('http://localhost:8000/models', {
      timeout: 5000
    })
    console.log(`[IPC] Backend /models response status: ${response.status}`)
    
    if (!response.ok) {
      const text = await response.text()
      console.error(`[IPC] Backend error response: ${text}`)
      throw new Error(`Backend error: ${response.status}`)
    }
    
    const data = await response.json()
    console.log('[IPC] Models fetched successfully:', data.models)
    return data.models || []
  } catch (error) {
    console.error('[IPC] Failed to fetch models from backend:', error)
    return [] // Return empty if backend unavailable
  }
})

ipcMain.handle('chat:sendMessage', async (
  _event,
  query: string,
  _model: string,
  _files: string[]
) => {
  try {
    // Backend expects: query field (not message), POST to /chat endpoint
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    })

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    return {
      answer: data.answer || 'No response',
      citations: data.citations || [],
      confidence: data.confidence || 0
    }
  } catch (error) {
    console.error('Chat error:', error)
    throw new Error(`Chat failed: ${error}`)
  }
})

// App event handlers
app.on('ready', createWindow)

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow()
  }
})

// Create context menu
const createMenu = () => {
  const template: any[] = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Exit',
          accelerator: 'CmdOrCtrl+Q',
          click: () => app.quit()
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { label: 'Undo', accelerator: 'CmdOrCtrl+Z', role: 'undo' },
        { label: 'Redo', accelerator: 'CmdOrCtrl+Y', role: 'redo' },
        { type: 'separator' },
        { label: 'Cut', accelerator: 'CmdOrCtrl+X', role: 'cut' },
        { label: 'Copy', accelerator: 'CmdOrCtrl+C', role: 'copy' },
        { label: 'Paste', accelerator: 'CmdOrCtrl+V', role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { label: 'Reload', accelerator: 'CmdOrCtrl+R', role: 'reload' },
        {
          label: 'Toggle Developer Tools',
          accelerator: 'CmdOrCtrl+Shift+I',
          role: 'toggleDevTools'
        }
      ]
    }
  ]

  const menu = Menu.buildFromTemplate(template)
  Menu.setApplicationMenu(menu)
}

app.whenReady().then(createMenu)
