import { contextBridge, ipcRenderer } from 'electron'
import type { IpcApi, FileMetadata } from '../types/ipc'

const electronApi: IpcApi = {
  openFileDialog: (options?: any) =>
    ipcRenderer.invoke('dialog:openFile', options),

  selectDirectory: () =>
    ipcRenderer.invoke('dialog:selectDirectory'),

  readFileMetadata: (filePath: string) =>
    ipcRenderer.invoke('file:readMetadata', filePath),

  getAvailableModels: () =>
    ipcRenderer.invoke('chat:getAvailableModels'),

  sendChatMessage: (message: string, model: string, files: string[]) =>
    ipcRenderer.invoke('chat:sendMessage', message, model, files)
}

contextBridge.exposeInMainWorld('electron', electronApi)

export type { IpcApi, FileMetadata }
