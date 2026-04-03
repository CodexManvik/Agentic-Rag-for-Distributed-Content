import { ipcRenderer } from 'electron'

export interface FileMetadata {
  name: string
  size: number
  path: string
  modified: Date
}

export interface IpcApi {
  openFileDialog: (options?: any) => Promise<{ canceled: boolean; filePaths: string[] }>
  readFileMetadata: (path: string) => Promise<FileMetadata>
  selectDirectory: () => Promise<{ canceled: boolean; filePath?: string }>
  getAvailableModels: () => Promise<string[]>
  sendChatMessage: (message: string, model: string, files: string[]) => Promise<string>
}

declare global {
  interface Window {
    electron: IpcApi
  }
}
