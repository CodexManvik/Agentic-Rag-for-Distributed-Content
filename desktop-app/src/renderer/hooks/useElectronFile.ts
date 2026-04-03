import { useCallback, useState } from 'react'
import type { FileMetadata } from '@/types/ipc'

export function useElectronFile() {
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const openFileDialog = useCallback(async () => {
    if (!window.electron) {
      setError('Electron API not available')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const result = await window.electron.openFileDialog({
        properties: ['openFile', 'multiSelections']
      })

      if (!result.canceled && result.filePaths.length > 0) {
        const metadata = await Promise.all(
          result.filePaths.map(path =>
            window.electron.readFileMetadata(path)
          )
        )
        setFiles(prev => [...prev, ...metadata])
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  const selectDirectory = useCallback(async () => {
    if (!window.electron) {
      setError('Electron API not available')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const result = await window.electron.selectDirectory()

      if (!result.canceled && result.filePath) {
        const metadata = await window.electron.readFileMetadata(
          result.filePath
        )
        setFiles(prev => [...prev, metadata])
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  const removeFile = useCallback((filePath: string) => {
    setFiles(prev => prev.filter(f => f.path !== filePath))
  }, [])

  const clearFiles = useCallback(() => {
    setFiles([])
  }, [])

  return {
    files,
    error,
    loading,
    openFileDialog,
    selectDirectory,
    removeFile,
    clearFiles,
    filePaths: files.map(f => f.path)
  }
}
