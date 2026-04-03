import { Plus, X, FileIcon } from 'lucide-react'
import { useRef, useState } from 'react'
import type { FileMetadata } from '@/types/ipc'

interface FileQueueProps {
  files: FileMetadata[]
  onRemove: (path: string) => void
  onAddClick: () => void
  loading?: boolean
}

export function FileQueue({ files, onRemove, onAddClick, loading }: FileQueueProps) {
  if (files.length === 0) return null

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="border-t border-slate-700 bg-slate-800/50 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">
          Files ({files.length})
        </h3>
        {files.length > 0 && (
          <button
            onClick={() => files.forEach(f => onRemove(f.path))}
            className="text-xs px-2 py-1 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-700 transition"
          >
            Clear All
          </button>
        )}
      </div>

      <div className="space-y-2 max-h-32 overflow-y-auto">
        {files.map(file => (
          <div
            key={file.path}
            className="flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-700/50 group hover:bg-slate-700 transition"
          >
            <FileIcon size={16} className="text-blue-400 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-slate-100 truncate">{file.name}</p>
              <p className="text-xs text-slate-400">{formatFileSize(file.size)}</p>
            </div>
            <button
              onClick={() => onRemove(file.path)}
              className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition"
            >
              <X size={16} />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

interface FileUploadProps {
  onFilesSelected: (files: FileMetadata[]) => void
  disabled?: boolean
}

export function FileUpload({ onFilesSelected, disabled }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const dragCounterRef = useRef(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounterRef.current++
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounterRef.current--
    if (dragCounterRef.current === 0) {
      setIsDragging(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    dragCounterRef.current = 0

    const droppedFiles = Array.from(e.dataTransfer.files)
    const validFiles = droppedFiles.filter(f =>
      ['application/pdf', 'text/plain', 'text/markdown', 'application/msword'].includes(f.type) ||
      f.name.match(/\.(pdf|txt|md|markdown|doc|docx)$/i)
    )

    if (validFiles.length > 0) {
      // Convert File to FileMetadata format
      const metadata = validFiles.map(file => ({
        name: file.name,
        size: file.size,
        path: file.webkitRelativePath || file.name,
        modified: new Date(file.lastModified)
      }))
      onFilesSelected(metadata)
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || [])
    const metadata = selectedFiles.map(file => ({
      name: file.name,
      size: file.size,
      path: file.webkitRelativePath || file.name,
      modified: new Date(file.lastModified)
    }))
    onFilesSelected(metadata)
    e.target.value = ''
  }

  return (
    <div
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      className={`transition-all rounded-t-lg border-t-2 border-x-2 border-dashed px-4 py-3 ${
        isDragging
          ? 'border-blue-500 bg-blue-500/10'
          : 'border-slate-600 hover:border-slate-500'
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        onChange={handleFileInput}
        className="hidden"
        accept=".pdf,.txt,.md,.markdown,.doc,.docx"
        disabled={disabled}
      />

      <div className="flex items-center gap-3">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-slate-600 disabled:to-slate-700 text-white font-medium transition-all flex-shrink-0"
        >
          <Plus size={18} />
          <span className="text-sm">Add Files</span>
        </button>

        <input
          type="text"
          placeholder="Type message... or drag files here"
          className="flex-1 bg-transparent outline-none text-white placeholder-slate-500 text-sm"
          disabled={disabled}
        />

        <button
          disabled={disabled}
          className="flex items-center justify-center p-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white transition-all flex-shrink-0"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8m0 8l-4-2m4 2l4-2"
            />
          </svg>
        </button>
      </div>
    </div>
  )
}
