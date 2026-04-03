import { useRef, useEffect, useState } from 'react'
import { useChat } from '@/renderer/context/ChatContext'
import { useElectronFile } from '@/renderer/hooks/useElectronFile'
import { useChat_API } from '@/renderer/hooks/useChat_API'
import { Message } from './Message'
import { FileQueue, FileUpload } from './FileUpload'
import { ModelSelector } from './ModelSelector'
import { AlertCircle, MessageSquare } from 'lucide-react'

export function ChatWindow() {
  const { currentSession, state, addMessage, setError } = useChat()
  const { files, openFileDialog, removeFile, filePaths, clearFiles } =
    useElectronFile()
  const { sendMessage, isLoading } = useChat_API()
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [localError, setLocalError] = useState<string | null>(null)

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentSession?.messages])

  if (!currentSession) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-gradient-to-br from-slate-900 to-slate-950">
        <MessageSquare size={64} className="text-slate-600 mb-4" />
        <h2 className="text-2xl font-bold text-slate-200 mb-2">No Conversation</h2>
        <p className="text-slate-400">Start a new chat to begin</p>
      </div>
    )
  }

  const handleSendMessage = async () => {
    if (!input.trim() && filePaths.length === 0) {
      setLocalError('Enter a message or select files')
      return
    }

    try {
      setLocalError(null)
      await sendMessage(input, state.selectedModel, filePaths)
      setInput('')
      clearFiles()
    } catch (err) {
      setLocalError(String(err))
    }
  }

  return (
    <div className="flex-1 flex flex-col bg-gradient-to-b from-slate-900 to-slate-950">
      {/* Header */}
      <div className="border-b border-slate-800 p-4 flex justify-between items-center bg-slate-900/50 backdrop-blur">
        <div>
          <h2 className="text-lg font-bold text-white">{currentSession.title}</h2>
          <p className="text-xs text-slate-400 mt-1">
            {currentSession.messages.length} messages • Model: {state.selectedModel}
          </p>
        </div>
        <ModelSelector />
      </div>

      {/* Error Messages */}
      {(state.error || localError) && (
        <div className="mx-4 mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 flex gap-3">
          <AlertCircle size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-200">{state.error || localError}</p>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {currentSession.messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <MessageSquare size={48} className="mx-auto text-slate-600 mb-3" />
              <p className="text-slate-400">Start the conversation</p>
              <p className="text-xs text-slate-500 mt-1">Ask anything or upload documents</p>
            </div>
          </div>
        ) : (
          <>
            {currentSession.messages.map(msg => (
              <Message
                key={msg.id}
                role={msg.role}
                content={msg.content}
                timestamp={msg.timestamp}
              />
            ))}
            {isLoading && (
              <div className="flex gap-3 py-4 px-1">
                <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                </div>
                <div className="flex-1">
                  <span className="text-sm text-slate-400">Assistant is thinking...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* File Queue */}
      {files.length > 0 && (
        <FileQueue files={files} onRemove={removeFile} onAddClick={openFileDialog} />
      )}

      {/* Input Area */}
      <div className="border-t border-slate-700 bg-slate-900/50 backdrop-blur">
        <div className="p-4 space-y-3">
          {/* File Upload Input */}
          <div className="relative">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
              placeholder="Type a message... (Shift+Enter for newline)"
              disabled={isLoading}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition disabled:opacity-50"
            />
          </div>

          {/* Send Button */}
          <div className="flex gap-2">
            <button
              onClick={openFileDialog}
              disabled={isLoading}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-medium transition-all disabled:opacity-50 text-sm"
            >
              + Add Files
            </button>
            <button
              onClick={handleSendMessage}
              disabled={isLoading || (!input.trim() && filePaths.length === 0)}
              className="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-slate-600 disabled:to-slate-700 text-white font-medium transition-all"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
