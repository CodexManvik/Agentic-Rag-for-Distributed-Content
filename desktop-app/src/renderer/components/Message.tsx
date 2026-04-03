import { Copy, Trash2, MessageCircle, User } from 'lucide-react'
import { useState } from 'react'

interface MessageProps {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export function Message({ role, content, timestamp }: MessageProps) {
  const [copied, setCopied] = useState(false)
  const isBotMessage = role === 'assistant'

  return (
    <div className={`flex gap-4 py-4 px-1 ${isBotMessage ? 'bg-slate-800/40' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold ${
          isBotMessage ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-100'
        }`}
      >
        {isBotMessage ? (
          <MessageCircle size={16} />
        ) : (
          <User size={16} />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <span className="font-semibold text-sm text-slate-200">
            {isBotMessage ? 'Assistant' : 'You'}
          </span>
          <span className="text-xs text-slate-500">
            {timestamp.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit'
            })}
          </span>
        </div>

        <p className="text-slate-100 leading-relaxed whitespace-pre-wrap break-words text-sm">
          {content}
        </p>

        {/* Actions for assistant messages */}
        {isBotMessage && (
          <div className="flex gap-2 mt-3">
            <button
              onClick={() => {
                navigator.clipboard.writeText(content)
                setCopied(true)
                setTimeout(() => setCopied(false), 2000)
              }}
              className="flex items-center gap-1 px-2 py-1 text-xs rounded hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition"
            >
              <Copy size={14} />
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
