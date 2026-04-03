import { Plus, MessageSquare, Trash2, ChevronDown } from 'lucide-react'
import { useChat } from '@/renderer/context/ChatContext'
import { useState } from 'react'

export function Sidebar() {
  const { state, newSession, selectSession, deleteSession } = useChat()
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const sortedSessions = [...state.sessions].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  )

  return (
    <div className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col h-screen">
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <button
          onClick={() => newSession()}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium transition-all shadow-lg hover:shadow-blue-500/20"
        >
          <Plus size={18} />
          <span>New Chat</span>
        </button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {sortedSessions.length === 0 ? (
          <div className="text-center py-8">
            <MessageSquare size={32} className="mx-auto text-slate-600 mb-2" />
            <p className="text-sm text-slate-500">No chats yet</p>
            <p className="text-xs text-slate-600 mt-1">Start a new conversation</p>
          </div>
        ) : (
          sortedSessions.map(session => (
            <div
              key={session.id}
              onMouseEnter={() => setHoveredId(session.id)}
              onMouseLeave={() => setHoveredId(null)}
              className={`group relative rounded-lg transition-all ${
                state.currentSessionId === session.id
                  ? 'bg-slate-800 shadow-lg'
                  : 'hover:bg-slate-800/50'
              }`}
            >
              <button
                onClick={() => selectSession(session.id)}
                className="w-full text-left px-4 py-3 rounded-lg"
              >
                <div className="flex items-start gap-2">
                  <MessageSquare
                    size={16}
                    className={`flex-shrink-0 mt-0.5 ${
                      state.currentSessionId === session.id
                        ? 'text-blue-400'
                        : 'text-slate-500'
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm font-medium truncate ${
                        state.currentSessionId === session.id
                          ? 'text-white'
                          : 'text-slate-300'
                      }`}
                    >
                      {session.title}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {session.messages.length} messages
                    </p>
                  </div>
                </div>
              </button>

              {/* Delete button on hover */}
              {hoveredId === session.id && (
                <button
                  onClick={e => {
                    e.stopPropagation()
                    deleteSession(session.id)
                  }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition"
                  title="Delete chat"
                >
                  <Trash2 size={16} />
                </button>
              )}
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-slate-800 text-center">
        <p className="text-xs text-slate-500">Aura v1.0</p>
      </div>
    </div>
  )
}
