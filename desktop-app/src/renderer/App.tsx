import { useEffect } from 'react'
import { Sidebar } from '@/renderer/components/Sidebar'
import { ChatWindow } from '@/renderer/components/ChatWindow'
import { ChatProvider, useChat } from '@/renderer/context/ChatContext'

function AppContent() {
  const { state, newSession } = useChat()

  // Create initial session on mount
  useEffect(() => {
    if (state.sessions.length === 0) {
      newSession()
    }
  }, [])

  return (
    <div className="flex h-screen bg-slate-900 text-white overflow-hidden">
      <Sidebar />
      <ChatWindow />
    </div>
  )
}

function App() {
  useEffect(() => {
    // Ensure dark mode is applied
    document.documentElement.classList.add('dark')
  }, [])

  return (
    <ChatProvider>
      <AppContent />
    </ChatProvider>
  )
}

export default App
