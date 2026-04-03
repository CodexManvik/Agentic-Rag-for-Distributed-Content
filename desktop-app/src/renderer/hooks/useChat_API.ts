import { useCallback, useState, useEffect } from 'react'
import { useChat } from '@/renderer/context/ChatContext'
import type { Message } from '@/renderer/context/ChatContext'

export function useChat_API() {
  const { addMessage, setLoading, setError, state } = useChat()
  const [models, setModels] = useState<string[]>([])

  // Fetch available models on mount
  useEffect(() => {
    let isMounted = true

    const fetchModels = async () => {
      try {
        if (window.electron) {
          const availableModels = await window.electron.getAvailableModels()
          console.log('useChat_API: Fetched models -', availableModels)
          if (isMounted) {
            setModels(availableModels)
          }
        }
      } catch (err) {
        console.error('Failed to fetch models:', err)
        if (isMounted) {
          setModels([]) // No fallback - user needs to know models aren't available
        }
      }
    }

    fetchModels()
    return () => {
      isMounted = false
    }
  }, [])

  const sendMessage = useCallback(
    async (
      userMessage: string,
      selectedModel: string,
      filePaths: string[] = []
    ) => {
      if (!userMessage.trim()) {
        setError('Message cannot be empty')
        return
      }

      // Check IPC availability early
      if (!window.electron) {
        const errorMsg = 'Unable to send message: IPC unavailable'
        setError(errorMsg)
        // Add error message to chat
        const errorMsg_obj: Message = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: errorMsg,
          timestamp: new Date()
        }
        addMessage(errorMsg_obj)
        return
      }

      try {
        setLoading(true)
        setError(null)

        // Add user message immediately
        const userMsg: Message = {
          id: crypto.randomUUID(),
          role: 'user',
          content: userMessage,
          timestamp: new Date(),
          files: filePaths.length > 0 ? filePaths : undefined
        }
        addMessage(userMsg)

        // Send to backend
        const response = await window.electron.sendChatMessage(
          userMessage,
          selectedModel,
          filePaths
        )

        // Add assistant response
        const assistantMsg: Message = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response,
          timestamp: new Date()
        }
        addMessage(assistantMsg)
      } catch (err) {
        setError(String(err))
        console.error('Chat error:', err)
      } finally {
        setLoading(false)
      }
    },
    [addMessage, setLoading, setError]
  )

  return {
    sendMessage,
    models,
    isLoading: state.isLoading
  }
}
