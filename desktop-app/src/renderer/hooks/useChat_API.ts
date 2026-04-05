import { useCallback, useState, useEffect } from 'react'
import { useChat } from '@/renderer/context/ChatContext'
import type { Message } from '@/renderer/context/ChatContext'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

export function useChat_API() {
  const { addMessage, setLoading, setError, state } = useChat()
  const [models, setModels] = useState<string[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [streamingTrace, setStreamingTrace] = useState<Array<Record<string, unknown>>>([])
  const [streamingCitations, setStreamingCitations] = useState<Array<Record<string, unknown>>>([])

  // Fetch available models on mount
  useEffect(() => {
    let isMounted = true

    const fetchModels = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/models`)
        if (!response.ok) {
          throw new Error(`Failed to load models (${response.status})`)
        }

        const data = await response.json()
        const availableModels = Array.isArray(data.models) ? data.models : []
        if (isMounted) {
          setModels(availableModels)
        }
      } catch (err) {
        console.error('Failed to fetch models:', err)
        if (isMounted) {
          setModels([])
          setError('Ollama not running or model API unavailable')
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
      if (!selectedModel || (models.length > 0 && !models.includes(selectedModel))) {
        const errorMsg = `Selected model is unavailable: ${selectedModel}`
        setError(errorMsg)
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
        setStreamingText('')
        setStreamingTrace([])
        setStreamingCitations([])

        // Add user message immediately
        const userMsg: Message = {
          id: crypto.randomUUID(),
          role: 'user',
          content: userMessage,
          timestamp: new Date(),
          files: filePaths.length > 0 ? filePaths : undefined
        }
        addMessage(userMsg)

        await new Promise<void>((resolve, reject) => {
          const streamUrl = `${BACKEND_URL}/chat/stream?query=${encodeURIComponent(userMessage)}`
          const eventSource = new EventSource(streamUrl)

          let collectedText = ''
          let collectedTrace: Array<Record<string, unknown>> = []
          let collectedCitations: Array<Record<string, unknown>> = []

          eventSource.addEventListener('trace', event => {
            try {
              const traceEvent = JSON.parse(event.data) as Record<string, unknown>
              collectedTrace = [...collectedTrace, traceEvent]
              setStreamingTrace(collectedTrace)
            } catch (parseError) {
              console.error('Failed to parse trace event:', parseError)
            }
          })

          eventSource.addEventListener('chunk', event => {
            try {
              const chunkData = JSON.parse(event.data) as { text?: string }
              const token = chunkData.text || ''
              collectedText += token
              setStreamingText(collectedText)
            } catch (parseError) {
              console.error('Failed to parse chunk event:', parseError)
            }
          })

          eventSource.addEventListener('complete', event => {
            try {
              const payload = JSON.parse(event.data) as {
                answer?: string
                citations?: Array<Record<string, unknown>>
                trace?: Array<Record<string, unknown>>
              }
              const finalAnswer = payload.answer || collectedText
              collectedCitations = Array.isArray(payload.citations) ? payload.citations : []
              collectedTrace = Array.isArray(payload.trace) ? payload.trace : collectedTrace

              addMessage({
                id: crypto.randomUUID(),
                role: 'assistant',
                content: finalAnswer,
                timestamp: new Date(),
                citations: collectedCitations,
                trace: collectedTrace,
              })

              setStreamingCitations(collectedCitations)
              setStreamingTrace(collectedTrace)
              setStreamingText('')

              eventSource.close()
              resolve()
            } catch (parseError) {
              eventSource.close()
              reject(new Error('Failed to parse complete event'))
            }
          })

          eventSource.addEventListener('error', event => {
            try {
              const maybeMessage = (event as MessageEvent).data
              if (typeof maybeMessage === 'string' && maybeMessage.length > 0) {
                const payload = JSON.parse(maybeMessage) as { error?: string }
                reject(new Error(payload.error || 'Stream error'))
              } else {
                reject(new Error('Connection lost'))
              }
            } catch {
              reject(new Error('Connection lost'))
            } finally {
              eventSource.close()
            }
          })
        })
      } catch (err) {
        setError(String(err))
        console.error('Chat error:', err)
      } finally {
        setLoading(false)
      }
    },
    [addMessage, setLoading, setError, models]
  )

  return {
    sendMessage,
    models,
    isLoading: state.isLoading,
    streamingText,
    streamingTrace,
    streamingCitations,
  }
}
