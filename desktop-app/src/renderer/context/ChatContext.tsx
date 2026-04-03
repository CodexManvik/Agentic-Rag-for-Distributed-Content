import React, { createContext, useContext, useReducer, ReactNode, useCallback } from 'react'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  files?: string[]
}

export interface ChatSession {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  model: string
}

export interface ChatState {
  sessions: ChatSession[]
  currentSessionId: string | null
  selectedModel: string
  isLoading: boolean
  error: string | null
}

export type ChatAction =
  | { type: 'NEW_SESSION'; payload: { model?: string } }
  | { type: 'ADD_MESSAGE'; payload: Message }
  | { type: 'SET_MODEL'; payload: string }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'LOAD_SESSIONS'; payload: ChatSession[] }
  | { type: 'SELECT_SESSION'; payload: string }
  | { type: 'DELETE_SESSION'; payload: string }
  | { type: 'UPDATE_SESSION_TITLE'; payload: { sessionId: string; title: string } }

const initialState: ChatState = {
  sessions: [],
  currentSessionId: null,
  selectedModel: 'mistral',
  isLoading: false,
  error: null
}

const chatReducer = (state: ChatState, action: ChatAction): ChatState => {
  switch (action.type) {
    case 'NEW_SESSION': {
      const newSession: ChatSession = {
        id: crypto.randomUUID(),
        title: 'New Chat',
        messages: [],
        createdAt: new Date(),
        model: action.payload?.model || state.selectedModel
      }
      return {
        ...state,
        sessions: [newSession, ...state.sessions],
        currentSessionId: newSession.id,
        selectedModel: newSession.model
      }
    }

    case 'ADD_MESSAGE': {
      if (!state.currentSessionId) return state
      return {
        ...state,
        sessions: state.sessions.map(s =>
          s.id === state.currentSessionId
            ? { ...s, messages: [...s.messages, action.payload] }
            : s
        )
      }
    }

    case 'SET_MODEL':
      return { ...state, selectedModel: action.payload }

    case 'SET_LOADING':
      return { ...state, isLoading: action.payload }

    case 'SET_ERROR':
      return { ...state, error: action.payload }

    case 'LOAD_SESSIONS':
      const newSessions = action.payload
      // Validate currentSessionId still exists in new sessions
      const validSessionId = newSessions.some(s => s.id === state.currentSessionId)
        ? state.currentSessionId
        : newSessions[0]?.id || null
      return { ...state, sessions: newSessions, currentSessionId: validSessionId }

    case 'SELECT_SESSION':
      return { ...state, currentSessionId: action.payload }

    case 'DELETE_SESSION':
      const remainingSessions = state.sessions.filter(s => s.id !== action.payload)
      return {
        ...state,
        sessions: remainingSessions,
        currentSessionId:
          state.currentSessionId === action.payload
            ? remainingSessions[0]?.id || null
            : state.currentSessionId
      }

    case 'UPDATE_SESSION_TITLE':
      return {
        ...state,
        sessions: state.sessions.map(s =>
          s.id === action.payload.sessionId
            ? { ...s, title: action.payload.title }
            : s
        )
      }

    default:
      return state
  }
}

interface ChatContextType {
  state: ChatState
  dispatch: React.Dispatch<ChatAction>
  addMessage: (message: Message) => void
  newSession: (model?: string) => void
  selectSession: (sessionId: string) => void
  deleteSession: (sessionId: string) => void
  setModel: (model: string) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  currentSession: ChatSession | null
}

const ChatContext = createContext<ChatContextType | null>(null)

export function ChatProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(chatReducer, initialState)

  const addMessage = useCallback((message: Message) => {
    dispatch({ type: 'ADD_MESSAGE', payload: message })
  }, [])

  const newSession = useCallback((model?: string) => {
    dispatch({ type: 'NEW_SESSION', payload: { model } })
  }, [])

  const selectSession = useCallback((sessionId: string) => {
    dispatch({ type: 'SELECT_SESSION', payload: sessionId })
  }, [])

  const deleteSession = useCallback((sessionId: string) => {
    dispatch({ type: 'DELETE_SESSION', payload: sessionId })
  }, [])

  const setModel = useCallback((model: string) => {
    dispatch({ type: 'SET_MODEL', payload: model })
  }, [])

  const setLoading = useCallback((loading: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: loading })
  }, [])

  const setError = useCallback((error: string | null) => {
    dispatch({ type: 'SET_ERROR', payload: error })
  }, [])

  const currentSession =
    state.sessions.find(s => s.id === state.currentSessionId) || null

  return (
    <ChatContext.Provider
      value={{
        state,
        dispatch,
        addMessage,
        newSession,
        selectSession,
        deleteSession,
        setModel,
        setLoading,
        setError,
        currentSession
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error('useChat must be used within ChatProvider')
  }
  return context
}
