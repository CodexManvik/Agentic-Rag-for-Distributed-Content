import { ChevronDown } from 'lucide-react'
import { useChat } from '@/renderer/context/ChatContext'
import { useChat_API } from '@/renderer/hooks/useChat_API'
import { useState } from 'react'

const CANDIDATE_MODELS = ['qwen:4b', 'mistral', 'llama3.2', 'phi3:mini', 'gemma:2b']

export function ModelSelector() {
  const { state, setModel } = useChat()
  const { models } = useChat_API()
  const [isOpen, setIsOpen] = useState(false)

  if (models.length === 0) {
    return (
      <div className="px-4 py-2 rounded-lg bg-slate-700 text-slate-300 text-sm">
        Loading models...
      </div>
    )
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 border border-slate-700 hover:border-slate-600 text-white font-medium transition-all min-w-[200px] justify-between"
      >
        <span className="text-sm">
          {models.find(m => m === state.selectedModel) || state.selectedModel}
        </span>
        <ChevronDown
          size={16}
          className={`transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {isOpen && (
        <div className="absolute top-full mt-2 right-0 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 min-w-[200px] overflow-hidden">
          {CANDIDATE_MODELS.map(model => {
            const available = models.includes(model)
            return (
            <button
              key={model}
              onClick={() => {
                if (available) {
                  setModel(model)
                }
                setIsOpen(false)
              }}
              disabled={!available}
              className={`w-full text-left px-4 py-3 text-sm transition-colors ${
                state.selectedModel === model
                  ? 'bg-blue-600 text-white font-medium'
                  : available
                    ? 'text-slate-200 hover:bg-slate-700'
                    : 'text-slate-500 bg-slate-900/40 cursor-not-allowed'
              }`}
            >
              {model}
              {!available && ' (unavailable)'}
            </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
