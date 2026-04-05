---
title: Implementation Quick Start Guide
version: 1.0
date: 2026-04-05
---

# Quick Start Guide - Where to Begin

This guide provides a quick reference for starting implementation immediately after planning.

---

## Document Map

**Three documents have been created:**

1. **feature-agentic-rag-production-1.md** (~400 KB)
   - Full project plan with 3 phases
   - All requirements, constraints, risks
   - Task breakdown with specific file paths
   - Testing strategy
   - Detailed implementation details

2. **technical-specification-1.md** (~150 KB)
   - API schemas and request/response formats
   - Complete directory structure
   - Code patterns and examples
   - Configuration files
   - Development setup instructions

3. **IMPLEMENTATION-CHECKLIST.md** (this document)
   - Quick reference for Phase 1 tasks
   - Copy-paste code snippets
   - Step-by-step instructions
   - Validation commands

---

## Phase 1: Quick Start (5 Sequential Days)

**Important**: Complete each day fully before starting the next. Validate at each checkpoint.

---

### Day 1: Backend Connectivity (Checkpoint 1)

**Goal**: Desktop app can reach backend, receive HTTP responses

#### Step 1.1: Update FastAPI with CORS (30 min)

File: `backend/app/main.py`

Add this after imports, before FastAPI app creation:

```python
from fastapi.middleware.cors import CORSMiddleware

# ... existing imports ...

# Add CORS support for desktop app
CORS_ORIGINS = [
    "http://localhost:5173",      # Vite dev
    "http://127.0.0.1:5173",
    "http://localhost:3000",      # Electron dev
    "http://127.0.0.1:3000",
    "file://",                    # Electron file protocol
]

app = FastAPI(...)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Validation**:
```bash
curl -X OPTIONS http://localhost:8000 \
  -H "Origin: http://localhost:5173" \
  -v
```

#### Step 1.2: Add Health Check Endpoint (15 min)

File: `backend/app/main.py`

Add simple health check endpoint.

```python
from fastapi import HTTPException
from fastapi import Query
from starlette.responses import StreamingResponse
import json

@app.get("/chat/stream")
async def chat_stream(query: str = Query(..., min_length=1)):
    """
    Stream RAG response with real-time trace events.
    Supports SSE (Server-Sent Events) for client streaming.
    """
    async def event_generator():
        try:
            # Initialize state
            state: NavigatorState = {
                "query": query,
                "original_query": request.query,
                "sub_queries": [],
                "retrieved_chunks": [],
                "final_response": "",
                "citations": [],
                "retrieval_quality": {},
                "retries_used": 0,
                "validation_retries_used": 0,
                "validation_errors": [],
                "used_deterministic_fallback": False,
                "abstained": False,
                "abstain_reason": None,
                "confidence": 0.0,
                "cited_indices": [],
                "synthesis_output": {},
                "trace": [],
                "stage_timings": {},
                "stage_timestamps": {},
            }
            
            logger.info(f"Starting stream for query: {request.query}")
            

            async for event in workflow.astream(state):
                event_type = event.get("type")

            if event_type == "trace":
                yield f"event: trace\ndata: {json.dumps(event['data'])}\n\n"

            elif event_type == "chunk":
                yield f"event: chunk\ndata: {json.dumps(event['data'])}\n\n"

            elif event_type == "complete":
                yield f"event: complete\ndata: {json.dumps(event['data'])}\n\n"

            # Create response
            response_data = {
                "answer": final_state.get("final_response", ""),
                "citations": final_state.get("citations", []),
                "sub_queries": final_state.get("sub_queries", []),
                "confidence": final_state.get("confidence", 0.0),
                "abstained": final_state.get("abstained", False),
                "abstain_reason": final_state.get("abstain_reason"),
                "trace": final_state.get("trace", []),
                "retrieval_quality": final_state.get("retrieval_quality", {}),
                "stage_timings": final_state.get("stage_timings", {}),
            }
            
            # Emit completion event
            yield f"event: <type>\ndata: {json.dumps({
                'type': 'complete',
                'data': response_data
            })}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"event: <type>\ndata: {json.dumps({
                'type': 'error',
                'error': str(e)
            })}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
```

**Validation**:
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"what is RAG?"}' \
  -N
```

(Note: `-N` disables buffering to show streaming output)

#### Step 1.3: Create Model Manager Service (1-2 hours)

File: `backend/app/services/model_manager.py` (Create new)

```python
import json
import requests
from typing import Dict, List, Optional
from loguru import logger
from app.config import settings

class ModelManager:
    """Manage Ollama models: listing, downloading, validation."""
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
    
    def list_models(self) -> Dict[str, List]:
        """Get list of downloaded models from Ollama."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            models = data.get('models', [])
            
            # Split into chat and embedding models
            chat_models = [m for m in models if 'embed' not in m.get('name', '').lower()]
            embedding_models = [m for m in models if 'embed' in m.get('name', '').lower()]
            
            return {
                'chat': chat_models,
                'embedding': embedding_models,
                'total': len(models)
            }
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            raise Exception("Ollama not connected")
    
    def get_model_status(self, model_name: str) -> Dict:
        """Check if model is downloaded."""
        try:
            models = self.list_models()
            all_models = models['chat'] + models['embedding']
            
            for model in all_models:
                if model_name in model.get('name', ''):
                    return {
                        'downloaded': True,
                        'name': model['name'],
                        'size': model.get('size', 0),
                    }
            
            return {
                'downloaded': False,
                'name': model_name,
                'size': 0,
            }
        except Exception as e:
            return {
                'downloaded': False,
                'error': str(e)
            }
    
    def pull_model(self, model_name: str):
        """Download a model from Ollama registry."""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=None  # No timeout for downloads
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    yield {
                        'status': data.get('status', ''),
                        'completed': data.get('completed', 0),
                        'total': data.get('total', 0),
                    }
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            raise

# Create global instance
model_manager = ModelManager()
```

Add to `backend/app/main.py`:

```python
from app.services.model_manager import model_manager

@app.get("/api/models")
async def get_models():
    """List available Ollama models."""
    try:
        models_info = await asyncio.to_thread(model_manager.list_models)
        return {
            "chat_models": models_info['chat'],
            "embedding_models": models_info['embedding'],
            "current_chat_model": settings.ollama_chat_model,
            "current_embedding_model": settings.ollama_embedding_model,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/api/models/pull/{model_name}")
async def pull_model(model_name: str):
    """Stream model download progress."""
    async def event_generator():
        try:
            for progress in model_manager.pull_model(model_name):
                yield f"event: <type>\ndata: {json.dumps(progress)}\n\n"
        except Exception as e:
            yield f"event: <type>\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

**Validation**:
```bash
curl http://localhost:8000/api/models
```

### Week 2: Frontend Updates

#### Step 2.1: Configure Backend URL (15 min)

File: `desktop-app/.env.local` (Create new)

```
VITE_BACKEND_URL=http://localhost:8000
VITE_DEBUG=true
```

Update `desktop-app/src/renderer/context/ChatContext.tsx`:

```typescript
// Add at top of file
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const BACKEND_STREAM_URL = `${BACKEND_URL}/chat/stream`;

// Export for other components
export const useBackendURL = () => BACKEND_URL;
```

#### Step 2.2: Implement SSE Client (1 hour)

File: `desktop-app/src/renderer/context/ChatContext.tsx`

Add this hook:

```typescript
interface UseSSEResult {
    streamResponse: (query: string) => Promise<void>;
    isLoading: boolean;
    error: string | null;
}

export const useSSE = (): UseSSEResult => {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const streamResponse = async (query: string) => {
        setIsLoading(true);
        setError(null);
        
        try {
            // Validate query
            if (!query || query.trim().length < 3) {
                setError('Query too short');
                setIsLoading(false);
                return;
            }
            
            // Create SSE connection
            const eventSource = new EventSource(
                `${BACKEND_STREAM_URL}?query=${encodeURIComponent(query)}`,
                { withCredentials: true }
            );
            
            eventSource.addEventListener('complete', (event) => {
                try {
                    const response = JSON.parse(event.data);
                    useChat.setState(prev => ({
                        ...prev,
                        messages: [...prev.messages, {
                            id: Date.now(),
                            role: 'assistant',
                            content: response.data.answer,
                            citations: response.data.citations,
                            trace: response.data.trace,
                        }]
                    }));
                    eventSource.close();
                } catch (e) {
                    setError('Failed to parse response');
                }
                setIsLoading(false);
            });
            
            eventSource.addEventListener('error', (event) => {
                try {
                    const err = JSON.parse(event.data);
                    setError(err.error || 'Stream error');
                } catch {
                    setError('Connection lost');
                }
                eventSource.close();
                setIsLoading(false);
            });
            
        } catch (err: any) {
            setError(err.message || 'Failed to connect');
            setIsLoading(false);
        }
    };
    
    return { streamResponse, isLoading, error };
};
```

#### Step 2.3: Create Trace Viewer Component (1 hour)

File: `desktop-app/src/renderer/components/TraceViewer.tsx` (Create new)

```typescript
import { useState } from 'react';
import { ChevronDown } from 'lucide-react';

interface TraceEvent {
    node: string;
    status: string;
    detail: string;
    ts: string;
    duration_ms?: number;
}

export function TraceViewer({ trace }: { trace: TraceEvent[] }) {
    const [isOpen, setIsOpen] = useState(true);
    
    return (
        <div className="trace-viewer bg-slate-800 rounded p-3 mt-2">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 w-full text-left text-sm font-medium"
            >
                <ChevronDown 
                    size={16} 
                    className={`transform transition ${isOpen ? '' : '-rotate-90'}`}
                />
                Reasoning Trace ({trace.length} steps)
            </button>
            
            {isOpen && (
                <div className="mt-3 space-y-2 text-xs">
                    {trace.map((event, i) => (
                        <div key={i} className="bg-slate-700 p-2 rounded">
                            <div className="flex justify-between">
                                <span className="font-mono font-bold text-blue-300">
                                    {event.node}
                                </span>
                                <span className={`text-${
                                    event.status === 'completed' ? 'green' : 
                                    event.status === 'failed' ? 'red' : 
                                    'yellow'
                                }-300`}>
                                    {event.status}
                                </span>
                            </div>
                            <p className="text-gray-300 mt-1">{event.detail}</p>
                            {event.duration_ms && (
                                <p className="text-gray-400 text-xs mt-1">
                                    {event.duration_ms.toFixed(0)}ms
                                </p>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
```

#### Step 2.4: Update ChatWindow Component (1 hour)

File: `desktop-app/src/renderer/components/ChatWindow.tsx`

Update the message rendering section:

```typescript
{messages.map((message) => (
    <div key={message.id} className={`message ${message.role}`}>
        {message.role === 'assistant' ? (
            <div className="assistant-message bg-slate-800 rounded p-4">
                {/* Main response text (streaming) */}
                <p className="text-white leading-relaxed">
                    {message.content}
                    {message.isStreaming && <span className="animate-pulse">▌</span>}
                </p>
                
                {/* Trace dropdown */}
                {message.trace && message.trace.length > 0 && (
                    <TraceViewer trace={message.trace} />
                )}
                
                {/* Citations */}
                {message.citations && message.citations.length > 0 && (
                    <div className="citations mt-4 pt-4 border-t border-slate-600">
                        <p className="text-xs font-bold text-gray-400 mb-2">Sources:</p>
                        {message.citations.map((cite, i) => (
                            <div key={i} className="text-xs mb-2 text-blue-300">
                                <span className="font-mono">[{cite.index}]</span>
                                {' '}{cite.source}
                                {cite.url && (
                                    <a href={cite.url} className="ml-2 underline">
                                        (link)
                                    </a>
                                )}
                                <p className="text-gray-400 mt-1 line-clamp-2">
                                    {cite.snippet}
                                </p>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        ) : (
            <div className="user-message bg-blue-900 rounded p-4 ml-auto">
                {message.content}
            </div>
        )}
    </div>
))}
```

### Week 3: Testing & Polish

#### Step 3.1: Test End-to-End

```bash
# Start backend
python -m uvicorn app.main:app --reload --port 8000

# In another terminal, start frontend
npm run dev --prefix desktop-app

# In app, type a query and verify:
# ✅ Response streams in real-time
# ✅ Trace renders in dropdown
# ✅ Citations display with sources
# ✅ No console errors
```

#### Step 3.2: Add Error Handling

File: `desktop-app/src/renderer/components/ChatWindow.tsx`

```typescript
{error && (
    <div className="bg-red-900 border border-red-700 rounded p-4 mb-4">
        <p className="font-bold text-red-100">Error</p>
        <p className="text-red-200 text-sm mt-1">{error}</p>
        {error.includes('Ollama') && (
            <p className="text-red-300 text-xs mt-2">
                → Start Ollama: ollama serve
            </p>
        )}
    </div>
)}
```

---

## Next Steps After Phase 1

Once Phase 1 is complete:

1. **Commit & Tag**: 
   ```bash
   git commit -m "feat: Phase 1 - Backend-Frontend Connectivity & Streaming"
   git tag phase-1-complete
   ```

2. **Start Phase 2** (Week 4):
   - Create agent registry system
   - Convert hardcoded nodes to agent configs
   - Test dynamic execution

3. **Start Phase 3** (Week 7):
   - Multi-agent coordination
   - Tool system
   - Advanced tracing

---

## Quick Reference: Common Commands

```bash
# Start backend
cd backend
python -m uvicorn app.main:app --reload

# Start frontend
cd desktop-app
npm run dev

# Test API
curl http://localhost:8000/api/models
curl -N http://localhost:8000/chat/stream -d '{"query":"test"}'

# Check Ollama
curl http://localhost:11434/api/tags

# Run tests
pytest backend/tests/

# Build for production
npm run build --prefix desktop-app
```

---

## Documentation Structure

All three documents work together:

1. **This file** (Quick Start) →  Start here, copy code
2. **Technical Spec** → Reference for architecture
3. **Full Plan** → Deep dive on requirements, risks, timeline

---

**Version**: 1.0  
**Created**: 2026-04-05  
**Next Update**: After Phase 1 completion