const BACKEND_BASE_URL = "http://localhost:8000";

export interface StreamTraceEvent {
  node: string;
  status: string;
  detail: string;
  ts?: string;
  duration_ms?: number;
}

export interface StreamCitation {
  index: number;
  source: string;
  url?: string | null;
  snippet: string;
  source_type?: string | null;
  section?: string | null;
  page_number?: number | null;
}

export interface StreamCompletePayload {
  answer: string;
  citations: StreamCitation[];
  sub_queries: string[];
  confidence: number;
  abstained: boolean;
  abstain_reason?: string | null;
  trace: StreamTraceEvent[];
  retrieval_quality: Record<string, unknown>;
  stage_timings: Record<string, number>;
  short_circuited?: boolean;
  session_id?: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: StreamCitation[];
  trace?: StreamTraceEvent[];
  stage_timings?: Record<string, number>;
  confidence?: number;
  abstained?: boolean;
  sub_queries?: string[];
  short_circuited?: boolean;
  ts: number;
}

export interface AppSettings {
  model_temperature: number;
  ollama_chat_model: string;
  context_chunk_limit: number;
  context_chunk_char_limit: number;
  planner_max_subqueries: number;
  chunk_size: number;
  chunk_overlap: number;
  runtime_profile: string;
  retrieval_top_k: number;
  max_retrieval_retries: number;
  max_validation_retries: number;
  enable_short_circuit_routing: boolean;
  short_circuit_confidence_threshold: number;
}

interface StreamHandlers {
  onChunk: (text: string) => void;
  onTrace: (event: StreamTraceEvent) => void;
  onComplete: (payload: StreamCompletePayload) => void | Promise<void>;
  onError: (error: Error) => void;
}

interface BackendConversationMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

interface BackendSessionState {
  session_id: string;
  user_id?: string | null;
  created_at: string;
  last_active: string;
  conversation_history: BackendConversationMessage[];
}

interface BackendSessionListResponse {
  sessions: BackendSessionState[];
  total: number;
  limit: number;
  offset: number;
}

export async function fetchModels(): Promise<string[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/models`);
  if (!response.ok) throw new Error(`Failed to fetch models: HTTP ${response.status}`);
  const data = (await response.json()) as { models?: unknown };
  if (!data || !Array.isArray(data.models)) throw new Error("Invalid models payload from backend");
  const models = data.models.filter((m): m is string => typeof m === "string" && m.length > 0);
  if (models.length === 0) throw new Error("Backend returned no available models");
  return models;
}

export async function fetchSettings(): Promise<AppSettings> {
  const response = await fetch(`${BACKEND_BASE_URL}/settings`);
  if (!response.ok) throw new Error(`Failed to fetch settings: HTTP ${response.status}`);
  return response.json();
}

export async function updateSettings(patch: Partial<AppSettings>): Promise<{ updated: Record<string, unknown> }> {
  const response = await fetch(`${BACKEND_BASE_URL}/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!response.ok) throw new Error(`Failed to update settings: HTTP ${response.status}`);
  return response.json();
}

export async function ingestFile(file: File): Promise<{ filename: string; chunks_added: number; skipped_duplicates: number; errors: string[] }> {
  console.log("🔄 Starting file ingest for:", file.name);
  const form = new FormData();
  form.append("file", file);
  
  const url = `${BACKEND_BASE_URL}/ingest`;
  console.log("📤 Sending POST to:", url);
  
  try {
    const response = await fetch(url, {
      method: "POST",
      body: form,
      // Don't set Content-Type header - let browser set it with boundary
    });
    console.log("📥 Response status:", response.status);
    
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error((err as { detail?: string }).detail || "Upload failed");
    }
    const result = await response.json();
    console.log("✅ Ingest successful:", result);
    return result;
  } catch (err) {
    console.error("❌ Ingest error:", err);
    throw err;
  }
}

export function streamChat(
  query: string,
  handlers: StreamHandlers,
  model?: string,
  sessionId?: string,
): EventSource {
  if (!query.trim()) throw new Error("Query cannot be empty");
  const url = new URL(`${BACKEND_BASE_URL}/chat/stream`);
  url.searchParams.set("query", query);
  if (model) url.searchParams.set("model", model);
  if (sessionId) url.searchParams.set("session_id", sessionId);
  const source = new EventSource(url.toString());

  source.addEventListener("chunk", (event) => {
    try {
      const payload = JSON.parse((event as MessageEvent).data) as { text?: unknown };
      if (typeof payload.text !== "string") throw new Error("Invalid chunk");
      handlers.onChunk(payload.text);
    } catch (error) {
      handlers.onError(error instanceof Error ? error : new Error("Failed to parse chunk"));
      source.close();
    }
  });

  source.addEventListener("trace", (event) => {
    try {
      handlers.onTrace(JSON.parse((event as MessageEvent).data) as StreamTraceEvent);
    } catch (error) {
      handlers.onError(error instanceof Error ? error : new Error("Failed to parse trace"));
      source.close();
    }
  });

  source.addEventListener("complete", (event) => {
    try {
      void handlers.onComplete(JSON.parse((event as MessageEvent).data) as StreamCompletePayload);
    } catch (error) {
      handlers.onError(error instanceof Error ? error : new Error("Failed to parse complete"));
    } finally {
      source.close();
    }
  });

  source.addEventListener("error", (event) => {
    try {
      const payload = JSON.parse((event as MessageEvent).data) as { error?: unknown };
      if (typeof payload.error === "string" && payload.error.length > 0) {
        handlers.onError(new Error(payload.error));
        source.close();
        return;
      }
    } catch { /* ignore */ }
    handlers.onError(new Error("Streaming connection failed. Backend may be unreachable."));
    source.close();
  });

  source.onerror = () => {
    handlers.onError(new Error("Streaming connection failed. Backend may be unreachable."));
    source.close();
  };

  return source;
}

// ── Chat sessions (backend APIs) ───────────────────────────────────────────

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

function toChatMessage(message: BackendConversationMessage): ChatMessage {
  const role = message.role === "assistant" ? "assistant" : "user";
  const metadata = message.metadata && typeof message.metadata === "object" ? message.metadata : {};
  return {
    id: `msg-${Date.parse(message.timestamp)}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    content: message.content,
    citations: Array.isArray(metadata.citations) ? (metadata.citations as StreamCitation[]) : [],
    trace: Array.isArray(metadata.trace) ? (metadata.trace as StreamTraceEvent[]) : [],
    stage_timings:
      metadata.stage_timings && typeof metadata.stage_timings === "object"
        ? (metadata.stage_timings as Record<string, number>)
        : {},
    confidence: typeof metadata.confidence === "number" ? metadata.confidence : undefined,
    abstained: typeof metadata.abstained === "boolean" ? metadata.abstained : undefined,
    sub_queries: Array.isArray(metadata.sub_queries) ? (metadata.sub_queries as string[]) : [],
    short_circuited:
      typeof metadata.short_circuited === "boolean" ? metadata.short_circuited : undefined,
    ts: Date.parse(message.timestamp) || Date.now(),
  };
}

function toChatSession(session: BackendSessionState): ChatSession {
  const messages = session.conversation_history.map(toChatMessage);
  const firstUser = messages.find((m) => m.role === "user");
  return {
    id: session.session_id,
    title: firstUser?.content.slice(0, 48) || "New chat",
    createdAt: Date.parse(session.created_at) || Date.now(),
    updatedAt: Date.parse(session.last_active) || Date.now(),
    messages,
  };
}

export async function listSessions(): Promise<ChatSession[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/sessions`);
  if (!response.ok) throw new Error(`Failed to list sessions: HTTP ${response.status}`);
  const data = (await response.json()) as BackendSessionListResponse;
  return (data.sessions || []).map(toChatSession);
}

export async function getSession(sessionId: string): Promise<ChatSession> {
  const response = await fetch(`${BACKEND_BASE_URL}/sessions/${encodeURIComponent(sessionId)}`);
  if (!response.ok) throw new Error(`Failed to fetch session: HTTP ${response.status}`);
  const data = (await response.json()) as BackendSessionState;
  return toChatSession(data);
}

export async function createSession(userId = "app-ui-user"): Promise<ChatSession> {
  const response = await fetch(`${BACKEND_BASE_URL}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  if (!response.ok) throw new Error(`Failed to create session: HTTP ${response.status}`);
  const data = (await response.json()) as { session_id?: string };
  if (!data.session_id) throw new Error("Invalid session create response");
  return getSession(data.session_id);
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${BACKEND_BASE_URL}/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Failed to delete session: HTTP ${response.status}`);
  }
}

export function newSession(): ChatSession {
  return {
    id: `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title: "New chat",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
}
