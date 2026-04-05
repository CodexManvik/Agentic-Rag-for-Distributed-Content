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
}

interface StreamHandlers {
  onChunk: (text: string) => void;
  onTrace: (event: StreamTraceEvent) => void;
  onComplete: (payload: StreamCompletePayload) => void;
  onError: (error: Error) => void;
}

export async function fetchModels(): Promise<string[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/models`);
  if (!response.ok) {
    throw new Error(`Failed to fetch models: HTTP ${response.status}`);
  }

  const data = (await response.json()) as { models?: unknown };
  if (!data || !Array.isArray(data.models)) {
    throw new Error("Invalid models payload from backend");
  }

  const models = data.models.filter((m): m is string => typeof m === "string" && m.length > 0);
  if (models.length === 0) {
    throw new Error("Backend returned no available models");
  }

  return models;
}

export function streamChat(query: string, handlers: StreamHandlers, model?: string): EventSource {
  if (!query.trim()) {
    throw new Error("Query cannot be empty");
  }

  const url = new URL(`${BACKEND_BASE_URL}/chat/stream`);
  url.searchParams.set("query", query);
  if (model) {
    url.searchParams.set("model", model);
  }

  const source = new EventSource(url.toString());

  source.addEventListener("chunk", (event) => {
    try {
      const payload = JSON.parse((event as MessageEvent).data) as { text?: unknown };
      if (typeof payload.text !== "string") {
        throw new Error("Invalid chunk event payload");
      }
      handlers.onChunk(payload.text);
    } catch (error) {
      handlers.onError(error instanceof Error ? error : new Error("Failed to parse chunk event"));
      source.close();
    }
  });

  source.addEventListener("trace", (event) => {
    try {
      const payload = JSON.parse((event as MessageEvent).data) as StreamTraceEvent;
      handlers.onTrace(payload);
    } catch (error) {
      handlers.onError(error instanceof Error ? error : new Error("Failed to parse trace event"));
      source.close();
    }
  });

  source.addEventListener("complete", (event) => {
    try {
      const payload = JSON.parse((event as MessageEvent).data) as StreamCompletePayload;
      handlers.onComplete(payload);
    } catch (error) {
      handlers.onError(error instanceof Error ? error : new Error("Failed to parse complete event"));
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
    } catch {
      // Ignore parse errors and report generic connection issue below.
    }
    handlers.onError(new Error("Streaming connection failed. Backend may be unreachable."));
    source.close();
  });

  source.onerror = () => {
    handlers.onError(new Error("Streaming connection failed. Backend may be unreachable."));
    source.close();
  };

  return source;
}