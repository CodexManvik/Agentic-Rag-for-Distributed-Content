import { useEffect, useRef, useState } from "react";
import type { StreamCitation, StreamTraceEvent, ChatMessage } from "../../../../api";

interface ArticleDetailSectionProps {
  models: string[];
  selectedModel: string;
  onModelChange: (model: string) => void;
  query: string;
  onQueryChange: (value: string) => void;
  onSend: () => void;
  messages: ChatMessage[];
  streamingAnswer: string;
  liveTrace: StreamTraceEvent[];
  error: string | null;
  isStreaming: boolean;
}

// Node → friendly label map
const NODE_LABELS: Record<string, string> = {
  normalize_query: "Normalizing query",
  planning: "Planning sub-queries",
  retrieval: "Retrieving relevant chunks",
  adequacy: "Assessing retrieval quality",
  reformulation: "Reformulating queries",
  synthesis: "Synthesizing answer",
  citation_validation: "Validating citations",
  finalize: "Finalizing response",
  abstain: "Abstaining (insufficient evidence)",
};

const NODE_ICONS: Record<string, string> = {
  normalize_query: "🔤",
  planning: "🗺️",
  retrieval: "🔍",
  adequacy: "📊",
  reformulation: "🔄",
  synthesis: "✍️",
  citation_validation: "✅",
  finalize: "🏁",
  abstain: "⚠️",
};

function AgentTracePulse({ trace, isStreaming }: { trace: StreamTraceEvent[]; isStreaming: boolean }) {
  const latest = trace[trace.length - 1];
  if (!isStreaming && trace.length === 0) return null;

  const nodeName = latest?.node?.replace(/^agent:|^tool:|^workflow:/, "") ?? "starting";
  const label = NODE_LABELS[nodeName] ?? nodeName.replace(/_/g, " ");
  const icon = NODE_ICONS[nodeName] ?? "⚙️";
  const status = latest?.status ?? "start";

  return (
    <div className="flex items-center gap-3 px-4 py-3 bg-indigo-50 border border-indigo-100 rounded-2xl mb-4">
      {/* Animated pulse ring */}
      <div className="relative flex-shrink-0">
        <span className="relative flex h-3 w-3">
          {isStreaming && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
          )}
          <span
            className={`relative inline-flex rounded-full h-3 w-3 ${
              status === "ok" ? "bg-emerald-400" : status === "failed" ? "bg-red-400" : "bg-indigo-500"
            }`}
          />
        </span>
      </div>
      <span className="text-lg leading-none">{icon}</span>
      <div className="flex flex-col min-w-0">
        <span className="text-xs font-semibold text-indigo-800 [font-family:'Plus_Jakarta_Sans',Helvetica] truncate">
          {label}
        </span>
        {latest?.detail && (
          <span className="text-[10px] text-indigo-500 [font-family:'Plus_Jakarta_Sans',Helvetica] truncate">
            {latest.detail}
          </span>
        )}
      </div>
      {isStreaming && (
        <div className="ml-auto flex gap-1 flex-shrink-0">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function StageTimingsDropdown({ timings }: { timings: Record<string, number> }) {
  const [open, setOpen] = useState(false);
  const entries = Object.entries(timings).sort(([, a], [, b]) => b - a);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  if (entries.length === 0) return null;

  return (
    <div className="mt-3 border border-slate-200 rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full px-4 py-3 bg-slate-50 hover:bg-slate-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-xs font-semibold text-slate-600 [font-family:'Plus_Jakarta_Sans',Helvetica]">
            Stage timings · {(total / 1000).toFixed(1)}s total
          </span>
        </div>
        <svg
          className={`w-4 h-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="px-4 pb-3 pt-2 bg-white flex flex-col gap-2">
          {entries.map(([stage, ms]) => {
            const pct = total > 0 ? (ms / total) * 100 : 0;
            const label = NODE_LABELS[stage] ?? stage.replace(/_/g, " ");
            const icon = NODE_ICONS[stage] ?? "⚙️";
            return (
              <div key={stage} className="flex flex-col gap-1">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-slate-600 [font-family:'Plus_Jakarta_Sans',Helvetica] flex items-center gap-1.5">
                    <span>{icon}</span> {label}
                  </span>
                  <span className="text-xs font-semibold text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
                    {ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`}
                  </span>
                </div>
                <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-400 rounded-full transition-all"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function CitationCard({ citation }: { citation: StreamCitation }) {
  return (
    <div className="flex flex-col items-start gap-2 p-3 bg-slate-50 rounded-2xl border border-slate-200 min-w-[140px] max-w-[200px] flex-shrink-0">
      <div className="flex items-center gap-1.5 w-full">
        <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-100 text-indigo-600 text-[10px] font-bold flex-shrink-0 [font-family:'Plus_Jakarta_Sans',Helvetica]">
          {citation.index}
        </span>
        {citation.source_type && (
          <span className="text-[9px] uppercase font-semibold text-slate-400 [font-family:'Plus_Jakarta_Sans',Helvetica]">
            {citation.source_type}
          </span>
        )}
      </div>
      <p className="text-xs font-medium text-slate-800 [font-family:'Plus_Jakarta_Sans',Helvetica] line-clamp-2 leading-4">
        {citation.snippet || citation.source}
      </p>
      <div className="flex items-center gap-1 w-full">
        <svg className="w-3 h-3 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <span className="text-[10px] text-slate-500 truncate [font-family:'Plus_Jakarta_Sans',Helvetica]">
          {citation.source}
          {citation.page_number ? ` · p.${citation.page_number}` : ""}
        </span>
      </div>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[420px] px-4 py-3 bg-indigo-600 text-white rounded-2xl rounded-tr-sm [font-family:'Plus_Jakarta_Sans',Helvetica] text-sm leading-relaxed">
          {msg.content}
        </div>
      </div>
    );
  }

  // Assistant message
  return (
    <div className="flex flex-col gap-3 mb-6">
      {/* Citations */}
      {msg.citations && msg.citations.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
            <span className="text-xs font-bold text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Sources
            </span>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {msg.citations.map((c, i) => (
              <CitationCard key={`${c.index}-${i}`} citation={c} />
            ))}
          </div>
        </div>
      )}

      {/* Answer */}
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <span className="text-xs font-bold text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">Answer</span>
          {typeof msg.confidence === "number" && (
            <span
              className={`ml-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full [font-family:'Plus_Jakarta_Sans',Helvetica] ${
                msg.confidence >= 0.7
                  ? "bg-emerald-100 text-emerald-700"
                  : msg.confidence >= 0.4
                  ? "bg-amber-100 text-amber-700"
                  : "bg-red-100 text-red-600"
              }`}
            >
              {Math.round(msg.confidence * 100)}% confidence
            </span>
          )}
          {msg.short_circuited && (
            <span className="ml-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-indigo-100 text-indigo-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              short-circuited
            </span>
          )}
          {msg.abstained && (
            <span className="ml-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-orange-100 text-orange-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              abstained
            </span>
          )}
        </div>
        <p className="text-sm text-slate-700 leading-relaxed [font-family:'Plus_Jakarta_Sans',Helvetica] whitespace-pre-wrap">
          {msg.content}
        </p>
      </div>

      {/* Sub-queries */}
      {msg.sub_queries && msg.sub_queries.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-1">
          {msg.sub_queries.map((q, i) => (
            <span
              key={i}
              className="text-[10px] px-2 py-1 bg-slate-100 text-slate-500 rounded-full [font-family:'Plus_Jakarta_Sans',Helvetica]"
            >
              {q}
            </span>
          ))}
        </div>
      )}

      {/* Stage timings */}
      {msg.stage_timings && Object.keys(msg.stage_timings).length > 0 && (
        <StageTimingsDropdown timings={msg.stage_timings} />
      )}
    </div>
  );
}

export const ArticleDetailSection = ({
  models,
  selectedModel,
  onModelChange,
  query,
  onQueryChange,
  onSend,
  messages,
  streamingAnswer,
  liveTrace,
  error,
  isStreaming,
}: ArticleDetailSectionProps): JSX.Element => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingAnswer, liveTrace]);

  const isEmpty = messages.length === 0 && !isStreaming;

  return (
    <div className="flex flex-col flex-1 h-screen bg-white overflow-hidden">
      {/* Topbar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-slate-200 flex-shrink-0">
        <div />
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 [font-family:'Plus_Jakarta_Sans',Helvetica]">Model</span>
          <select
            className="[font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-indigo-600 text-sm bg-transparent border-0 outline-none cursor-pointer"
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        <div />
      </div>

      {/* Chat area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-12 py-6">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <div className="text-5xl">🍁</div>
            <h2 className="text-2xl font-bold text-slate-800 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              What do you want to know?
            </h2>
            <p className="text-slate-500 text-sm [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Ask anything about your uploaded documents
            </p>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}

            {/* Live streaming state */}
            {isStreaming && (
              <div className="flex flex-col gap-3 mb-6">
                <AgentTracePulse trace={liveTrace} isStreaming={isStreaming} />
                {streamingAnswer && (
                  <div>
                    <div className="flex items-center gap-1.5 mb-2">
                      <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      <span className="text-xs font-bold text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
                        Answer
                      </span>
                    </div>
                    <p className="text-sm text-slate-700 leading-relaxed [font-family:'Plus_Jakarta_Sans',Helvetica] whitespace-pre-wrap">
                      {streamingAnswer}
                      <span className="inline-block w-0.5 h-4 bg-indigo-500 ml-0.5 animate-pulse align-middle" />
                    </p>
                  </div>
                )}
                {!streamingAnswer && (
                  <div className="flex gap-1.5 py-2">
                    {[0, 1, 2].map((i) => (
                      <div
                        key={i}
                        className="w-2 h-2 rounded-full bg-indigo-300 animate-bounce"
                        style={{ animationDelay: `${i * 120}ms` }}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-12 mb-2 px-4 py-3 bg-red-50 border border-red-200 rounded-2xl flex items-center gap-2">
          <svg className="w-4 h-4 text-red-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm text-red-600 [font-family:'Plus_Jakarta_Sans',Helvetica]">{error}</span>
        </div>
      )}

      {/* Input bar */}
      <div className="px-12 pb-6 pt-2 flex-shrink-0">
        <div className="max-w-2xl mx-auto flex items-center gap-3">
          <div className="flex-1 flex items-center gap-2.5 px-4 py-3 bg-white rounded-2xl border border-slate-300 shadow-sm focus-within:border-indigo-400 focus-within:shadow-indigo-100 focus-within:shadow-md transition-all">
            <svg className="w-5 h-5 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            <input
              className="flex-1 bg-transparent border-none outline-none text-slate-700 placeholder:text-slate-400 text-sm [font-family:'Plus_Jakarta_Sans',Helvetica]"
              placeholder={isStreaming ? "Agent is working…" : "Ask a follow up…"}
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!query.trim()) return;
                  onSend();
                }
              }}
              disabled={isStreaming}
            />
          </div>
          <button
            onClick={onSend}
            disabled={isStreaming || !query.trim()}
            className="w-11 h-11 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed rounded-2xl flex items-center justify-center transition-colors shadow-sm"
          >
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};
