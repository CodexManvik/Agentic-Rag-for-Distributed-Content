import { useEffect, useRef, useState, useCallback } from "react";
import {
  fetchModels,
  streamChat,
  fetchSettings,
  updateSettings,
  ingestFile,
  listSessions,
  getSession,
  createSession,
  deleteSession,
  type StreamCompletePayload,
  type StreamTraceEvent,
  type AppSettings,
  type ChatMessage,
  type ChatSession,
} from "../../api";
import { ArticleDetailSection } from "./sections/ArticleDetailSection";
import { NavigationSidebarSection } from "./sections/NavigationSidebarSection";
import { SettingsPanel } from "./sections/SettingsPanel";

export const Desktop = (): JSX.Element => {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [query, setQuery] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [sidebarTab, setSidebarTab] = useState<"history" | "upload">("history");

  // Session management
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);

  // Live streaming state (pre-commit)
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [liveTrace, setLiveTrace] = useState<StreamTraceEvent[]>([]);

  const streamRef = useRef<EventSource | null>(null);

  const messageKey = useCallback((message: ChatMessage): string => {
    if (message.id) return `id:${message.id}`;
    return `fallback:${message.role}:${message.ts}:${message.content}`;
  }, []);

  // Load everything on mount
  useEffect(() => {
    let mounted = true;
    const init = async () => {
      try {
        const [modelList, settingsData, sessionList] = await Promise.all([
          fetchModels(),
          fetchSettings(),
          listSessions(),
        ]);
        if (!mounted) return;
        setModels(modelList);
        setSelectedModel(modelList[0]);
        setAppSettings(settingsData);
        setSessions(sessionList);
        if (sessionList.length > 0) {
          setActiveSession(sessionList[0]);
        } else {
          const created = await createSession();
          if (!mounted) return;
          setSessions([created]);
          setActiveSession(created);
        }
        setError(null);
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Failed to connect to backend");
      }
    };
    init();
    return () => {
      mounted = false;
      streamRef.current?.close();
    };
  }, []);

  const handleStreamComplete = useCallback(
    async (payload: StreamCompletePayload) => {
      if (!activeSession) return;
      const assistantMsg: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: "assistant",
        content: payload.answer || "",
        citations: Array.isArray(payload.citations) ? payload.citations : [],
        trace: Array.isArray(payload.trace) ? payload.trace : [],
        stage_timings: payload.stage_timings || {},
        confidence: payload.confidence,
        abstained: payload.abstained,
        sub_queries: payload.sub_queries || [],
        short_circuited: payload.short_circuited,
        ts: Date.now(),
      };

      const current = await getSession(activeSession.id);
      const mergedMessages: ChatMessage[] = [];
      const seen = new Set<string>();
      const appendUnique = (messages: ChatMessage[]) => {
        for (const msg of messages) {
          const key = messageKey(msg);
          if (seen.has(key)) continue;
          seen.add(key);
          mergedMessages.push(msg);
        }
      };

      appendUnique(current.messages || []);
      appendUnique(activeSession.messages || []);
      appendUnique([assistantMsg]);

      const merged: ChatSession = {
        ...current,
        messages: mergedMessages,
      };

      setActiveSession(merged);
      setSessions((prev) => {
        const next = prev.filter((s) => s.id !== merged.id);
        return [merged, ...next];
      });

      setStreamingAnswer("");
      setLiveTrace([]);
      setIsStreaming(false);
      streamRef.current = null;
    },
    [activeSession, messageKey],
  );

  const handleSendQuery = useCallback(() => {
    if (!query.trim()) {
      setError("Please enter a query before sending.");
      return;
    }
    if (!selectedModel) {
      setError("No model selected.");
      return;
    }

    streamRef.current?.close();
    streamRef.current = null;

    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content: query.trim(),
      ts: Date.now(),
    };

    if (!activeSession) {
      setError("No active session. Create a new chat first.");
      return;
    }
    setActiveSession((prev) =>
      prev
        ? {
            ...prev,
            updatedAt: Date.now(),
            messages: [...prev.messages, userMsg],
          }
        : prev,
    );

    setError(null);
    setStreamingAnswer("");
    setLiveTrace([]);
    setIsStreaming(true);
    setQuery("");

    try {
      streamRef.current = streamChat(
        userMsg.content,
        {
          onChunk: (text) => setStreamingAnswer((prev) => prev + text),
          onTrace: (event) => setLiveTrace((prev) => [...prev, event]),
          onComplete: handleStreamComplete,
          onError: (err) => {
            setError(err.message);
            setIsStreaming(false);
            setStreamingAnswer("");
            setLiveTrace([]);
            streamRef.current?.close();
            streamRef.current = null;
          },
        },
        selectedModel,
        activeSession.id,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start stream");
      setIsStreaming(false);
    }
  }, [query, selectedModel, handleStreamComplete, activeSession]);

  const handleNewChat = useCallback(async () => {
    streamRef.current?.close();
    streamRef.current = null;
    try {
      const session = await createSession();
      setStreamingAnswer("");
      setLiveTrace([]);
      setIsStreaming(false);
      setQuery("");
      setError(null);
      setActiveSession(session);
      setSessions((prev) => [session, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create a new chat session");
    }
  }, []);

  const handleSelectSession = useCallback(async (session: ChatSession) => {
    const prevActive = activeSession;
    streamRef.current?.close();
    streamRef.current = null;
    try {
      const full = await getSession(session.id);
      setStreamingAnswer("");
      setLiveTrace([]);
      setIsStreaming(false);
      setQuery("");
      setError(null);
      setActiveSession(full);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load session");
      setActiveSession(prevActive);
    }
  }, [activeSession]);

  const handleDeleteSession = useCallback(
    async (id: string) => {
      try {
        await deleteSession(id);
        const refreshed = await listSessions();
        setSessions(refreshed);
        if (activeSession?.id === id) {
          if (refreshed.length > 0) setActiveSession(refreshed[0]);
          else {
            const created = await createSession();
            setActiveSession(created);
            setSessions([created]);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to delete session");
      }
    },
    [activeSession],
  );

  const handleSettingsSave = useCallback(async (patch: Partial<AppSettings>) => {
    try {
      await updateSettings(patch);
      const fresh = await fetchSettings();
      setAppSettings(fresh);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to save settings";
      setError(message);
      throw err;
    }
  }, []);

  const handleUpload = useCallback(async (file: File): Promise<{ chunks_added: number }> => {
    return ingestFile(file);
  }, []);

  const navIcons = [
    { src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add.svg", alt: "Home" },
    { src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-1.svg", alt: "Discover" },
    { src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-2.svg", alt: "Library" },
    { src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-3.svg", alt: "Upload" },
    { src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-4.svg", alt: "Analytics" },
    { src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-5.svg", alt: "Settings" },
  ];

  return (
    <div className="flex h-screen items-start relative bg-white w-full min-w-[1440px] overflow-hidden">
      {/* Icon rail */}
      <div className="inline-flex flex-col items-start gap-[auto] px-4 py-6 relative flex-shrink-0 bg-slate-50 border-r border-slate-200 h-full justify-between">
        <div className="inline-flex flex-col items-center gap-8">
          <img className="w-12 h-12" alt="Logo" src="https://c.animaapp.com/zPfth9Ad/img/logomark.svg" />
          <div className="inline-flex flex-col items-start gap-4">
            {navIcons.map((icon, index) => (
              <button
                key={index}
                onClick={() => {
                  if (index === 3) setSidebarTab("upload");
                  if (index === 5) setShowSettings(true);
                }}
                className={`w-12 h-12 flex items-center justify-center gap-2.5 p-4 rounded-[123px] overflow-hidden transition-colors ${
                  index === 0 ? "bg-white shadow-sm" : "hover:bg-slate-100"
                }`}
              >
                <img className="w-6 h-6" alt={icon.alt} src={icon.src} />
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col items-start gap-4">
          <div className="relative w-12 h-12 rounded-full bg-[url(https://c.animaapp.com/zPfth9Ad/img/avatar@2x.png)] bg-cover bg-center" />
        </div>
      </div>

      <NavigationSidebarSection
        sessions={sessions}
        activeSessionId={activeSession?.id || ""}
        sidebarTab={sidebarTab}
        onTabChange={setSidebarTab}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onUpload={handleUpload}
      />

      <ArticleDetailSection
        models={models}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        query={query}
        onQueryChange={setQuery}
        onSend={handleSendQuery}
        messages={activeSession?.messages || []}
        streamingAnswer={streamingAnswer}
        liveTrace={liveTrace}
        error={error}
        isStreaming={isStreaming}
      />

      {showSettings && appSettings && (
        <SettingsPanel
          settings={appSettings}
          models={models}
          onSave={handleSettingsSave}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
};
