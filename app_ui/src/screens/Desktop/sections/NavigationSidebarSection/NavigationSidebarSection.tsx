import { useState, useRef, useCallback, useEffect } from "react";
import type { ChatSession } from "../../../../api";

interface NavigationSidebarSectionProps {
  sessions: ChatSession[];
  activeSessionId: string;
  sidebarTab: "history" | "upload";
  onTabChange: (tab: "history" | "upload") => void;
  onNewChat: () => void;
  onSelectSession: (session: ChatSession) => void;
  onDeleteSession: (id: string) => void;
  onUpload: (file: File) => Promise<{ chunks_added: number }>;
}

export const NavigationSidebarSection = ({
  sessions,
  activeSessionId,
  sidebarTab,
  onTabChange,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onUpload,
}: NavigationSidebarSectionProps): JSX.Element => {
  const [uploadState, setUploadState] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [uploadMessage, setUploadMessage] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadResetTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (uploadResetTimerRef.current !== null) {
        window.clearTimeout(uploadResetTimerRef.current);
      }
    };
  }, []);

  const handleFile = useCallback(
    async (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!["pdf", "txt", "md"].includes(ext || "")) {
        setUploadState("error");
        setUploadMessage("Only PDF, TXT, and MD files are supported.");
        return;
      }
      setUploadState("uploading");
      setUploadMessage(`Ingesting ${file.name}…`);
      try {
        const result = await onUpload(file);
        setUploadState("done");
        setUploadMessage(`✓ ${file.name} — ${result.chunks_added} chunks added`);
        if (uploadResetTimerRef.current !== null) {
          window.clearTimeout(uploadResetTimerRef.current);
        }
        uploadResetTimerRef.current = window.setTimeout(() => setUploadState("idle"), 4000);
      } catch (err) {
        setUploadState("error");
        setUploadMessage(err instanceof Error ? err.message : "Upload failed");
      }
    },
    [onUpload],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const formatRelativeTime = (ts: number) => {
    const diff = Date.now() - ts;
    if (diff < 60000) return "just now";
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return new Date(ts).toLocaleDateString();
  };

  return (
    <div className="flex flex-col w-80 h-screen items-start justify-between bg-slate-50 border-r border-slate-200 flex-shrink-0">
      {/* Header */}
      <div className="flex flex-col items-start w-full">
        <div className="flex flex-col items-start justify-center gap-4 px-6 py-6 w-full border-b border-slate-200">
          <div className="flex items-center justify-between w-full">
            <div className="font-bold text-slate-800 text-2xl tracking-[-0.29px] leading-8 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              🍁 Aura
            </div>
            <button
              onClick={onNewChat}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 text-white rounded-full text-xs font-semibold [font-family:'Plus_Jakarta_Sans',Helvetica] hover:bg-indigo-700 transition-colors"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 4v16m8-8H4" />
              </svg>
              New chat
            </button>
          </div>

          {/* Tab switcher */}
          <div className="flex w-full bg-slate-200 rounded-lg p-0.5 gap-0.5">
            {(["history", "upload"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => onTabChange(tab)}
                className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all [font-family:'Plus_Jakarta_Sans',Helvetica] ${
                  sidebarTab === tab
                    ? "bg-white text-slate-800 shadow-sm"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {tab === "history" ? "History" : "Upload"}
              </button>
            ))}
          </div>
        </div>

        {sidebarTab === "history" ? (
          <div className="flex flex-col w-full overflow-y-auto max-h-[calc(100vh-160px)]">
            {sessions.length === 0 ? (
              <div className="px-6 py-8 text-center">
                <div className="text-3xl mb-2">💬</div>
                <p className="text-slate-500 text-sm [font-family:'Plus_Jakarta_Sans',Helvetica]">
                  No chats yet. Ask something!
                </p>
              </div>
            ) : (
              <div className="flex flex-col pt-2 pb-4">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={`group flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors ${
                      session.id === activeSessionId ? "bg-indigo-50 border-l-2 border-indigo-600" : "hover:bg-slate-100 border-l-2 border-transparent"
                    }`}
                    onClick={() => onSelectSession(session)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onSelectSession(session);
                      }
                    }}
                    role="button"
                    tabIndex={0}
                    aria-current={session.id === activeSessionId ? "page" : undefined}
                  >
                    <div className="flex-1 min-w-0">
                      <p
                        className={`text-sm font-medium truncate [font-family:'Plus_Jakarta_Sans',Helvetica] ${
                          session.id === activeSessionId ? "text-indigo-700" : "text-slate-800"
                        }`}
                      >
                        {session.title || "Untitled"}
                      </p>
                      <p className="text-xs text-slate-400 mt-0.5 [font-family:'Plus_Jakarta_Sans',Helvetica]">
                        {session.messages.length} msg · {formatRelativeTime(session.updatedAt)}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(session.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 transition-opacity p-1 rounded hover:bg-red-100 text-slate-400 hover:text-red-500"
                      aria-label={`Delete session ${session.title || "Untitled"}`}
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col w-full p-6 gap-4">
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setIsDragging(true);
                  fileInputRef.current?.click();
                }
              }}
              onKeyUp={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  setIsDragging(false);
                }
              }}
              onFocus={() => setIsDragging(true)}
              onBlur={() => setIsDragging(false)}
              role="button"
              tabIndex={0}
              className={`flex flex-col items-center justify-center gap-3 p-6 rounded-2xl border-2 border-dashed cursor-pointer transition-all ${
                isDragging
                  ? "border-indigo-400 bg-indigo-50"
                  : "border-slate-300 bg-white hover:border-indigo-300 hover:bg-slate-50"
              }`}
            >
              <div className="w-12 h-12 rounded-full bg-indigo-50 flex items-center justify-center">
                <svg className="w-6 h-6 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
                  Drop files here
                </p>
                <p className="text-xs text-slate-400 mt-1 [font-family:'Plus_Jakarta_Sans',Helvetica]">
                  PDF, TXT, MD — click or drag
                </p>
              </div>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.txt,.md"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
                e.target.value = "";
              }}
            />

            {/* Upload status */}
            {uploadState !== "idle" && (
              <div
                className={`flex items-center gap-2.5 p-3 rounded-xl text-sm [font-family:'Plus_Jakarta_Sans',Helvetica] ${
                  uploadState === "uploading"
                    ? "bg-indigo-50 text-indigo-700"
                    : uploadState === "done"
                    ? "bg-emerald-50 text-emerald-700"
                    : "bg-red-50 text-red-600"
                }`}
              >
                {uploadState === "uploading" && (
                  <svg className="w-4 h-4 animate-spin flex-shrink-0" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                )}
                <span className="font-medium">{uploadMessage}</span>
              </div>
            )}

            <div className="mt-2">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider [font-family:'Plus_Jakarta_Sans',Helvetica] mb-2">
                Supported formats
              </p>
              {[
                { ext: "PDF", desc: "Research papers, docs" },
                { ext: "TXT", desc: "Plain text, logs" },
                { ext: "MD", desc: "Markdown notes" },
              ].map((f) => (
                <div key={f.ext} className="flex items-center gap-2.5 py-1.5">
                  <span className="text-[10px] font-bold bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded [font-family:'Plus_Jakarta_Sans',Helvetica]">
                    {f.ext}
                  </span>
                  <span className="text-xs text-slate-500 [font-family:'Plus_Jakarta_Sans',Helvetica]">{f.desc}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};