import { useEffect, useRef, useState } from "react";
import {
  fetchModels,
  streamChat,
  type StreamCitation,
  type StreamCompletePayload,
  type StreamTraceEvent,
} from "../../api";
import { ArticleDetailSection } from "./sections/ArticleDetailSection";
import { NavigationSidebarSection } from "./sections/NavigationSidebarSection";

export const Desktop = (): JSX.Element => {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<StreamCitation[]>([]);
  const [trace, setTrace] = useState<StreamTraceEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [libraryItems, setLibraryItems] = useState<Array<{ id: string; text: string; active: boolean }>>([]);
  const streamRef = useRef<EventSource | null>(null);

  useEffect(() => {
    let mounted = true;
    const loadModels = async () => {
      try {
        const modelList = await fetchModels();
        if (!mounted) {
          return;
        }
        setModels(modelList);
        setSelectedModel(modelList[0]);
        setError(null);
      } catch (err) {
        if (!mounted) {
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load models");
      }
    };
    loadModels();

    return () => {
      mounted = false;
      if (streamRef.current) {
        streamRef.current.close();
      }
    };
  }, []);

  const handleStreamComplete = (payload: StreamCompletePayload) => {
    setAnswer(payload.answer || "");
    setSources(Array.isArray(payload.citations) ? payload.citations : []);
    setTrace(Array.isArray(payload.trace) ? payload.trace : []);
    if (Array.isArray(payload.sub_queries)) {
      setLibraryItems((prev) => {
        const fromSubQueries = payload.sub_queries.map((item, index) => ({
          id: `sub-${Date.now()}-${index}`,
          text: item,
          active: false,
        }));
        return [...prev.map((item) => ({ ...item, active: false })), ...fromSubQueries].slice(-16);
      });
    }
    setIsStreaming(false);
    streamRef.current = null;
  };

  const handleSendQuery = () => {
    if (!query.trim()) {
      setError("Please enter a query before sending.");
      return;
    }
    if (!selectedModel) {
      setError("No model selected. Check backend model availability.");
      return;
    }

    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }

    setError(null);
    setAnswer("");
    setSources([]);
    setTrace([]);
    setIsStreaming(true);
    setLibraryItems((prev) => [
      { id: `q-${Date.now()}`, text: query.trim(), active: true },
      ...prev.map((item) => ({ ...item, active: false })),
    ].slice(0, 16));

    try {
      streamRef.current = streamChat(
        query,
        {
          onChunk: (text) => {
            setAnswer((prev) => prev + text);
          },
          onTrace: (event) => {
            setTrace((prev) => [...prev, event]);
          },
          onComplete: (payload) => {
            handleStreamComplete(payload);
          },
          onError: (err) => {
            setError(err.message);
            setIsStreaming(false);
            if (streamRef.current) {
              streamRef.current.close();
              streamRef.current = null;
            }
          },
        },
        selectedModel,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start stream");
      setIsStreaming(false);
    }
  };

  const navIcons = [
    {
      src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add.svg",
      alt: "Monotone add",
    },
    {
      src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-1.svg",
      alt: "Monotone add",
    },
    {
      src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-2.svg",
      alt: "Monotone add",
    },
    {
      src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-3.svg",
      alt: "Monotone add",
    },
    {
      src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-4.svg",
      alt: "Monotone add",
    },
    {
      src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-5.svg",
      alt: "Monotone add",
    },
  ];

  const bottomNavIcons = [
    {
      src: "https://c.animaapp.com/zPfth9Ad/img/monotone-add-6.svg",
      alt: "Monotone add",
    },
  ];

  return (
    <div
      className="flex h-[960px] items-start relative bg-white w-full min-w-[1440px]"
      data-model-id="10301:23262"
    >
      <div className="inline-flex flex-col items-start gap-[368px] px-4 py-6 relative flex-[0_0_auto] bg-slate-50 border-r [border-right-style:solid] border-slate-200">
        <div className="inline-flex flex-col items-center gap-8 relative flex-[0_0_auto]">
          <img
            className="relative w-12 h-12 mt-[-4.00px]"
            alt="Logomark"
            src="https://c.animaapp.com/zPfth9Ad/img/logomark.svg"
          />

          <div className="inline-flex flex-col items-start gap-4 relative flex-[0_0_auto]">
            {navIcons.map((icon, index) => (
              <div
                key={index}
                className={`${index === 0 ? "w-12 h-12 relative bg-white flex items-center justify-center gap-2.5 p-4 rounded-[123px] overflow-hidden" : "flex w-12 h-12 gap-2.5 p-4 rounded-[123px] overflow-hidden items-center justify-center relative"}`}
              >
                <img
                  className="relative w-6 h-6 mt-[-4.00px] mb-[-4.00px] ml-[-4.00px] mr-[-4.00px]"
                  alt={icon.alt}
                  src={icon.src}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="inline-flex flex-col items-start gap-4 relative flex-[0_0_auto]">
          {bottomNavIcons.map((icon, index) => (
            <div
              key={index}
              className="flex w-12 h-12 gap-2.5 p-4 rounded-[123px] overflow-hidden items-center justify-center relative"
            >
              <img
                className="relative w-6 h-6 mt-[-4.00px] mb-[-4.00px] ml-[-4.00px] mr-[-4.00px]"
                alt={icon.alt}
                src={icon.src}
              />
            </div>
          ))}

          <div className="relative w-12 h-12 rounded-[92.25px] bg-[url(https://c.animaapp.com/zPfth9Ad/img/avatar@2x.png)] bg-cover bg-[50%_50%]" />
        </div>
      </div>

      <NavigationSidebarSection libraryItems={libraryItems} />
      <ArticleDetailSection
        models={models}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        query={query}
        onQueryChange={setQuery}
        onSend={handleSendQuery}
        answer={answer}
        sources={sources}
        trace={trace}
        error={error}
        isStreaming={isStreaming}
      />
    </div>
  );
};
