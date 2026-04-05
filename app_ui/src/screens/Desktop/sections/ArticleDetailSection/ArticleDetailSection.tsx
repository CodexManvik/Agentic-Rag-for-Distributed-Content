import type { StreamCitation, StreamTraceEvent } from "../../../../api";

interface ArticleDetailSectionProps {
  models: string[];
  selectedModel: string;
  onModelChange: (model: string) => void;
  query: string;
  onQueryChange: (value: string) => void;
  onSend: () => void;
  answer: string;
  sources: StreamCitation[];
  trace: StreamTraceEvent[];
  error: string | null;
  isStreaming: boolean;
}

export const ArticleDetailSection = ({
  models,
  selectedModel,
  onModelChange,
  query,
  onQueryChange,
  onSend,
  answer,
  sources,
  trace,
  error,
  isStreaming,
}: ArticleDetailSectionProps): JSX.Element => {
  return (
    <div className="flex flex-col w-full items-start relative mb-[-194.00px] bg-white">
      <div className="flex items-center gap-4 px-6 py-2 relative self-stretch w-full flex-[0_0_auto] border-b [border-bottom-style:solid] border-slate-200">
        <div className="flex items-center gap-2 relative flex-1 grow">
          
          
        </div>

        <div className="flex flex-col items-center gap-2.5 relative flex-1 grow">
          <div className="gap-2 inline-flex items-center relative flex-[0_0_auto]">
            <div className="w-8 h-8 relative flex items-center justify-center gap-2.5 p-4 rounded-[123px] overflow-hidden">
              <img
                className="relative w-5 h-5 mt-[-10.00px] mb-[-10.00px] ml-[-10.00px] mr-[-10.00px]"
                alt="Monotone add"
                src="https://c.animaapp.com/zPfth9Ad/img/monotone-add-8.svg"
              />
            </div>


            <div className="gap-1 inline-flex items-center relative flex-[0_0_auto]">
              
            </div>


            <div className="inline-flex items-center gap-1 relative flex-[0_0_auto]">
              
            </div>


            <div className="inline-flex items-center gap-1 relative flex-[0_0_auto]">
              <select
                className="relative w-fit mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-indigo-600 text-sm text-center tracking-[-0.08px] leading-5 whitespace-nowrap bg-transparent border-0 outline-none"
                value={selectedModel}
                onChange={(e) => onModelChange(e.target.value)}
              >
                {models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        
      </div>

      <div className="flex w-full items-start px-24 py-0 relative flex-[0_0_auto]">
        <div className="inline-flex flex-col items-start justify-center gap-8 px-6 py-8 relative flex-[0_0_auto] border-l [border-left-style:solid] border-slate-200 flex-1">
          <p className="relative w-[514px] mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-slate-800 text-3xl tracking-[-0.39px] leading-[38px]">
            Do androids dream of electric sheep or not? Explained in ELI5
          </p>

          <div className="flex flex-col w-[514px] items-start gap-3 relative flex-[0_0_auto]">
            <div className="inline-flex items-center gap-2 relative flex-[0_0_auto]">
              <img
                className="relative w-6 h-6"
                alt="Icon"
                src="https://c.animaapp.com/zPfth9Ad/img/icon-7.svg"
              />
              <div className="relative w-fit [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-slate-800 text-base tracking-[-0.11px] leading-[22px] whitespace-nowrap">
                Sources
              </div>
            </div>

            <div className="flex items-center gap-2 relative self-stretch w-full flex-[0_0_auto]">
              {sources.length === 0 ? (
                <div className="flex flex-col items-start gap-4 p-3 relative bg-slate-50 rounded-2xl overflow-hidden border border-solid border-slate-200 w-full">
                  <p className="relative self-stretch mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-medium text-slate-800 text-xs tracking-[-0.06px] leading-4">
                    Sources will appear after backend retrieval.
                  </p>
                </div>
              ) : (
                sources.map((source, index) => (
                  <div
                    key={`${source.index}-${index}`}
                    className={`flex flex-col items-start gap-4 p-3 relative bg-slate-50 rounded-2xl overflow-hidden border border-solid border-slate-200 ${index === 0 ? "w-[157px]" : "flex-1 grow"}`}
                  >
                    <p className="relative self-stretch mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-medium text-slate-800 text-xs tracking-[-0.06px] leading-4 overflow-hidden text-ellipsis [display:-webkit-box] [-webkit-line-clamp:2] [-webkit-box-orient:vertical]">
                      {source.snippet || source.source}
                    </p>
                    <div className="inline-flex items-center gap-1 relative flex-[0_0_auto]">
                      <img
                        className="relative w-4 h-4"
                        alt="Globe simple"
                        src="https://c.animaapp.com/zPfth9Ad/img/globesimple-2.svg"
                      />
                      <div className="relative w-fit [font-family:'Plus_Jakarta_Sans',Helvetica] font-medium text-slate-600 text-[10px] tracking-[-0.04px] leading-[14px] whitespace-nowrap overflow-hidden text-ellipsis max-w-[140px]">
                        {source.source}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="flex flex-col w-[514px] items-start gap-3 relative flex-[0_0_auto]">
            <div className="inline-flex items-center gap-2 relative flex-[0_0_auto]">
              <img
                className="relative w-6 h-6"
                alt="Icon"
                src="https://c.animaapp.com/zPfth9Ad/img/icon-8.svg"
              />
              <div className="relative w-fit [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-slate-800 text-base tracking-[-0.11px] leading-[22px] whitespace-nowrap">
                Answer
              </div>
            </div>

            <p className="relative self-stretch [font-family:'Plus_Jakarta_Sans',Helvetica] font-normal text-slate-600 text-sm tracking-[0] leading-[22.4px] whitespace-pre-wrap">
              {answer}
            </p>

            {trace.length > 0 && (
              <div className="relative self-stretch [font-family:'Plus_Jakarta_Sans',Helvetica] font-normal text-slate-600 text-xs tracking-[0] leading-[18px]">
                {trace.map((event, index) => (
                  <p key={`${event.node}-${index}`}>{`${event.node}: ${event.detail}`}</p>
                ))}
              </div>
            )}
          </div>

          <div className="flex items-center gap-3 relative self-stretch w-full flex-[0_0_auto]">
            <div className="flex items-center gap-2.5 p-3 relative flex-1 grow bg-white rounded-[1234px] border border-solid border-slate-300 shadow-[0px_2px_4px_-2px_#1717170f,0px_4px_8px_-2px_#1717171a]">
              <div className="flex items-center gap-2 relative flex-1 grow">
                <img
                  className="relative w-6 h-6"
                  alt="Plus circle"
                  src="https://c.animaapp.com/zPfth9Ad/img/pluscircle.svg"
                />
                <input
                  className="relative grow border-[none] [background:none] flex-1 [font-family:'Plus_Jakarta_Sans',Helvetica] font-normal text-slate-600 text-base tracking-[0] leading-[25.6px] mt-[-1.00px] p-0 outline-none"
                  placeholder="Ask a follow up..."
                  value={query}
                  onChange={(e) => onQueryChange(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      onSend();
                    }
                  }}
                  disabled={isStreaming}
                />
              </div>

              <div className="inline-flex items-center gap-3 relative flex-[0_0_auto]">
                <img
                  className="relative w-6 h-6"
                  alt="Microphone"
                  src="https://c.animaapp.com/zPfth9Ad/img/microphone.svg"
                />
                <img
                  className="relative w-6 h-6"
                  alt="Paperclip"
                  src="https://c.animaapp.com/zPfth9Ad/img/paperclip.svg"
                />
              </div>
            </div>

            <button
              onClick={onSend}
              disabled={isStreaming}
              className="all-[unset] box-border w-10 h-10 relative bg-indigo-600 flex items-center justify-center gap-2.5 p-4 rounded-[123px] overflow-hidden cursor-pointer"
            >
              <img
                className="relative w-6 h-6 mt-[-8.00px] mb-[-8.00px] ml-[-8.00px] mr-[-8.00px]"
                alt="Monotone add"
                src="https://c.animaapp.com/zPfth9Ad/img/monotone-add-9.svg"
              />
            </button>
          </div>

          {error && (
            <div className="relative w-[514px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-medium text-red-600 text-sm tracking-[0] leading-[22.4px]">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};