import type { StreamCitation, StreamTraceEvent } from "../../../../api";

const sidebarActions = [
  {
    icon: "https://c.animaapp.com/zPfth9Ad/img/play.svg",
    alt: "Play",
    label: "Search Videos",
  },
  {
    icon: "https://c.animaapp.com/zPfth9Ad/img/imagesquare.svg",
    alt: "Image square",
    label: "Generate Image",
  },
  {
    icon: "https://c.animaapp.com/zPfth9Ad/img/airplaneinflight.svg",
    alt: "Airplane in flight",
    label: "Book Tickets",
  },
  {
    icon: "https://c.animaapp.com/zPfth9Ad/img/graduationcap.svg",
    alt: "Graduation cap",
    label: "Learn & Educate",
  },
];

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
    <div className="flex flex-col w-[1040px] items-start relative mb-[-194.00px] bg-white">
      <div className="flex items-center gap-4 px-6 py-2 relative self-stretch w-full flex-[0_0_auto] border-b [border-bottom-style:solid] border-slate-200">
        <div className="flex items-center gap-2 relative flex-1 grow">
          <img
            className="relative w-5 h-5"
            alt="User"
            src="https://c.animaapp.com/zPfth9Ad/img/user.svg"
          />
          <div className="relative w-fit mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-slate-800 text-base tracking-[-0.11px] leading-[22px] whitespace-nowrap">
            x-ae-a-221b
          </div>
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

            <div className="relative w-5 h-5 bg-[url(https://c.animaapp.com/zPfth9Ad/img/caretright-2.svg)] bg-[100%_100%]" />

            <div className="gap-1 inline-flex items-center relative flex-[0_0_auto]">
              <div className="relative w-fit mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-slate-600 text-sm text-center tracking-[-0.08px] leading-5 whitespace-nowrap">
                Private
              </div>
            </div>

            <div className="relative w-5 h-5 bg-[url(https://c.animaapp.com/zPfth9Ad/img/caretright-2.svg)] bg-[100%_100%]" />

            <div className="inline-flex items-center gap-1 relative flex-[0_0_auto]">
              <div className="relative w-fit mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-slate-600 text-sm text-center tracking-[-0.08px] leading-5 whitespace-nowrap">
                Notes
              </div>
            </div>

            <div className="relative w-5 h-5 bg-[url(https://c.animaapp.com/zPfth9Ad/img/caretright-2.svg)] bg-[100%_100%]" />

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

        <div className="flex items-center justify-end gap-4 relative flex-1 grow">
          <div className="inline-flex items-center gap-1 relative flex-[0_0_auto]">
            <img
              className="relative w-4 h-4"
              alt="Clock"
              src="https://c.animaapp.com/zPfth9Ad/img/clock.svg"
            />
            <div className="relative w-fit mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-medium text-slate-800 text-xs tracking-[-0.06px] leading-4 whitespace-nowrap">
              22 hours ago
            </div>
          </div>

          <button className="all-[unset] box-border inline-flex gap-2 px-3 py-1.5 flex-[0_0_auto] bg-indigo-600 rounded-[1234px] overflow-hidden items-center justify-center relative cursor-pointer">
            <img
              className="relative w-4 h-4"
              alt="Share fat"
              src="https://c.animaapp.com/zPfth9Ad/img/sharefat.svg"
            />
            <div className="w-fit font-bold text-white text-sm tracking-[-0.08px] leading-5 whitespace-nowrap relative mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Share
            </div>
          </button>
        </div>
      </div>

      <div className="flex w-[1040px] items-start px-24 py-0 relative flex-[0_0_auto]">
        <div className="inline-flex flex-col items-start justify-center gap-8 px-6 py-8 relative flex-[0_0_auto] border-r [border-right-style:solid] border-l [border-left-style:solid] border-slate-200">
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

        <div className="flex flex-col w-[286px] items-start px-0 py-8 relative self-stretch border-r [border-right-style:solid] border-slate-200">
          <div className="flex flex-col items-start relative self-stretch w-full flex-[0_0_auto] border-b [border-bottom-style:solid] border-slate-200">
            <div className="flex items-center gap-4 px-6 py-3 relative self-stretch w-full flex-[0_0_auto]">
              <div className="flex items-center gap-2 relative flex-1 grow">
                <img
                  className="relative w-6 h-6"
                  alt="Info"
                  src="https://c.animaapp.com/zPfth9Ad/img/info.svg"
                />
                <div className="relative flex-1 [font-family:'Plus_Jakarta_Sans',Helvetica] font-semibold text-slate-800 text-base tracking-[-0.11px] leading-[22px]">
                  Overview
                </div>
              </div>

              <button className="all-[unset] box-border inline-flex gap-2 flex-[0_0_auto] items-center justify-center relative cursor-pointer">
                <div className="w-fit font-bold text-indigo-600 text-sm tracking-[-0.08px] leading-5 whitespace-nowrap relative mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica]">
                  View All
                </div>
              </button>
            </div>

            <img
              className="relative self-stretch w-full flex-[0_0_auto]"
              alt="Frame"
              src="https://c.animaapp.com/zPfth9Ad/img/frame.svg"
            />
          </div>

          {sidebarActions.map((action, index) => (
            <div
              key={index}
              className="flex h-12 items-center gap-4 px-6 py-3 relative self-stretch w-full border-b [border-bottom-style:solid] border-slate-200"
            >
              <div className="flex items-center gap-2 relative flex-1 grow">
                <img
                  className="relative w-6 h-6"
                  alt={action.alt}
                  src={action.icon}
                />
                <div className="relative flex-1 [font-family:'Plus_Jakarta_Sans',Helvetica] font-semibold text-slate-800 text-base tracking-[-0.11px] leading-[22px]">
                  {action.label}
                </div>
              </div>
              <img
                className="relative w-5 h-5"
                alt="Plus"
                src="https://c.animaapp.com/zPfth9Ad/img/plus-3.svg"
              />
            </div>
          ))}
        </div>
      </div>

      <div className="w-12 h-12 absolute right-6 bottom-6 bg-slate-800 shadow-[0px_4px_6px_-2px_#10182808,0px_12px_16px_-4px_#10182814] flex items-center justify-center gap-2.5 p-4 rounded-[123px] overflow-hidden">
        <img
          className="relative w-6 h-6 mt-[-4.00px] mb-[-4.00px] ml-[-4.00px] mr-[-4.00px]"
          alt="Monotone add"
          src="https://c.animaapp.com/zPfth9Ad/img/monotone-add-10.svg"
        />
      </div>
    </div>
  );
};