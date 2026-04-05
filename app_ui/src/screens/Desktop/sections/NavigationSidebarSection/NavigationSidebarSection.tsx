import { useState } from "react";

const navItems = [
  {
    label: "Home",
    iconSrc: "https://c.animaapp.com/zPfth9Ad/img/icon-1.svg",
    chevronSrc: "https://c.animaapp.com/zPfth9Ad/img/icon-4.svg",
  },
  {
    label: "Discover",
    iconSrc: "https://c.animaapp.com/zPfth9Ad/img/icon-3.svg",
    chevronSrc: "https://c.animaapp.com/zPfth9Ad/img/icon-4.svg",
  },
  {
    label: "Library",
    iconSrc: "https://c.animaapp.com/zPfth9Ad/img/icon-5.svg",
    chevronSrc: "https://c.animaapp.com/zPfth9Ad/img/icon-6.svg",
  },
];

interface LibraryItem {
  id: string;
  text: string;
  active: boolean;
}

interface NavigationSidebarSectionProps {
  libraryItems: LibraryItem[];
}

export const NavigationSidebarSection = ({
  libraryItems,
}: NavigationSidebarSectionProps): JSX.Element => {
  const [searchValue, setSearchValue] = useState("");
  const [showBanner, setShowBanner] = useState(true);

  return (
    <div className="flex flex-col w-80 h-[960px] items-start justify-between relative bg-slate-50 border-r [border-right-style:solid] border-slate-200">
      <div className="flex flex-col items-start relative self-stretch w-full flex-[0_0_auto]">
        <div className="flex flex-col items-start justify-center gap-4 px-6 py-8 relative self-stretch w-full flex-[0_0_auto] border-b [border-bottom-style:solid] border-slate-200">
          <div className="flex items-center gap-2 relative self-stretch w-full flex-[0_0_auto]">
            <div className="relative flex-1 [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-slate-800 text-2xl tracking-[-0.29px] leading-8">
              🌸 slothplexity
            </div>

            <div className="w-10 h-10 relative border border-solid border-slate-300 flex items-center justify-center gap-2.5 p-4 rounded-[123px] overflow-hidden">
              <img
                className="relative w-6 h-6 mt-[-8.00px] mb-[-8.00px] ml-[-8.00px] mr-[-8.00px]"
                alt="Monotone add"
                src="https://c.animaapp.com/zPfth9Ad/img/monotone-add-7.svg"
              />
            </div>
          </div>

          <div className="flex flex-col w-[272px] items-end gap-1.5 relative flex-[0_0_auto]">
            <div className="flex flex-col items-end gap-1.5 relative self-stretch w-full flex-[0_0_auto]">
              <div className="flex min-h-10 items-center gap-3 px-3 py-2 relative self-stretch w-full flex-[0_0_auto] bg-white rounded-[123px] overflow-hidden border border-solid border-slate-300">
                <input
                  className="relative grow border-[none] [background:none] flex-1 font-medium text-slate-600 text-base tracking-[-0.11px] leading-[22px] mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] p-0"
                  placeholder="Search anything..."
                  type="text"
                  value={searchValue}
                  onChange={(e) => setSearchValue(e.target.value)}
                />

                <img
                  className="relative w-5 h-5"
                  alt="Icon"
                  src="https://c.animaapp.com/zPfth9Ad/img/icon.svg"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-start pt-4 pb-0 px-0 relative self-stretch w-full flex-[0_0_auto]">
          {navItems.map((item) => (
            <div
              key={item.label}
              className="flex items-center gap-3 px-6 py-4 relative self-stretch w-full flex-[0_0_auto]"
            >
              <img className="relative w-6 h-6" alt="Icon" src={item.iconSrc} />

              <div className="relative flex-1 mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica] font-bold text-slate-800 text-lg tracking-[-0.14px] leading-6">
                {item.label}
              </div>

              <img
                className="relative w-6 h-6"
                alt="Icon"
                src={item.chevronSrc}
              />
            </div>
          ))}

          <div className="flex flex-col items-start pl-9 pr-6 py-2 relative self-stretch w-full flex-[0_0_auto]">
            {libraryItems.length === 0 ? (
              <div className="flex items-start gap-2.5 pl-4 pr-0 py-2 relative self-stretch w-full flex-[0_0_auto] border-l [border-left-style:solid] border-slate-400">
                <p className="relative flex-1 [font-family:'Plus_Jakarta_Sans',Helvetica] font-medium text-slate-600 text-base tracking-[-0.11px] leading-[22px] overflow-hidden text-ellipsis [display:-webkit-box] [-webkit-line-clamp:1] [-webkit-box-orient:vertical] mt-[-1.00px]">
                  No active queries yet.
                </p>
              </div>
            ) : (
              libraryItems.map((item) => (
                <div
                  key={item.id}
                  className={`flex items-start gap-2.5 pl-4 pr-0 py-2 relative self-stretch w-full flex-[0_0_auto] ${
                    item.active
                      ? "border-l-2 [border-left-style:solid] border-indigo-600"
                      : "border-l [border-left-style:solid] border-slate-400"
                  }`}
                >
                  <p
                    className={`relative flex-1 [font-family:'Plus_Jakarta_Sans',Helvetica] font-medium text-slate-600 text-base tracking-[-0.11px] leading-[22px] overflow-hidden text-ellipsis [display:-webkit-box] [-webkit-line-clamp:1] [-webkit-box-orient:vertical] ${
                      item.active ? "mt-[-2.00px]" : "mt-[-1.00px]"
                    }`}
                  >
                    {item.text}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col items-start gap-2.5 p-6 relative self-stretch w-full flex-[0_0_auto]">
        {showBanner && (
          <div className="flex flex-col items-start gap-4 p-4 relative self-stretch w-full flex-[0_0_auto] bg-white rounded-3xl overflow-hidden border border-solid border-slate-200">
            <div className="flex items-start gap-4 relative self-stretch w-full flex-[0_0_auto]">
              <div className="flex items-start gap-2.5 relative flex-1 grow">
                <div className="flex w-10 h-10 items-center justify-center gap-2.5 relative bg-slate-100 rounded-[123px]">
                  <img
                    className="relative w-5 h-5"
                    alt="Device mobile camera"
                    src="https://c.animaapp.com/zPfth9Ad/img/devicemobilecamera.svg"
                  />
                </div>
              </div>

              <button
                onClick={() => setShowBanner(false)}
                className="all-[unset] box-border cursor-pointer"
                aria-label="Dismiss banner"
              >
                <img
                  className="relative w-5 h-5"
                  alt="X"
                  src="https://c.animaapp.com/zPfth9Ad/img/x.svg"
                />
              </button>
            </div>

            <p className="relative self-stretch [font-family:'Plus_Jakarta_Sans',Helvetica] font-normal text-slate-600 text-sm tracking-[0] leading-[22.4px]">
              Download our app and receive extra 100 chats daily.
            </p>

            <div className="inline-flex items-start gap-4 relative flex-[0_0_auto]">
              <button
                onClick={() => setShowBanner(false)}
                className="all-[unset] box-border inline-flex gap-2 flex-[0_0_auto] items-center justify-center relative cursor-pointer"
              >
                <div className="w-fit font-bold text-slate-600 text-sm tracking-[-0.08px] leading-5 whitespace-nowrap relative mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica]">
                  Dismiss
                </div>
              </button>

              <button className="all-[unset] box-border inline-flex gap-2 flex-[0_0_auto] items-center justify-center relative cursor-pointer">
                <div className="w-fit font-bold text-indigo-600 text-sm tracking-[-0.08px] leading-5 whitespace-nowrap relative mt-[-1.00px] [font-family:'Plus_Jakarta_Sans',Helvetica]">
                  Download App
                </div>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
