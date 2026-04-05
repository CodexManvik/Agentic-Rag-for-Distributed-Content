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
              🍁 Aura
            </div>

            
          </div>

          <div className="flex flex-col w-[272px] items-end gap-1.5 relative flex-[0_0_auto]">
            <div className="flex flex-col items-end gap-1.5 relative self-stretch w-full flex-[0_0_auto]">
              
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
          <div>
            <div className="flex items-start gap-4 relative self-stretch w-full flex-[0_0_auto]">
              <div className="flex items-start gap-2.5 relative flex-1 grow">
                
              </div>

              
            </div>

            

            <div className="inline-flex items-start gap-4 relative flex-[0_0_auto]">
              <button
                onClick={() => setShowBanner(false)}
                className="all-[unset] box-border inline-flex gap-2 flex-[0_0_auto] items-center justify-center relative cursor-pointer"
              >
                
              </button>

              
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
