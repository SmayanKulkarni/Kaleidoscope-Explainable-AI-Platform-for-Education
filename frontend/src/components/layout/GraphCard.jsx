import { Suspense, useState } from 'react';

export default function GraphCard({ title, legend, children, height = 420 }) {
  const [fullscreen, setFullscreen] = useState(false);

  return (
    <div className={`bg-surface-container-lowest rounded-2xl border border-outline-variant/10 overflow-hidden flex flex-col ${fullscreen ? 'fixed inset-4 z-50 shadow-2xl' : ''}`}>
      <div className="flex items-center justify-between px-6 py-4 border-b border-outline-variant/10">
        <h3 className="font-headline font-bold text-base text-on-surface">{title}</h3>
        <div className="flex items-center gap-4">
          {legend && (
            <div className="hidden sm:flex items-center gap-3">
              {legend.map((item) => (
                <div key={item.label} className="flex items-center gap-1.5 text-xs font-label text-slate-500">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: item.color }} />
                  {item.label}
                </div>
              ))}
            </div>
          )}
          <button
            onClick={() => setFullscreen((v) => !v)}
            title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            className="material-symbols-outlined text-slate-400 hover:text-primary text-xl"
          >
            {fullscreen ? 'close_fullscreen' : 'open_in_full'}
          </button>
        </div>
      </div>
      <div className="flex-1 relative" style={{ height: fullscreen ? undefined : height }}>
        <Suspense fallback={
          <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm font-label">
            <span className="material-symbols-outlined animate-spin mr-2">refresh</span> Loading 3D graph…
          </div>
        }>
          {children}
        </Suspense>
      </div>
    </div>
  );
}
