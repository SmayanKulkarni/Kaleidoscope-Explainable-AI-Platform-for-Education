import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useAuth } from '../context/AuthContext';
import { getMyHistory, getLearnerHistory } from '../api/history';
import { getInstructorStudents } from '../api/recommend';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { track } from '../services/eventTracker';
import { PAGE_VIEW, HISTORY_VIEWED } from '../constants/eventTypes';

// ── helpers ────────────────────────────────────────────────────────────────────
function fmt(isoStr) {
  if (!isoStr) return '—';
  const d = new Date(isoStr);
  return d.toLocaleString('en-GB', {
    day: 'numeric', month: 'long', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function pct(v) {
  if (v == null) return null;
  return Math.round(Math.abs(v) * 100);
}

// ── SnapshotCard ───────────────────────────────────────────────────────────────
function SnapshotCard({ snap, index }) {
  const shapEntries = Object.entries(snap.shap_values ?? {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 4);

  const hasDrift = !!snap.drift_detected;
  const isCritical = hasDrift && snap.drift_delta != null && snap.drift_delta > 0.3;

  let statusGradient, borderClass, iconColor, iconName, bgClass;
  if (isCritical) {
    statusGradient = 'linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(153,27,27,0.05) 100%)';
    borderClass = 'border-error/30 group-hover:border-error/50 group-hover:shadow-[0_8px_30px_rgba(239,68,68,0.15)]';
    iconColor = 'text-error';
    iconName = 'report';
    bgClass = 'bg-error/20 border-error/50 shadow-[0_0_15px_rgba(239,68,68,0.4)]';
  } else if (hasDrift) {
    statusGradient = 'linear-gradient(135deg, rgba(245,158,11,0.15) 0%, rgba(180,83,9,0.05) 100%)';
    borderClass = 'border-amber-500/30 group-hover:border-amber-500/50 group-hover:shadow-[0_8px_30px_rgba(245,158,11,0.15)]';
    iconColor = 'text-amber-400';
    iconName = 'warning';
    bgClass = 'bg-amber-500/20 border-amber-500/50 shadow-[0_0_15px_rgba(245,158,11,0.4)]';
  } else {
    statusGradient = 'linear-gradient(135deg, rgba(13,148,136,0.1) 0%, rgba(15,118,110,0.05) 100%)';
    borderClass = 'border-primary/20 group-hover:border-primary/50 group-hover:shadow-[0_8px_30px_rgba(13,148,136,0.15)]';
    iconColor = 'text-primary';
    iconName = 'camera';
    bgClass = 'bg-primary/20 border-primary/50 shadow-[0_0_15px_rgba(13,148,136,0.4)]';
  }

  return (
    <div className="flex gap-8 group relative z-10 w-full">
      {/* Timeline indicator node */}
      <div className="relative z-20 flex-shrink-0 mt-4">
        <div
          className={`w-12 h-12 rounded-full border-2 flex items-center justify-center backdrop-blur-md
            group-hover:scale-110 transition-all duration-300 ${bgClass}`}
        >
          <span
            className={`material-symbols-outlined ${iconColor} drop-shadow-md transition-transform group-hover:rotate-12`}
            style={{ fontVariationSettings: "'FILL' 1" }}
          >
            {iconName}
          </span>
        </div>
      </div>

      {/* Card body */}
      <div className="flex-1 pb-12 w-full max-w-4xl">
        <div
          className={`relative overflow-hidden rounded-2xl p-7 transition-all duration-500 cursor-default
            border backdrop-blur-xl ${borderClass}`}
          style={{ background: statusGradient }}
        >
          {/* Subtle glow hover effect */}
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none"
               style={{ background: 'radial-gradient(circle at top right, rgba(255,255,255,0.08), transparent 60%)' }} />

          {/* Header row */}
          <div className="flex justify-between items-start mb-6 gap-4 relative z-10">
            <div>
              <div className="flex items-center gap-3 mb-1">
                <h3 className="text-xl font-headline font-black text-on-surface tracking-tight">
                  Snapshot #{index + 1}
                </h3>
              </div>
              <p className="text-sm font-medium text-slate-400 flex items-center gap-1.5 bg-surface-container-low/50 w-fit px-3 py-1 rounded-full border border-outline-variant/10">
                <span className="material-symbols-outlined text-[14px]">event</span>
                {fmt(snap.created_at)}
              </p>
            </div>

            <div className="flex gap-2 flex-wrap justify-end items-center">
              {snap.model_version && (
                <span className="bg-surface/50 text-on-surface-variant text-[11px] px-3 py-1.5 rounded-full font-bold uppercase tracking-wider border border-outline-variant/30 shadow-sm backdrop-blur-md flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[14px]">memory</span>
                  V_{snap.model_version}
                </span>
              )}
              {isCritical && (
                <span className="bg-error/20 text-error-container text-xs px-3 py-1.5 rounded-full font-black uppercase tracking-widest flex items-center gap-1 border border-error/30 shadow-[0_0_10px_rgba(239,68,68,0.2)]">
                  <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>warning</span>
                  CRITICAL DRIFT
                </span>
              )}
              {hasDrift && !isCritical && (
                <span className="bg-amber-500/20 text-amber-300 text-xs px-3 py-1.5 rounded-full font-black uppercase tracking-widest border border-amber-500/30 flex items-center gap-1 shadow-[0_0_10px_rgba(245,158,11,0.2)]">
                  <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>warning</span>
                  FEATURE DRIFT
                </span>
              )}
              {!hasDrift && (
                <span className="bg-primary/20 text-primary-fixed text-xs px-3 py-1.5 rounded-full font-black uppercase tracking-widest border border-primary/30 flex items-center gap-1 shadow-[0_0_10px_rgba(13,148,136,0.2)]">
                  <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>verified</span>
                  STABLE
                </span>
              )}
            </div>
          </div>

          {/* Body grid */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-8 relative z-10">
            {/* SHAP bars */}
            <div className="md:col-span-8 bg-surface-container-low/40 p-5 rounded-xl border border-outline-variant/10 shadow-inner">
              <div className="flex items-center gap-2 mb-4">
                <span className="material-symbols-outlined text-primary text-sm">bar_chart</span>
                <p className="text-[11px] font-bold text-slate-300 uppercase tracking-widest">
                  Key Influencing Features
                </p>
              </div>
              
              {shapEntries.length > 0 ? (
                <div className="space-y-4">
                  {shapEntries.map(([feat, val]) => {
                    const positive = val >= 0;
                    const width = pct(val);
                    const colorClass = positive ? 'bg-primary' : 'bg-tertiary';
                    const textClass = positive ? 'text-primary-fixed' : 'text-tertiary-fixed-dim';
                    
                    return (
                      <div key={feat} className="group/bar">
                        <div className="flex justify-between items-end mb-1.5">
                          <span className="text-sm font-medium text-on-surface-variant truncate max-w-[70%] group-hover/bar:text-on-surface transition-colors">
                            {feat.replace(/_/g, ' ')}
                          </span>
                          <span className={`font-black text-sm drop-shadow-sm ${textClass}`}>
                            {positive ? '+' : '-'}{width}%
                          </span>
                        </div>
                        <div className="w-full bg-surface-container/80 h-2.5 rounded-full overflow-hidden border border-outline-variant/5 shadow-inner">
                          <div
                            className={`h-full rounded-full transition-all duration-1000 ease-out relative ${colorClass}`}
                            style={{ width: `${Math.min(width, 100)}%` }}
                          >
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent to-white/20"></div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-6 text-slate-500 italic bg-surface-container/30 rounded-lg border border-outline-variant/5">
                  <span className="material-symbols-outlined text-3xl mb-2 opacity-50">data_loss_prevention</span>
                  <p className="text-sm">No SHAP values recorded</p>
                </div>
              )}
            </div>

            {/* Risk + Drift summary */}
            <div className="md:col-span-4 flex flex-col gap-4">
              {snap.risk_score != null && (
                <div className={`p-5 rounded-xl border relative overflow-hidden flex flex-col justify-center h-full min-h-[120px] transition-colors
                  ${isCritical ? 'bg-error/10 border-error/30' : 'bg-surface-container-low/50 border-outline-variant/15'}`}>
                  {/* Background decoration */}
                  <div className="absolute -right-4 -bottom-4 opacity-5 pointer-events-none">
                    <span className="material-symbols-outlined text-[100px]">monitoring</span>
                  </div>
                  
                  <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1">
                    <span className="material-symbols-outlined text-[14px]">health_and_safety</span>
                    Risk Assessment
                  </p>
                  
                  <div className="flex items-baseline gap-2 mt-2">
                    <span className={`text-5xl font-black drop-shadow-md tracking-tighter
                      ${snap.risk_score > 0.65 ? 'text-error' : snap.risk_score > 0.4 ? 'text-amber-400' : 'text-primary'}`}>
                      {Math.round(snap.risk_score * 100)}%
                    </span>
                  </div>
                  
                  {snap.risk_label && (
                    <span className={`mt-2 font-bold text-sm tracking-wide 
                      ${snap.risk_score > 0.65 ? 'text-error-container' : snap.risk_score > 0.4 ? 'text-amber-300' : 'text-primary-fixed-dim'}`}>
                      {snap.risk_label}
                    </span>
                  )}
                </div>
              )}
              
              {hasDrift && snap.drift_delta != null && (
                <div className="p-5 rounded-xl border border-outline-variant/20 bg-surface-container-low/60 flex flex-col justify-center shadow-inner">
                  <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                    <span className="material-symbols-outlined text-[14px]">change_history</span>
                    Drift Magnitude
                  </p>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-4xl font-black drop-shadow-md tracking-tighter ${isCritical ? 'text-error' : 'text-amber-400'}`}>
                      {snap.drift_delta.toFixed(2)}
                    </span>
                    <span className="text-xl font-bold text-slate-500">Δ</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Loading skeleton ────────────────────────────────────────────────────────────
function SkeletonCard() {
  return (
    <div className="flex gap-8 animate-pulse w-full">
      <div className="w-12 h-12 mt-4 rounded-full bg-surface-container flex-shrink-0 border-2 border-outline-variant/10 shadow-[0_0_15px_rgba(0,0,0,0.2)]" />
      <div className="flex-1 bg-surface-container-low/50 backdrop-blur border border-outline-variant/10 rounded-2xl p-7 space-y-6 max-w-4xl pb-12 w-full">
        <div className="flex justify-between">
          <div>
            <div className="h-6 mt-1 bg-surface-container rounded-md w-32 mb-3" />
            <div className="h-5 bg-surface-container rounded-full w-40" />
          </div>
          <div className="h-6 mt-1 bg-surface-container rounded-full w-24" />
        </div>
        <div className="grid grid-cols-12 gap-8">
          <div className="col-span-8 space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-2">
                <div className="flex justify-between">
                  <div className="h-3 bg-surface-container rounded w-1/3" />
                  <div className="h-3 bg-surface-container rounded w-8" />
                </div>
                <div className="h-2.5 bg-surface-container/50 rounded-full" />
              </div>
            ))}
          </div>
          <div className="col-span-4 h-full min-h-[120px] bg-surface-container/40 rounded-xl" />
        </div>
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function HistoryPage() {
  const { user } = useAuth();
  const isInstructor = user?.role === 'instructor' || user?.role === 'admin';

  useEffect(() => { track(PAGE_VIEW, { page: '/history' }); }, []);

  const [selectedLearnerId, setSelectedLearnerId] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);

  // Instructor: fetch student list
  const { data: studentList } = useQuery({
    queryKey: ['instructor-students'],
    queryFn: () => import('../api/recommend').then(m => m.getInstructorStudents()),
    enabled: isInstructor,
    staleTime: 60_000,
    retry: false,
  });

  const targetId = isInstructor
    ? (selectedLearnerId || studentList?.[0]?.learner_id || studentList?.[0]?.id || '')
    : (user?.learner_id ?? user?.id ?? '');

  // Fetch history
  const {
    data: historyData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: isInstructor ? ['learner-history', targetId] : ['my-history'],
    queryFn: isInstructor
      ? () => getLearnerHistory(targetId)
      : getMyHistory,
    enabled: !isInstructor || !!targetId,
    staleTime: 30_000,
    retry: 1,
  });

  const snapshots = historyData?.history ?? historyData?.timeline ?? [];

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-16 min-h-screen">

        {/* ── Sticky header ── */}
        <header className="sticky top-16 z-20 bg-surface/80 backdrop-blur-md border-b border-outline-variant/10 h-14 px-8 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-lg" style={{ fontVariationSettings: "'FILL' 0" }}>history</span>
            <h1 className="text-lg font-headline font-extrabold text-primary tracking-tight">
              Explanation History
            </h1>
          </div>

          <div className="flex items-center gap-3">
            {/* Instructor: student picker */}
            {isInstructor && (
              <div className="relative">
                <button
                  onClick={() => setDropdownOpen((o) => !o)}
                  className="flex items-center gap-2 bg-surface-container/50 border border-outline-variant/30 rounded-full px-4 py-1.5 hover:border-primary/50 transition-colors text-sm"
                >
                  <span className="material-symbols-outlined text-primary text-sm">person_search</span>
                  <span className="font-medium text-on-surface-variant">
                    Student: {targetId || 'Select…'}
                  </span>
                  <span className="material-symbols-outlined text-slate-500 text-sm">expand_more</span>
                </button>

                {dropdownOpen && (
                  <div className="absolute right-0 mt-2 w-64 bg-surface-container border border-outline-variant/20 rounded-xl shadow-xl z-50 overflow-hidden">
                    {studentList?.length > 0 ? (
                      studentList.map((s) => {
                        const id = s.learner_id ?? s.id;
                        return (
                          <button
                            key={id}
                            onClick={() => { setSelectedLearnerId(id); setDropdownOpen(false); }}
                            className="w-full text-left px-4 py-3 text-sm hover:bg-surface-container-high transition-colors"
                          >
                            <span className="font-medium">{s.name ?? id}</span>
                            {s.name && <span className="ml-2 text-xs text-slate-500">{id}</span>}
                          </button>
                        );
                      })
                    ) : (
                      <div className="px-4 py-3 text-sm text-slate-500">No students found.</div>
                    )}
                  </div>
                )}
              </div>
            )}

            <button
              onClick={() => refetch()}
              className="w-9 h-9 flex items-center justify-center rounded-full text-slate-400 hover:text-primary hover:bg-surface-container/50 transition-colors"
              title="Refresh"
            >
              <span className="material-symbols-outlined text-sm">refresh</span>
            </button>
          </div>
        </header>

        {/* ── Content ── */}
        <div className="px-8 pt-8 pb-20 max-w-5xl mx-auto">

          {/* Error */}
          {error && (
            <div className="mb-6 px-4 py-3 rounded-lg bg-error/10 border border-error/20 text-error text-sm font-label">
              {error.response?.data?.detail ?? error.message ?? 'Failed to load history.'}
            </div>
          )}

          {/* Empty state */}
          {!isLoading && !error && snapshots.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 space-y-3 text-center">
              <span className="material-symbols-outlined text-5xl text-slate-600">history_toggle_off</span>
              <p className="font-headline font-bold text-lg text-on-surface-variant">No explanation history yet</p>
              <p className="text-sm text-slate-500 max-w-xs">
                Explanation snapshots appear here after the AI generates recommendations for this learner.
              </p>
            </div>
          )}

          {/* Timeline */}
          <div className="relative">
            {/* Vertical glowing line */}
            {(isLoading || snapshots.length > 0) && (
              <div
                className="absolute left-[22px] top-12 bottom-0 w-1 shadow-[0_0_15px_rgba(68,229,204,0.4)] z-0 rounded-full"
                style={{ 
                  background: 'linear-gradient(to bottom, var(--color-primary) 0%, rgba(68, 229, 204, 0.4) 50%, transparent 100%)',
                  opacity: 0.8
                }}
              />
            )}

            <div className="space-y-10">
              {isLoading
                ? [1, 2, 3].map((i) => <SkeletonCard key={i} />)
                : snapshots.map((snap, i) => (
                    <SnapshotCard key={snap.id ?? snap.created_at ?? i} snap={snap} index={i} />
                  ))
              }
            </div>
          </div>

          {/* Load more hint */}
          {snapshots.length >= 10 && (
            <div className="mt-12 flex justify-center">
              <button
                onClick={() => refetch()}
                className="flex items-center gap-2 px-8 py-3 bg-surface-container hover:bg-surface-container-high text-on-surface-variant rounded-full transition-all border border-outline-variant/20 text-sm font-bold active:scale-95"
              >
                <span className="material-symbols-outlined">refresh</span>
                Load Previous Snapshots
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
