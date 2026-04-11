const WEEK_SLOTS = [2, 4, 6, 8, 10, 12];
const SIM_OPTIONS = [100, 500, 1000, 5000];

export function getValidTargetWeeks(currentWeek) {
  return WEEK_SLOTS.filter((w) => w > currentWeek);
}

export default function SimulationSetupPanel({ currentWeek, onRun, loading }) {
  const validTargets = getValidTargetWeeks(currentWeek ?? 0);
  const defaultTarget = validTargets[validTargets.length - 1] ?? null;

  const handleSubmit = (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    onRun({
      target_week: Number(fd.get('target_week')),
      n_simulations: Number(fd.get('n_simulations')),
    });
  };

  if (!validTargets.length) {
    return (
      <div className="p-6 rounded-xl bg-surface-container border border-outline-variant/10 text-sm text-slate-500 text-center">
        You are at week {currentWeek} — no future weeks available to simulate.
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="p-6 rounded-xl bg-surface-container-low border border-outline-variant/10 space-y-5">
      <h3 className="font-headline font-bold text-base">Simulation Setup</h3>

      <div>
        <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
          Project to Week
        </label>
        <div className="flex gap-2 flex-wrap">
          {validTargets.map((w) => (
            <label key={w} className="cursor-pointer">
              <input
                type="radio"
                name="target_week"
                value={w}
                defaultChecked={w === defaultTarget}
                className="sr-only peer"
              />
              <span className="px-4 py-2 rounded-lg border border-outline-variant/30 text-sm font-bold text-slate-600 peer-checked:bg-primary peer-checked:text-white peer-checked:border-primary transition-colors">
                Week {w}
              </span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
          Simulations
        </label>
        <div className="flex gap-2 flex-wrap">
          {SIM_OPTIONS.map((n) => (
            <label key={n} className="cursor-pointer">
              <input
                type="radio"
                name="n_simulations"
                value={n}
                defaultChecked={n === 1000}
                className="sr-only peer"
              />
              <span className="px-4 py-2 rounded-lg border border-outline-variant/30 text-sm font-bold text-slate-600 peer-checked:bg-primary peer-checked:text-white peer-checked:border-primary transition-colors">
                {n.toLocaleString()}
              </span>
            </label>
          ))}
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 bg-gradient-to-r from-primary to-primary-container text-white rounded-xl font-bold text-sm shadow-sm hover:shadow-md transition-all flex items-center justify-center gap-2 disabled:opacity-60"
      >
        <span className="material-symbols-outlined text-sm">
          {loading ? 'hourglass_empty' : 'play_circle'}
        </span>
        {loading ? 'Simulating…' : 'Run Simulation'}
      </button>
    </form>
  );
}
