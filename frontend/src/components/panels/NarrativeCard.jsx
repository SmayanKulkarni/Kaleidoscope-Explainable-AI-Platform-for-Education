export default function NarrativeCard({ narratives, audience = 'learner' }) {
  const text = narratives?.[audience] ?? narratives?.learner ?? narratives?.instructor;
  if (!text) return null;

  const isInstructor = audience === 'instructor';

  return (
    <div className="bg-gradient-to-br from-primary/5 to-primary-container/10 backdrop-blur-md p-6 rounded-2xl border border-primary/10 shadow-sm">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-primary to-primary-container flex items-center justify-center shrink-0 shadow-inner">
          <span
            className="material-symbols-outlined text-white text-lg"
            style={{ fontVariationSettings: "'FILL' 1" }}
          >
            smart_toy
          </span>
        </div>
        <div>
          <h3 className="font-headline font-bold text-sm">
            {isInstructor ? 'Instructor Insight' : 'AI Coach Says'}
          </h3>
          <span className="text-[10px] font-label text-slate-400 uppercase tracking-tighter">
            Narrated by Groq LLM
          </span>
        </div>
      </div>

      <p className="text-sm text-on-surface/80 leading-relaxed italic">"{text}"</p>

      {narratives?.learner && narratives?.instructor && (
        <div className="mt-3 flex gap-2">
          {['learner', 'instructor'].map((a) => (
            <span
              key={a}
              className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                a === audience
                  ? 'bg-primary text-white'
                  : 'bg-surface-container text-slate-400'
              }`}
            >
              {a}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
