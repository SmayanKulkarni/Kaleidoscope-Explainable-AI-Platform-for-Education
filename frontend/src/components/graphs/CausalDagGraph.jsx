import { useRef, useEffect, useMemo } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { causalDagToGraph } from '../../lib/graphTransforms';
import { useCausalGraph } from '../../hooks/useCausalGraph';

const GROUP_COLORS = {
  confounder:  '#94a3b8',
  engagement:  '#60a5fa',
  performance: '#34d399',
  risk_signal: '#f87171',
  outcome:     '#fbbf24',
};

const LEGEND_ENTRIES = Object.entries(GROUP_COLORS);

export default function CausalDagGraph({ explanation, width, height = 400 }) {
  const fgRef = useRef();
  const { data: apiGraph, isLoading, error } = useCausalGraph();

  const graphData = useMemo(() => {
    if (apiGraph?.nodes?.length) {
      return {
        nodes: apiGraph.nodes.map((n) => ({
          id:    n.id,
          name:  n.label ?? n.id,
          val:   n.is_causal ? Math.abs(n.ate ?? 0) * 300 + 10 : 6,
          color: GROUP_COLORS[n.group] ?? '#94a3b8',
          group: n.group,
        })),
        links: apiGraph.edges.map((e) => ({
          source: e.from,
          target: e.to,
          color:  '#6b7280',
        })),
      };
    }
    return causalDagToGraph(explanation);
  }, [apiGraph, explanation]);

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge')?.strength(-100);
    }
  }, [graphData]);

  if (isLoading) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label animate-pulse">
      Loading causal graph…
    </div>
  );

  if (error && !graphData.nodes.length) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label">
      Causal graph unavailable.
    </div>
  );

  if (!graphData.nodes.length) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label">
      No causal data available.
    </div>
  );

  return (
    <div className="relative w-full" style={{ height }}>
      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        width={width}
        height={height}
        backgroundColor="transparent"
        nodeLabel="name"
        nodeVal="val"
        nodeColor="color"
        linkColor="color"
        linkWidth={1.5}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkLineDash={(l) => (l.dashed ? [2, 3] : null)}
        controlType="orbit"
      />
      {apiGraph?.nodes?.length > 0 && (
        <div className="absolute top-2 right-2 bg-surface-container-lowest/80 backdrop-blur-sm rounded-xl p-2 space-y-1 text-[10px] font-label border border-outline-variant/10">
          {LEGEND_ENTRIES.map(([group, color]) => (
            <div key={group} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
              <span className="text-slate-500 capitalize">{group.replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
