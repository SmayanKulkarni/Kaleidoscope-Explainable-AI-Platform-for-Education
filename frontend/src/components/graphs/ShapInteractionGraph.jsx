import { useRef, useEffect } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { shapInteractionToGraph } from '../../lib/graphTransforms';
import { CAUSAL_COLORS } from '../../lib/colors';

const COLOR_RE = /^(#(?:[0-9a-fA-F]{3,8})|rgba?\([^\)]+\)|hsla?\([^\)]+\))$/;
const safeColor = (value, fallback) => {
  if (typeof value !== 'string') return fallback;
  const normalized = value.trim();
  return COLOR_RE.test(normalized) ? normalized : fallback;
};

export default function ShapInteractionGraph({ explanation, width, height = 400 }) {
  const fgRef = useRef();
  const { nodes, links } = shapInteractionToGraph(explanation);

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge')?.strength(-120);
    }
  }, []);

  if (!nodes.length) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label">
      No interaction data available.
    </div>
  );

  return (
    <ForceGraph3D
      ref={fgRef}
      graphData={{ nodes, links }}
      width={width}
      height={height}
      backgroundColor="rgba(0,0,0,0)"
      nodeLabel="name"
      nodeVal="val"
      nodeColor={(n) => safeColor(n?.color, '#94a3b8')}
      linkWidth={(l) => (l.value ?? 0.01) * 500 + 0.5}
      linkColor={(l) => safeColor(l?.color, '#cbd5e1')}
      linkDirectionalArrowLength={4}
      linkDirectionalArrowRelPos={1}
      nodeThreeObject={null}
      controlType="orbit"
    />
  );
}
