import { useRef, useEffect } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { shapInteractionToGraph } from '../../lib/graphTransforms';
import { CAUSAL_COLORS } from '../../lib/colors';

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
      backgroundColor="transparent"
      nodeLabel="name"
      nodeVal="val"
      nodeColor="color"
      linkWidth={(l) => (l.value ?? 0.01) * 500 + 0.5}
      linkColor="color"
      linkDirectionalArrowLength={4}
      linkDirectionalArrowRelPos={1}
      nodeThreeObject={null}
      controlType="orbit"
    />
  );
}
