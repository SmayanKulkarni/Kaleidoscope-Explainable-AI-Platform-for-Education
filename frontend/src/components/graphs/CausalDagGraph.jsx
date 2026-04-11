import { useRef, useEffect } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { causalDagToGraph } from '../../lib/graphTransforms';

export default function CausalDagGraph({ explanation, width, height = 400 }) {
  const fgRef = useRef();
  const { nodes, links } = causalDagToGraph(explanation);

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge')?.strength(-100);
    }
  }, []);

  if (!nodes.length) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label">
      No causal data available.
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
      linkColor="color"
      linkWidth={1.5}
      linkDirectionalArrowLength={5}
      linkDirectionalArrowRelPos={1}
      linkLineDash={(l) => (l.dashed ? [2, 3] : null)}
      controlType="orbit"
    />
  );
}
