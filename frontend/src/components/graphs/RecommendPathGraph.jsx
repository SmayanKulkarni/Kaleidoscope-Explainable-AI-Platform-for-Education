import { useRef, useEffect } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { recommendPathToGraph } from '../../lib/graphTransforms';

export default function RecommendPathGraph({ recommendation, learnerId, width, height = 400 }) {
  const fgRef = useRef();
  const { nodes, links } = recommendPathToGraph(recommendation, learnerId ?? 'You');

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge')?.strength(-60);
    }
  }, []);

  if (!recommendation) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label">
      Select a recommendation to visualise its path.
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
      nodeColor={(n) => n.isStudent ? '#6366f1' : n.isResource ? '#f59e0b' : '#94a3b8'}
      linkColor={() => '#cbd5e1'}
      linkWidth={1}
      linkDirectionalArrowLength={4}
      linkDirectionalArrowRelPos={1}
      controlType="orbit"
    />
  );
}
