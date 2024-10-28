import React, { useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const KnowledgeGraph = () => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchGraph = async () => {
      try {
        const response = await fetch('http://localhost:8000/query?query=Show me all connections');
        const data = await response.json();
        
        if (data.relationships) {
          // Convert relationships to graph format
          const nodes = new Set();
          const links = [];
          
          data.relationships.forEach(rel => {
            nodes.add(rel.subject);
            nodes.add(rel.object);
            
            links.push({
              source: rel.subject,
              target: rel.object,
              label: rel.predicate
            });
          });

          setGraphData({
            nodes: Array.from(nodes).map(id => ({ 
              id,
              label: id,
              // Different colors for different types of nodes
              color: id.includes('@') ? '#ff6b6b' : '#4ecdc4'
            })),
            links
          });
        }
      } catch (error) {
        console.error('Error fetching knowledge graph:', error);
        setError('Failed to load knowledge graph');
      } finally {
        setLoading(false);
      }
    };

    fetchGraph();
  }, []);

  if (loading) return <div>Loading knowledge graph...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="knowledge-graph-container">
      <h1>Knowledge Graph Visualization</h1>
      <div className="graph-wrapper" style={{ height: '80vh', width: '100%' }}>
        <ForceGraph2D
          graphData={graphData}
          nodeLabel="label"
          linkLabel="label"
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = node.label;
            const fontSize = 12/globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.fillStyle = node.color;
            ctx.beginPath();
            ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI, false);
            ctx.fill();
            
            // Draw node label
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#000';
            ctx.fillText(label, node.x, node.y + 10);
          }}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          linkCurvature={0.25}
          linkColor={() => '#999'}
          linkWidth={1}
          onNodeClick={(node) => {
            // Handle node click - could show additional information
            console.log('Clicked node:', node);
          }}
        />
      </div>
      <div className="graph-legend">
        <h3>Legend</h3>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#ff6b6b' }}></span>
          <span>Organizations</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ backgroundColor: '#4ecdc4' }}></span>
          <span>People & Topics</span>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeGraph;
