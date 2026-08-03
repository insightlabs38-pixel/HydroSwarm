import { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import type { IncidentView } from '../types';
import { Panel } from '../components/Panel';

export function TopologyPage({ incident }: { incident: IncidentView }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const instance = cytoscape({
      container: ref.current,
      elements: [
        ...incident.nodes.map((node) => ({
          data: { id: node.id, probability: node.probability, candidate: node.candidate },
        })),
        ...incident.links.map((link) => ({
          data: { id: link.id, source: link.source, target: link.target, flow: link.flow },
        })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(id)',
            'background-color': '#6bd6dd',
            color: '#d8f3f5',
            'font-size': 10,
            'text-valign': 'bottom',
            'text-margin-y': 6,
          },
        },
        {
          selector: 'node[candidate]',
          style: { 'background-color': '#f4b45f', width: 28, height: 28 },
        },
        {
          selector: 'edge',
          style: {
            width: 2,
            'line-color': '#456271',
            'target-arrow-color': '#6bd6dd',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
          },
        },
      ],
      layout: { name: 'breadthfirst', directed: true, spacingFactor: 1.2 },
    });
    return () => instance.destroy();
  }, [incident]);
  return (
    <div className="page-stack">
      <header className="page-heading">
        <p className="eyebrow">ENGINEERING DEBUG VIEW</p>
        <h1>Directed hydraulic topology</h1>
        <p>
          Current flow orientation, including reversed links. This view is for topology inspection;
          the 2D map remains primary.
        </p>
      </header>
      <Panel title="Upstream → downstream transport graph" eyebrow="CYTOSCAPE">
        <div
          ref={ref}
          className="topology-canvas"
          role="img"
          aria-label="Directed topology from reservoir R1 through candidate source J117 toward sample J123 and flush node J131"
        />
      </Panel>
    </div>
  );
}
