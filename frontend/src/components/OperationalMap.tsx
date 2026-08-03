import { useEffect, useMemo, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import type { FeatureCollection, LineString, Point } from 'geojson';
import type { IncidentView } from '../types';

const layerLabels = [
  'Assets',
  'Flow',
  'Concentration',
  'Candidates',
  'Sensors',
  'Sample',
  'Actions',
] as const;

export function OperationalMap({ incident }: { incident: IncidentView }) {
  const container = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);
  const [layers, setLayers] = useState<Record<string, boolean>>(
    Object.fromEntries(layerLabels.map((label) => [label, true])),
  );
  const nodeById = useMemo(
    () => new Map(incident.nodes.map((node) => [node.id, node])),
    [incident],
  );

  useEffect(() => {
    if (!container.current) return;
    const nodes: FeatureCollection<Point> = {
      type: 'FeatureCollection',
      features: incident.nodes.map((node) => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: node.coordinates },
        properties: {
          id: node.id,
          probability: node.probability,
          concentration: node.concentration,
          candidate: node.candidate,
          sensor: Boolean(node.sensor),
        },
      })),
    };
    const links: FeatureCollection<LineString> = {
      type: 'FeatureCollection',
      features: incident.links.map((link) => ({
        type: 'Feature',
        geometry: {
          type: 'LineString',
          coordinates: [
            nodeById.get(link.source)!.coordinates,
            nodeById.get(link.target)!.coordinates,
          ],
        },
        properties: {
          id: link.id,
          flow: link.flow,
          concentration: link.concentration,
          action: link.action ?? '',
        },
      })),
    };
    const map = new maplibregl.Map({
      container: container.current,
      style: {
        version: 8,
        sources: {},
        layers: [
          { id: 'background', type: 'background', paint: { 'background-color': '#091922' } },
        ],
      },
      center: [-79.995, 35.008],
      zoom: 12.8,
      attributionControl: false,
      interactive: true,
    });
    mapInstance.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
    map.on('load', () => {
      map.addSource('links', { type: 'geojson', data: links });
      map.addSource('nodes', { type: 'geojson', data: nodes });
      map.addLayer({
        id: 'assets',
        type: 'line',
        source: 'links',
        paint: { 'line-color': '#456271', 'line-width': 3 },
      });
      map.addLayer({
        id: 'concentration',
        type: 'line',
        source: 'links',
        paint: {
          'line-color': [
            'interpolate',
            ['linear'],
            ['get', 'concentration'],
            0,
            '#315565',
            0.4,
            '#f4b45f',
            0.8,
            '#f16c62',
          ],
          'line-width': 6,
          'line-opacity': 0.65,
        },
      });
      map.addLayer({
        id: 'flow',
        type: 'symbol',
        source: 'links',
        layout: {
          'symbol-placement': 'line',
          'symbol-spacing': 55,
          'text-field': '›',
          'text-size': 20,
          'text-keep-upright': false,
        },
        paint: { 'text-color': '#a9e6ef' },
      });
      map.addLayer({
        id: 'candidates',
        type: 'circle',
        source: 'nodes',
        filter: ['==', ['get', 'candidate'], true],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'probability'], 0, 8, 1, 22],
          'circle-color': '#f4b45f',
          'circle-opacity': 0.28,
          'circle-stroke-color': '#ffd79e',
          'circle-stroke-width': 2,
        },
      });
      map.addLayer({
        id: 'node-assets',
        type: 'circle',
        source: 'nodes',
        paint: {
          'circle-radius': 5,
          'circle-color': '#d8f3f5',
          'circle-stroke-color': '#07151d',
          'circle-stroke-width': 2,
        },
      });
      map.addLayer({
        id: 'sensors',
        type: 'circle',
        source: 'nodes',
        filter: ['==', ['get', 'sensor'], true],
        paint: {
          'circle-radius': 9,
          'circle-color': '#07151d',
          'circle-stroke-color': '#6bd6dd',
          'circle-stroke-width': 3,
        },
      });
      map.addLayer({
        id: 'sample',
        type: 'circle',
        source: 'nodes',
        filter: ['==', ['get', 'id'], incident.recommendedSample.nodeId],
        paint: {
          'circle-radius': 13,
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-color': '#e9f871',
          'circle-stroke-width': 3,
          'circle-stroke-opacity': 0.9,
        },
      });
      map.addLayer({
        id: 'node-labels',
        type: 'symbol',
        source: 'nodes',
        layout: { 'text-field': ['get', 'id'], 'text-size': 11, 'text-offset': [0, 1.2] },
        paint: { 'text-color': '#d8f3f5', 'text-halo-color': '#07151d', 'text-halo-width': 2 },
      });
      map.addLayer({
        id: 'actions',
        type: 'line',
        source: 'links',
        filter: ['!=', ['get', 'action'], ''],
        paint: { 'line-color': '#e9f871', 'line-width': 5, 'line-dasharray': [1, 1] },
      });
    });
    return () => {
      mapInstance.current = null;
      map.remove();
    };
  }, [incident, nodeById]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map?.isStyleLoaded()) return;
    const ids: Record<string, string[]> = {
      Assets: ['assets', 'node-assets', 'node-labels'],
      Flow: ['flow'],
      Concentration: ['concentration'],
      Candidates: ['candidates'],
      Sensors: ['sensors'],
      Sample: ['sample'],
      Actions: ['actions'],
    };
    for (const [label, layerIds] of Object.entries(ids)) {
      for (const id of layerIds) {
        if (map.getLayer(id))
          map.setLayoutProperty(id, 'visibility', layers[label] ? 'visible' : 'none');
      }
    }
  }, [layers]);

  return (
    <div className="map-shell">
      <div className="layer-controls" role="group" aria-label="Network map layers">
        {layerLabels.map((label) => (
          <label key={label}>
            <input
              type="checkbox"
              checked={layers[label]}
              onChange={(event) =>
                setLayers((current) => ({ ...current, [label]: event.target.checked }))
              }
            />
            {label}
          </label>
        ))}
      </div>
      <div
        ref={container}
        className="map-canvas"
        aria-label="2D water network map showing flow, contamination, candidate sources, sample J123, and response actions"
        role="img"
      />
      <div className="map-legend" aria-label="Map legend">
        <span>
          <i className="legend-candidate" /> Candidate region
        </span>
        <span>
          <i className="legend-flow" /> Directed flow
        </span>
        <span>
          <i className="legend-action" /> Response action
        </span>
      </div>
      <p className="sr-only">
        Leading source J117 at 76 percent. Candidate region also includes J121. Recommended sample
        J123. Plan B closes P4 and flushes J131.
      </p>
    </div>
  );
}
