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

/** Fit the map to the network's real node geometry -- never a hard-coded
 * demo center (ui-work.txt 8.6). Handles degenerate networks explicitly:
 * a single node cannot be fed to fitBounds (a zero-area box), so it is
 * centered directly instead. */
function fitToNodes(map: maplibregl.Map, nodes: IncidentView['nodes']) {
  if (nodes.length === 0) return;
  if (nodes.length === 1) {
    map.jumpTo({ center: nodes[0].coordinates, zoom: 15 });
    return;
  }
  const bounds = nodes.reduce(
    (box, node) => box.extend(node.coordinates),
    new maplibregl.LngLatBounds(nodes[0].coordinates, nodes[0].coordinates),
  );
  map.fitBounds(bounds, { padding: 56, maxZoom: 17, duration: 0 });
}

export function OperationalMap({ incident }: { incident: IncidentView }) {
  const container = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);
  const hasFlowData = useMemo(() => incident.links.some((link) => link.flow !== null), [incident]);
  const hasConcentrationData = useMemo(
    () => incident.links.some((link) => link.concentration !== null),
    [incident],
  );
  const [layers, setLayers] = useState<Record<string, boolean>>(
    Object.fromEntries(
      layerLabels.map((label) => [
        label,
        label === 'Flow' ? hasFlowData : label === 'Concentration' ? hasConcentrationData : true,
      ]),
    ),
  );
  const nodeById = useMemo(
    () => new Map(incident.nodes.map((node) => [node.id, node])),
    [incident],
  );
  const leadingCandidate = incident.candidates[0];
  const secondaryCandidate = incident.candidates[1];
  const recommendedPlan = incident.plans.find((plan) => plan.status === 'RECOMMENDED');

  useEffect(() => {
    if (!container.current || incident.nodes.length === 0) return;
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
      features: incident.links
        .filter((link) => nodeById.has(link.source) && nodeById.has(link.target))
        .map((link) => ({
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
            // Layers reading these are only ever made visible when the
            // corresponding hasFlowData/hasConcentrationData flag is true
            // (see layer-visibility effect below); the 0 fallback here
            // only keeps the maplibre numeric expression well-typed while
            // the layer is hidden, and is never rendered as a value.
            flow: link.flow ?? 0,
            concentration: link.concentration ?? 0,
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
      center: [0, 0],
      zoom: 1,
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
        layout: { visibility: hasConcentrationData ? 'visible' : 'none' },
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
          visibility: hasFlowData ? 'visible' : 'none',
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
        filter: ['==', ['get', 'id'], incident.recommendedSample?.nodeId ?? ''],
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
      fitToNodes(map, incident.nodes);
    });
    return () => {
      mapInstance.current = null;
      map.remove();
    };
  }, [incident, nodeById, hasFlowData, hasConcentrationData]);

  useEffect(() => {
    setLayers((current) => ({
      ...current,
      Flow: hasFlowData,
      Concentration: hasConcentrationData,
    }));
  }, [hasFlowData, hasConcentrationData]);

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

  if (incident.nodes.length === 0) {
    return (
      <div className="map-shell map-empty" role="status">
        <p>No network geometry available for this incident.</p>
      </div>
    );
  }

  return (
    <div className="map-shell">
      <div className="layer-controls" role="group" aria-label="Network map layers">
        {layerLabels.map((label) => {
          const unavailable =
            (label === 'Flow' && !hasFlowData) ||
            (label === 'Concentration' && !hasConcentrationData);
          return (
            <label key={label} className={unavailable ? 'layer-unavailable' : ''}>
              <input
                type="checkbox"
                checked={layers[label]}
                disabled={unavailable}
                onChange={(event) =>
                  setLayers((current) => ({ ...current, [label]: event.target.checked }))
                }
              />
              {label}
              {unavailable && <small> (data unavailable)</small>}
            </label>
          );
        })}
      </div>
      <div
        ref={container}
        className="map-canvas"
        aria-label={`2D water network map showing candidate sources${incident.recommendedSample ? `, sample ${incident.recommendedSample.nodeId}` : ''}, and response actions${hasFlowData ? ', directed flow' : ''}${hasConcentrationData ? ', link concentration' : ''}`}
        role="img"
      />
      <div className="map-legend" aria-label="Map legend">
        <span>
          <i className="legend-candidate" /> Candidate region
        </span>
        {hasFlowData && (
          <span>
            <i className="legend-flow" /> Directed flow
          </span>
        )}
        <span>
          <i className="legend-action" /> Response action
        </span>
      </div>
      <p className="sr-only">
        {leadingCandidate && (
          <>
            Leading source {leadingCandidate.nodeId} at{' '}
            {Math.round(leadingCandidate.probability * 100)} percent.{' '}
          </>
        )}
        {secondaryCandidate && <>Candidate region also includes {secondaryCandidate.nodeId}. </>}
        {incident.recommendedSample && (
          <>Recommended sample {incident.recommendedSample.nodeId}. </>
        )}
        {recommendedPlan && (
          <>
            Recommended plan: {recommendedPlan.name} ({recommendedPlan.actions.length} action
            {recommendedPlan.actions.length === 1 ? '' : 's'}).
          </>
        )}
      </p>
    </div>
  );
}
