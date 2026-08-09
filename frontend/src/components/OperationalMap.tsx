import { useEffect, useMemo, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import type { FeatureCollection, LineString, Point } from 'geojson';
import type { IncidentView, PlanAction } from '../types';
import { useConsoleStore, type MapLayerState } from '../store';

const layerLabels = [
  'Assets',
  'Flow',
  'Concentration',
  'Candidates',
  'Sensors',
  'Sample',
  'Actions',
] as const;

/** ConsoleUiState.mapLayers (ui-work.txt 10) uses lowercase keys shared
 * with the rest of the app; this view renders the capitalized on-canvas
 * labels from ui-work.txt 11's layer list. */
const LAYER_STORE_KEY: Record<(typeof layerLabels)[number], keyof MapLayerState> = {
  Assets: 'assets',
  Flow: 'flow',
  Concentration: 'concentration',
  Candidates: 'candidates',
  Sensors: 'sensors',
  Sample: 'sample',
  Actions: 'actions',
};

const NODE_ACTION_TYPES = ['ISOLATE_ZONE', 'FLUSH_NODE', 'MONITOR_NODE', 'COLLECT_SAMPLE'] as const;
const LINK_ACTION_TYPES = ['CLOSE_PIPE', 'OPEN_PIPE'] as const;

const NODE_ACTION_COLORS: Record<string, string> = {
  ISOLATE_ZONE: '#f16c62',
  FLUSH_NODE: '#6bd6dd',
  MONITOR_NODE: '#f4b45f',
  COLLECT_SAMPLE: '#cadd73',
};
const LINK_ACTION_COLORS: Record<string, string> = {
  CLOSE_PIPE: '#f16c62',
  OPEN_PIPE: '#b6df83',
};
const ACTION_LABELS: Record<string, string> = {
  ISOLATE_ZONE: 'Isolate zone',
  CLOSE_PIPE: 'Close pipe',
  OPEN_PIPE: 'Open pipe',
  FLUSH_NODE: 'Flush node',
  MONITOR_NODE: 'Monitor node',
  COLLECT_SAMPLE: 'Collect sample',
  WAIT: 'Wait',
  END_PLAN: 'End plan',
};

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

function popupContent(title: string, rows: [string, string][]): HTMLElement {
  const container = document.createElement('div');
  container.className = 'map-popup';
  const heading = document.createElement('strong');
  heading.textContent = title;
  container.appendChild(heading);
  const dl = document.createElement('dl');
  for (const [label, value] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = value;
    dl.appendChild(dt);
    dl.appendChild(dd);
  }
  container.appendChild(dl);
  return container;
}

export function OperationalMap({ incident }: { incident: IncidentView }) {
  const container = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);
  const popupInstance = useRef<maplibregl.Popup | null>(null);
  const {
    selectedNodeId,
    selectedLinkId,
    selectedPlanId,
    selectNode,
    selectLink,
    mapLayers,
    setMapLayer,
    mapLayerControlVisible,
    mapFitRequestedAt,
  } = useConsoleStore();

  const hasFlowData = useMemo(() => incident.links.some((link) => link.flow !== null), [incident]);
  const hasConcentrationData = useMemo(
    () => incident.links.some((link) => link.concentration !== null),
    [incident],
  );
  const nodeById = useMemo(
    () => new Map(incident.nodes.map((node) => [node.id, node])),
    [incident],
  );
  const linkById = useMemo(
    () => new Map(incident.links.map((link) => [link.id, link])),
    [incident],
  );
  const leadingCandidate = incident.candidates[0];
  const secondaryCandidate = incident.candidates[1];

  // ui-work.txt 22: "Selecting from table/frontier/recommendation must
  // update: plan action overlay." The operator's explicit selection wins;
  // otherwise the strategist-ranked top proposal (if any) is shown.
  const activePlan =
    incident.plans.find((plan) => plan.id === selectedPlanId) ??
    incident.plans.find((plan) => plan.id === incident.recommendedPlanId) ??
    null;

  function actionFeatures(actions: PlanAction[]) {
    const nodeFeatures: FeatureCollection<Point> = { type: 'FeatureCollection', features: [] };
    const linkFeatures: FeatureCollection<LineString> = { type: 'FeatureCollection', features: [] };
    for (const action of actions) {
      if (!action.targetId) continue;
      const node = nodeById.get(action.targetId);
      if (node && (NODE_ACTION_TYPES as readonly string[]).includes(action.actionType)) {
        nodeFeatures.features.push({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: node.coordinates },
          properties: { actionType: action.actionType },
        });
        continue;
      }
      const link = linkById.get(action.targetId);
      if (link && (LINK_ACTION_TYPES as readonly string[]).includes(action.actionType)) {
        const source = nodeById.get(link.source);
        const target = nodeById.get(link.target);
        if (source && target) {
          linkFeatures.features.push({
            type: 'Feature',
            geometry: { type: 'LineString', coordinates: [source.coordinates, target.coordinates] },
            properties: { actionType: action.actionType },
          });
        }
      }
    }
    return { nodeFeatures, linkFeatures };
  }

  useEffect(() => {
    if (!container.current || incident.nodes.length === 0) return;
    const nodes: FeatureCollection<Point> = {
      type: 'FeatureCollection',
      features: incident.nodes.map((node) => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: node.coordinates },
        properties: {
          id: node.id,
          kind: node.kind,
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
            // The Flow/Concentration layers reading these are only ever
            // made visible when the corresponding hasFlowData/
            // hasConcentrationData flag is true (see the layer-visibility
            // effect below); the 0 fallback here only keeps the maplibre
            // numeric expression well-typed while the layer is hidden,
            // and is never rendered as a value.
            flow: link.flow ?? 0,
            concentration: link.concentration ?? 0,
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
    // Real browsers fire 'load' asynchronously (unlike the synchronous
    // test mock), so the separate activePlan-driven effect below can run
    // and bail out (map not loaded yet) before this callback ever runs,
    // and never gets a second chance since its dependency hasn't changed
    // since mount. Seeding these sources with the plan active *right now*
    // (captured in this closure) avoids ever showing an empty overlay for
    // a plan that really does have actions.
    const { nodeFeatures: initialActionNodes, linkFeatures: initialActionLinks } = actionFeatures(
      activePlan?.actions ?? [],
    );
    map.on('load', () => {
      map.addSource('links', { type: 'geojson', data: links });
      map.addSource('nodes', { type: 'geojson', data: nodes });
      map.addSource('action-nodes', { type: 'geojson', data: initialActionNodes });
      map.addSource('action-links', { type: 'geojson', data: initialActionLinks });
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
        id: 'action-links',
        type: 'line',
        source: 'action-links',
        paint: {
          'line-color': [
            'match',
            ['get', 'actionType'],
            'CLOSE_PIPE',
            LINK_ACTION_COLORS.CLOSE_PIPE,
            'OPEN_PIPE',
            LINK_ACTION_COLORS.OPEN_PIPE,
            '#e9f871',
          ],
          'line-width': 5,
          'line-dasharray': [1, 1],
        },
      });
      map.addLayer({
        id: 'action-nodes',
        type: 'circle',
        source: 'action-nodes',
        paint: {
          'circle-radius': 16,
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-width': 3,
          'circle-stroke-color': [
            'match',
            ['get', 'actionType'],
            'ISOLATE_ZONE',
            NODE_ACTION_COLORS.ISOLATE_ZONE,
            'FLUSH_NODE',
            NODE_ACTION_COLORS.FLUSH_NODE,
            'MONITOR_NODE',
            NODE_ACTION_COLORS.MONITOR_NODE,
            'COLLECT_SAMPLE',
            NODE_ACTION_COLORS.COLLECT_SAMPLE,
            '#e9f871',
          ],
        },
      });
      map.addLayer({
        id: 'selected-highlight',
        type: 'circle',
        source: 'nodes',
        filter: ['==', ['get', 'id'], ''],
        paint: {
          'circle-radius': 19,
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-color': '#6bd6dd',
          'circle-stroke-width': 2,
          'circle-stroke-opacity': 0.9,
        },
      });
      map.addLayer({
        id: 'selected-link-highlight',
        type: 'line',
        source: 'links',
        filter: ['==', ['get', 'id'], ''],
        paint: { 'line-color': '#6bd6dd', 'line-width': 8, 'line-opacity': 0.45 },
      });

      const clickableLayers = ['node-assets', 'candidates', 'sensors'];
      for (const layerId of clickableLayers) {
        map.on('click', layerId, (event) => {
          const feature = event.features?.[0];
          if (!feature) return;
          const props = feature.properties as Record<string, unknown>;
          const nodeId = String(props.id);
          selectNode(nodeId);
          const node = nodeById.get(nodeId);
          const rows: [string, string][] = [
            ['Type', String(props.kind ?? 'node')],
            ['Probability', `${Math.round(Number(props.probability ?? 0) * 100)}%`],
          ];
          if (node?.sensor) {
            rows.push(['Sensor health', node.sensor.health]);
            rows.push(['Sensor quality', `${Math.round(node.sensor.quality * 100)}%`]);
          }
          popupInstance.current?.remove();
          popupInstance.current = new maplibregl.Popup({ closeButton: true, maxWidth: '220px' })
            .setLngLat((feature.geometry as Point).coordinates as [number, number])
            .setDOMContent(popupContent(nodeId, rows))
            .addTo(map);
        });
        map.on('mouseenter', layerId, () => {
          map.getCanvas().style.cursor = 'pointer';
        });
        map.on('mouseleave', layerId, () => {
          map.getCanvas().style.cursor = '';
        });
      }
      map.on('click', 'assets', (event) => {
        const feature = event.features?.[0];
        if (!feature) return;
        // Node markers sit on top of their own links; a click that also
        // hits a node should select only the node, not both.
        const nodeHit = map.queryRenderedFeatures(event.point, { layers: clickableLayers });
        if (nodeHit.length > 0) return;
        const props = feature.properties as Record<string, unknown>;
        const linkId = String(props.id);
        selectLink(linkId);
        popupInstance.current?.remove();
        const rows: [string, string][] = [];
        if (typeof props.flow === 'number' && hasFlowData) rows.push(['Flow', `${props.flow}`]);
        if (typeof props.concentration === 'number' && hasConcentrationData)
          rows.push(['Concentration', `${props.concentration} mg/L`]);
        popupInstance.current = new maplibregl.Popup({ closeButton: true, maxWidth: '220px' })
          .setLngLat(event.lngLat)
          .setDOMContent(popupContent(linkId, rows))
          .addTo(map);
      });
      map.on('mouseenter', 'assets', () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', 'assets', () => {
        map.getCanvas().style.cursor = '';
      });

      fitToNodes(map, incident.nodes);
    });
    return () => {
      mapInstance.current = null;
      popupInstance.current = null;
      map.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incident, nodeById, linkById, hasFlowData, hasConcentrationData]);

  // Layer checkbox visibility lives in the shared store (ui-work.txt 10)
  // so WorkspaceToolbar's layer control can reach it too; force Flow/
  // Concentration off whenever their data is unavailable regardless of
  // the stored preference. Neither of these touches map sources, so
  // selection or plan changes never force a map rebuild (ui-work.txt 26).
  useEffect(() => {
    if (!hasFlowData && mapLayers.flow) setMapLayer('flow', false);
    if (!hasConcentrationData && mapLayers.concentration) setMapLayer('concentration', false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasFlowData, hasConcentrationData]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map?.isStyleLoaded()) return;
    const ids: Record<keyof MapLayerState, string[]> = {
      assets: ['assets', 'node-assets', 'node-labels'],
      flow: ['flow'],
      concentration: ['concentration'],
      candidates: ['candidates'],
      sensors: ['sensors'],
      sample: ['sample'],
      actions: ['action-links', 'action-nodes'],
    };
    for (const [key, layerIds] of Object.entries(ids) as [keyof MapLayerState, string[]][]) {
      for (const id of layerIds) {
        if (map.getLayer(id))
          map.setLayoutProperty(id, 'visibility', mapLayers[key] ? 'visible' : 'none');
      }
    }
  }, [mapLayers]);

  // Selection highlight -- a source/filter update only, never a rebuild.
  useEffect(() => {
    const map = mapInstance.current;
    if (!map?.isStyleLoaded() || !map.getLayer('selected-highlight')) return;
    map.setFilter('selected-highlight', ['==', ['get', 'id'], selectedNodeId ?? '']);
  }, [selectedNodeId]);
  useEffect(() => {
    const map = mapInstance.current;
    if (!map?.isStyleLoaded() || !map.getLayer('selected-link-highlight')) return;
    map.setFilter('selected-link-highlight', ['==', ['get', 'id'], selectedLinkId ?? '']);
  }, [selectedLinkId]);

  // Plan action overlay -- a source update only, never a rebuild.
  useEffect(() => {
    const map = mapInstance.current;
    if (!map?.isStyleLoaded()) return;
    const actionNodesSource = map.getSource('action-nodes') as maplibregl.GeoJSONSource | undefined;
    const actionLinksSource = map.getSource('action-links') as maplibregl.GeoJSONSource | undefined;
    const { nodeFeatures, linkFeatures } = actionFeatures(activePlan?.actions ?? []);
    actionNodesSource?.setData(nodeFeatures);
    actionLinksSource?.setData(linkFeatures);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePlan]);

  // WorkspaceToolbar's "Fit network" control -- a viewport change only,
  // never a rebuild. Skips the initial mount value (0) so this doesn't
  // fight the map's own on-load fit.
  useEffect(() => {
    const map = mapInstance.current;
    if (!map?.isStyleLoaded() || mapFitRequestedAt === 0) return;
    fitToNodes(map, incident.nodes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapFitRequestedAt]);

  const activeActionTypes = useMemo(
    () => new Set((activePlan?.actions ?? []).map((action) => action.actionType)),
    [activePlan],
  );

  if (incident.nodes.length === 0) {
    return (
      <div className="map-shell map-empty" role="status">
        <p>No network geometry available for this incident.</p>
      </div>
    );
  }

  return (
    <div className="map-shell">
      {mapLayerControlVisible && (
        <div className="layer-controls" role="group" aria-label="Network map layers">
          {layerLabels.map((label) => {
            const unavailable =
              (label === 'Flow' && !hasFlowData) ||
              (label === 'Concentration' && !hasConcentrationData);
            const key = LAYER_STORE_KEY[label];
            return (
              <label key={label} className={unavailable ? 'layer-unavailable' : ''}>
                <input
                  type="checkbox"
                  checked={mapLayers[key]}
                  disabled={unavailable}
                  onChange={(event) => setMapLayer(key, event.target.checked)}
                />
                {label}
                {unavailable && <small> (data unavailable)</small>}
              </label>
            );
          })}
        </div>
      )}
      <div
        ref={container}
        className="map-canvas"
        aria-label={`2D water network map showing candidate sources${incident.recommendedSample ? `, sample ${incident.recommendedSample.nodeId}` : ''}${activePlan ? `, and ${activePlan.name} response actions` : ''}${hasFlowData ? ', directed flow' : ''}${hasConcentrationData ? ', link concentration' : ''}`}
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
        {[...NODE_ACTION_TYPES, ...LINK_ACTION_TYPES].map(
          (type) =>
            activeActionTypes.has(type) && (
              <span key={type}>
                <i
                  className="legend-action-swatch"
                  style={{
                    borderColor: (NODE_ACTION_COLORS[type] ?? LINK_ACTION_COLORS[type]) as string,
                  }}
                />
                {ACTION_LABELS[type]}
              </span>
            ),
        )}
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
        {activePlan && (
          <>
            {selectedPlanId === activePlan.id ? 'Selected' : 'Recommended'} plan: {activePlan.name}{' '}
            ({activePlan.actions.length} action{activePlan.actions.length === 1 ? '' : 's'}:{' '}
            {activePlan.actions
              .map(
                (action) =>
                  `${ACTION_LABELS[action.actionType] ?? action.actionType}${action.targetId ? ` at ${action.targetId}` : ''}`,
              )
              .join(', ')}
            ).
          </>
        )}
      </p>
    </div>
  );
}
