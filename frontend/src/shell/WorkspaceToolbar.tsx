import { useConsoleStore, WORKSPACE_LABELS } from '../store';

const MAP_WORKSPACES = new Set(['incident', 'source', 'sampling', 'response', 'approval']);

export function WorkspaceToolbar() {
  const {
    workspace,
    selectedNodeId,
    selectedLinkId,
    selectedPlanId,
    selectNode,
    selectLink,
    selectPlan,
    mapLayerControlVisible,
    toggleMapLayerControl,
    requestMapFit,
  } = useConsoleStore();
  const hasSelection = selectedNodeId || selectedLinkId || selectedPlanId;
  const showsMap = MAP_WORKSPACES.has(workspace);
  return (
    <div className="workspace-toolbar">
      <div className="workspace-toolbar-title">
        <span className="eyebrow">{WORKSPACE_LABELS[workspace]}</span>
        {hasSelection && (
          <span className="breadcrumb mono">
            {selectedNodeId && <>node {selectedNodeId}</>}
            {selectedLinkId && <>link {selectedLinkId}</>}
            {selectedPlanId && <>plan {selectedPlanId}</>}
          </span>
        )}
      </div>
      <div className="workspace-toolbar-controls">
        {showsMap && (
          <>
            <button type="button" onClick={requestMapFit}>
              Fit network
            </button>
            <button
              type="button"
              aria-pressed={mapLayerControlVisible}
              onClick={toggleMapLayerControl}
            >
              Layers
            </button>
          </>
        )}
        <button
          type="button"
          disabled={!hasSelection}
          onClick={() => {
            selectNode(null);
            selectLink(null);
            selectPlan(null);
          }}
        >
          Clear selection
        </button>
      </div>
    </div>
  );
}
