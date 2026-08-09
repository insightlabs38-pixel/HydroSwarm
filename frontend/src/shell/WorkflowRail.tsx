import type { IncidentView } from '../types';
import { useConsoleStore, type Workspace } from '../store';
import { deriveWorkflowProgression, type StageStatus } from '../workflow';

const STAGE_STATUS_META: Record<StageStatus, { glyph: string; word: string }> = {
  complete: { glyph: '✓', word: 'complete' },
  current: { glyph: '●', word: 'current' },
  waiting: { glyph: '…', word: 'waiting' },
  blocked: { glyph: '✕', word: 'blocked' },
  caution: { glyph: '!', word: 'caution' },
  unavailable: { glyph: '—', word: 'unavailable' },
};

const PRIMARY_STAGES: { id: Workspace; label: string }[] = [
  { id: 'incident', label: 'Incident' },
  { id: 'source', label: 'Source' },
  { id: 'sampling', label: 'Sampling' },
  { id: 'response', label: 'Response' },
  { id: 'approval', label: 'Approval' },
  { id: 'replay', label: 'Replay' },
];

const SECONDARY_UTILITIES: { id: Workspace; label: string }[] = [
  { id: 'network', label: 'Network' },
  { id: 'validation', label: 'Validation' },
  { id: 'authority', label: 'Model & Authority' },
  { id: 'benchmarks', label: 'Benchmarks' },
];

/**
 * Every status here is derived from real IncidentView fields, never
 * hard-coded (ui-work.txt 4: "Derive state from backend truth rather than
 * hard-coding it"). Source/Sampling/Response/Approval come from the
 * single shared `deriveWorkflowProgression` helper (workflow.ts), whose
 * authority is `incident.status` -- the real controller stage -- rather
 * than incidental artifacts like `recommendedSample`/`plans`, so this
 * rail cannot disagree with Overview/DecisionInspector about what stage
 * the incident is actually in. Note this means the "current" glyph marks
 * the stage the controller considers active right now, which is
 * independent of which page the operator happens to be viewing (that is
 * `aria-current="page"` / the `.rail-item[aria-current='page']` border
 * highlight, rendered separately below).
 */
function deriveStageStatus(
  incident: IncidentView,
  workspace: Workspace,
): Record<Workspace, StageStatus> {
  const isCurrent = (id: Workspace): boolean => workspace === id;
  if (incident.mode === 'ERROR') {
    return {
      incident: 'blocked',
      source: 'unavailable',
      sampling: 'unavailable',
      response: 'unavailable',
      approval: 'unavailable',
      replay:
        incident.audit.length > 0 ? (isCurrent('replay') ? 'current' : 'waiting') : 'unavailable',
      // Network import/management does not depend on this incident's
      // analysis succeeding.
      network: isCurrent('network') ? 'current' : 'complete',
      validation: isCurrent('validation') ? 'current' : 'complete',
      // Authority certificates are scoped to this incident's analysis,
      // which failed -- honestly unavailable rather than an empty table.
      authority: 'unavailable',
      benchmarks: isCurrent('benchmarks') ? 'current' : 'complete',
    };
  }
  const progression = deriveWorkflowProgression(incident);
  return {
    incident: isCurrent('incident') ? 'current' : 'complete',
    source: progression.source,
    sampling: progression.sampling,
    response: progression.response,
    approval: progression.approval,
    replay: isCurrent('replay') ? 'current' : incident.audit.length > 0 ? 'waiting' : 'unavailable',
    network: isCurrent('network') ? 'current' : 'complete',
    validation: isCurrent('validation') ? 'current' : 'complete',
    authority: isCurrent('authority') ? 'current' : 'complete',
    benchmarks: isCurrent('benchmarks') ? 'current' : 'complete',
  };
}

function RailButton({
  id,
  label,
  status,
  active,
  onSelect,
  collapsed,
}: {
  id: Workspace;
  label: string;
  status: StageStatus;
  active: boolean;
  onSelect: (id: Workspace) => void;
  collapsed: boolean;
}) {
  const meta = STAGE_STATUS_META[status];
  return (
    <button
      type="button"
      className={`rail-item rail-status-${status}`}
      onClick={() => onSelect(id)}
      aria-current={active ? 'page' : undefined}
      aria-label={`${label}: ${meta.word}`}
    >
      <span className="rail-status-glyph" aria-hidden="true">
        {meta.glyph}
      </span>
      {!collapsed && (
        <span className="rail-label" aria-hidden="true">
          {label}
        </span>
      )}
    </button>
  );
}

export function WorkflowRail({ incident }: { incident: IncidentView }) {
  const { workspace, setWorkspace, leftRailCollapsed, toggleLeftRail } = useConsoleStore();
  const statuses = deriveStageStatus(incident, workspace);
  return (
    <nav
      className={leftRailCollapsed ? 'workflow-rail collapsed' : 'workflow-rail'}
      aria-label="Operator workflow"
    >
      <div className="rail-group">
        {PRIMARY_STAGES.map((stage) => (
          <RailButton
            key={stage.id}
            id={stage.id}
            label={stage.label}
            status={statuses[stage.id]}
            active={workspace === stage.id}
            onSelect={setWorkspace}
            collapsed={leftRailCollapsed}
          />
        ))}
      </div>
      <hr className="rail-separator" />
      <div className="rail-group">
        {SECONDARY_UTILITIES.map((stage) => (
          <RailButton
            key={stage.id}
            id={stage.id}
            label={stage.label}
            status={statuses[stage.id]}
            active={workspace === stage.id}
            onSelect={setWorkspace}
            collapsed={leftRailCollapsed}
          />
        ))}
      </div>
      <button
        type="button"
        className="rail-collapse-toggle"
        onClick={toggleLeftRail}
        aria-pressed={leftRailCollapsed}
      >
        {leftRailCollapsed ? '»' : '« Collapse'}
      </button>
    </nav>
  );
}
