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

/** Compact semantic initials shown ONLY in the collapsed rail, so
 * secondary utility buttons never render blank (`.rail-label` is hidden
 * when collapsed, and secondary buttons carry no workflow-status glyph). */
const SECONDARY_INITIAL: Partial<Record<Workspace, string>> = {
  network: 'N',
  validation: 'V',
  authority: 'A',
  benchmarks: 'B',
};

/**
 * Secondary utilities are NOT workflow stages -- they are neutral
 * navigation links with active-page state only.
 */
function deriveStageStatus(
  incident: IncidentView,
  workspace: Workspace,
): Record<Workspace, StageStatus> {
  const isCurrent = (id: Workspace): boolean => workspace === id;
  const secondaryNeutral = (): StageStatus => 'waiting';

  if (incident.mode === 'ERROR') {
    return {
      incident: 'blocked',
      source: 'unavailable',
      sampling: 'unavailable',
      response: 'unavailable',
      approval: 'unavailable',
      replay:
        incident.audit.length > 0 ? (isCurrent('replay') ? 'current' : 'waiting') : 'unavailable',
      network: secondaryNeutral(),
      validation: secondaryNeutral(),
      authority: 'unavailable',
      benchmarks: secondaryNeutral(),
    };
  }
  const progression = deriveWorkflowProgression(incident);
  const primaryReplay: StageStatus =
    incident.audit.length > 0 ? (isCurrent('replay') ? 'current' : 'waiting') : 'unavailable';

  return {
    incident: isCurrent('incident') ? 'current' : 'complete',
    source: progression.source,
    sampling: progression.sampling,
    response: progression.response,
    approval: progression.approval,
    replay: primaryReplay,
    network: secondaryNeutral(),
    validation: secondaryNeutral(),
    authority: secondaryNeutral(),
    benchmarks: secondaryNeutral(),
  };
}

function RailButton({
  id,
  label,
  status,
  active,
  onSelect,
  collapsed,
  isSecondary,
}: {
  id: Workspace;
  label: string;
  status: StageStatus;
  active: boolean;
  onSelect: (id: Workspace) => void;
  collapsed: boolean;
  isSecondary?: boolean;
}) {
  // Secondary utilities: no workflow-status glyph, just active-page
  // highlight -- but the collapsed rail still needs a visible compact
  // initial, or the button renders blank once `.rail-label` is hidden.
  if (isSecondary) {
    return (
      <button
        type="button"
        className={`rail-item rail-secondary${active ? ' rail-active' : ''}`}
        onClick={() => onSelect(id)}
        aria-current={active ? 'page' : undefined}
        aria-label={label}
        title={label}
      >
        {collapsed && (
          <span className="rail-secondary-initial" aria-hidden="true">
            {SECONDARY_INITIAL[id] ?? label.charAt(0)}
          </span>
        )}
        <span className="rail-label" aria-hidden={collapsed}>
          {label}
        </span>
      </button>
    );
  }
  const meta = STAGE_STATUS_META[status];
  // Unavailable stages should not be navigable.
  const disabled = status === 'unavailable';
  return (
    <button
      type="button"
      className={`rail-item rail-status-${status}`}
      onClick={() => onSelect(id)}
      disabled={disabled}
      aria-current={active ? 'page' : undefined}
      aria-label={`${label}: ${meta.word}`}
    >
      <span className="rail-status-glyph" aria-hidden="true">
        {meta.glyph}
      </span>
      <span className="rail-label" aria-hidden={collapsed}>
        {label}
      </span>
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
            isSecondary
          />
        ))}
      </div>
      <button
        type="button"
        className="rail-collapse-toggle"
        onClick={toggleLeftRail}
        aria-pressed={leftRailCollapsed}
      >
        {leftRailCollapsed ? '» Expand workflow' : '« Collapse workflow'}
      </button>
    </nav>
  );
}
