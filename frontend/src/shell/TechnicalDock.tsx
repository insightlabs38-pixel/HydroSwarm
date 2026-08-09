import { useCallback, useRef } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import type { IncidentView } from '../types';
import { useConsoleStore, type DockTab } from '../store';
import { Timeline } from '../components/Timeline';
import { EvidencePanel } from '../components/EvidencePanel';
import { HydraulicChart } from '../components/HydraulicChart';
import { KeyValueGrid } from '../components/common/KeyValueGrid';
import { EmptyState } from '../components/common/EmptyState';

const DOCK_TABS: { id: DockTab; label: string }[] = [
  { id: 'timeline', label: 'Timeline' },
  { id: 'evidence', label: 'Evidence' },
  { id: 'hydraulics', label: 'Hydraulics' },
  { id: 'verification', label: 'Verification' },
  { id: 'provenance', label: 'Provenance' },
  { id: 'audit', label: 'Audit' },
];

const DOCK_HEIGHT_MIN = 160;
const DOCK_HEIGHT_MAX_VH = 0.45;

function VerificationTab({ incident }: { incident: IncidentView }) {
  const { selectedPlanId } = useConsoleStore();
  // Matches the same selected-or-recommended fallback the Response/
  // Approval workspaces use, so the dock agrees with whichever plan
  // they're actually showing instead of only reacting to an explicit
  // click (ui-work.txt 22 cross-panel synchronization).
  const plan =
    incident.plans.find((item) => item.id === selectedPlanId) ??
    incident.plans.find((item) => item.id === incident.recommendedPlanId) ??
    incident.plans[0];
  if (!plan) {
    return (
      <EmptyState
        title="No plan selected."
        detail="Select a plan from the verified plan comparison table to inspect its verification."
      />
    );
  }
  const verification = plan.verification;
  if (!verification) {
    return (
      <EmptyState title={`Plan ${plan.id} has not yet been submitted for exact verification.`} />
    );
  }
  const consequences = verification.consequences;
  return (
    <div className="inspector-stack">
      <KeyValueGrid
        entries={[
          { key: 'plan', label: 'Plan', value: `${plan.id} · ${plan.name}` },
          { key: 'decision', label: 'Decision', value: verification.decision },
          {
            key: 'status',
            label: 'Verification status',
            value: verification.verificationStatus,
          },
          {
            key: 'simulator',
            label: 'Simulator',
            value: `${verification.simulator} ${verification.simulatorVersion}`,
          },
          { key: 'state-hash', label: 'State hash', value: verification.stateHash, hash: true },
          {
            key: 'context-hash',
            label: 'Context hash',
            value: verification.contextHash,
            hash: true,
          },
          { key: 'verified-at', label: 'Verified at', value: verification.verifiedAt },
        ]}
      />
      {verification.rejectionCodes.length > 0 && (
        <p className="supporting">Rejection codes: {verification.rejectionCodes.join(', ')}</p>
      )}
      {verification.abstentionReason && (
        <p className="supporting">Abstention reason: {verification.abstentionReason}</p>
      )}
      {consequences ? (
        <KeyValueGrid
          entries={[
            {
              key: 'pressure-margin',
              label: 'Pressure margin',
              value:
                consequences.pressureMarginM === null
                  ? null
                  : `${consequences.pressureMarginM.toFixed(1)} m`,
            },
            {
              key: 'service-margin',
              label: 'Service availability margin',
              value:
                consequences.serviceAvailabilityMargin === null
                  ? null
                  : `${(consequences.serviceAvailabilityMargin * 100).toFixed(1)} pp`,
            },
            {
              key: 'sensitive',
              label: 'Numerically sensitive',
              value: consequences.numericallySensitive ? 'yes' : 'no',
            },
            {
              key: 'exposure-evaluated',
              label: 'Exposure evaluated',
              value: consequences.exposureEvaluated ? 'yes' : 'no',
            },
          ]}
        />
      ) : (
        <EmptyState title="No consequence metrics for this verification." />
      )}
    </div>
  );
}

function ProvenanceTab({ incident }: { incident: IncidentView }) {
  return (
    <KeyValueGrid
      entries={[
        {
          key: 'network',
          label: 'Network hash',
          value: incident.provenance.networkHash || null,
          hash: true,
        },
        {
          key: 'feature-schema',
          label: 'Feature schema hash',
          value: incident.provenance.featureSchemaHash || null,
          hash: true,
        },
        {
          key: 'checkpoint',
          label: 'Model checkpoint hash',
          value: incident.provenance.modelCheckpointHash || null,
          hash: true,
        },
        {
          key: 'calibration-version',
          label: 'Calibration version',
          value: incident.provenance.calibrationVersion || null,
        },
        {
          key: 'calibration-hash',
          label: 'Calibration hash',
          value: incident.provenance.calibrationHash || null,
          hash: true,
        },
        { key: 'simulator', label: 'Simulator', value: incident.provenance.simulator },
        {
          key: 'simulator-version',
          label: 'Simulator version',
          value: incident.provenance.simulatorVersion,
        },
        {
          key: 'runtime-mode',
          label: 'Analysis runtime mode',
          value: incident.runtimeAnalysisMode,
        },
      ]}
    />
  );
}

function AuditTab({ incident }: { incident: IncidentView }) {
  const { selectedAuditSequence, selectAuditSequence, setReplayIndex } = useConsoleStore();
  if (incident.audit.length === 0) {
    return <EmptyState title="No audit events recorded for this incident." />;
  }
  return (
    <ol className="audit-list">
      {incident.audit.map((event, index) => (
        <li key={event.sequence}>
          <button
            type="button"
            className="audit-list-row"
            aria-pressed={selectedAuditSequence === event.sequence}
            onClick={() => {
              selectAuditSequence(event.sequence);
              // Keeps the Timeline tab's marker in agreement (ui-work.txt
              // 22) even before the operator switches to that tab.
              setReplayIndex(index);
            }}
          >
            <span className="audit-sequence">{event.sequence.toString().padStart(2, '0')}</span>
            <time>{event.timestamp}</time>
            <div>
              <strong>{event.type.replaceAll('_', ' ')}</strong>
              <small>{event.actor}</small>
              <p>{event.detail}</p>
            </div>
          </button>
        </li>
      ))}
    </ol>
  );
}

export function TechnicalDock({ incident }: { incident: IncidentView }) {
  const { dockTab, setDockTab, dockCollapsed, toggleDock, dockHeight, setDockHeight } =
    useConsoleStore();
  const dragState = useRef<{ startY: number; startHeight: number } | null>(null);

  const clampHeight = useCallback((value: number) => {
    const maxHeight = window.innerHeight * DOCK_HEIGHT_MAX_VH;
    return Math.min(maxHeight, Math.max(DOCK_HEIGHT_MIN, value));
  }, []);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      dragState.current = { startY: event.clientY, startHeight: dockHeight };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [dockHeight],
  );
  const onPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!dragState.current) return;
      const delta = dragState.current.startY - event.clientY;
      setDockHeight(clampHeight(dragState.current.startHeight + delta));
    },
    [clampHeight, setDockHeight],
  );
  const onPointerUp = useCallback(() => {
    dragState.current = null;
  }, []);
  const onSplitterKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'ArrowUp') {
        setDockHeight(clampHeight(dockHeight + 16));
        event.preventDefault();
      } else if (event.key === 'ArrowDown') {
        setDockHeight(clampHeight(dockHeight - 16));
        event.preventDefault();
      }
    },
    [clampHeight, dockHeight, setDockHeight],
  );

  if (dockCollapsed) {
    return (
      <div className="technical-dock collapsed">
        <button type="button" onClick={toggleDock} aria-label="Expand technical dock">
          Technical dock ▲
        </button>
      </div>
    );
  }

  return (
    <section className="technical-dock" style={{ height: dockHeight }} aria-label="Technical dock">
      <div
        className="dock-splitter"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize technical dock"
        aria-valuenow={Math.round(dockHeight)}
        aria-valuemin={DOCK_HEIGHT_MIN}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onKeyDown={onSplitterKeyDown}
      />
      <div className="dock-tabs">
        <div role="tablist" aria-label="Technical dock tabs" className="dock-tablist">
          {DOCK_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`dock-tab-${tab.id}`}
              aria-selected={dockTab === tab.id}
              aria-controls={`dock-panel-${tab.id}`}
              className={dockTab === tab.id ? 'active' : ''}
              onClick={() => setDockTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          className="dock-collapse-toggle"
          onClick={toggleDock}
          aria-label="Collapse technical dock"
        >
          ▼
        </button>
      </div>
      {/* Every panel stays mounted (hidden via the native `hidden` attribute
          rather than conditional rendering) so each tab button's
          aria-controls always references a real element -- a conditionally
          unmounted panel would leave aria-controls dangling for every tab
          except the active one. Six lightweight panels is cheap enough to
          keep mounted here; revisit if UI-10 profiling says otherwise. */}
      <div
        className="dock-panel"
        role="tabpanel"
        id="dock-panel-timeline"
        aria-labelledby="dock-tab-timeline"
        hidden={dockTab !== 'timeline'}
      >
        <Timeline events={incident.audit} />
      </div>
      <div
        className="dock-panel"
        role="tabpanel"
        id="dock-panel-evidence"
        aria-labelledby="dock-tab-evidence"
        hidden={dockTab !== 'evidence'}
      >
        <EvidencePanel incident={incident} />
      </div>
      <div
        className="dock-panel"
        role="tabpanel"
        id="dock-panel-hydraulics"
        aria-labelledby="dock-tab-hydraulics"
        hidden={dockTab !== 'hydraulics'}
      >
        <HydraulicChart incident={incident} />
      </div>
      <div
        className="dock-panel"
        role="tabpanel"
        id="dock-panel-verification"
        aria-labelledby="dock-tab-verification"
        hidden={dockTab !== 'verification'}
      >
        <VerificationTab incident={incident} />
      </div>
      <div
        className="dock-panel"
        role="tabpanel"
        id="dock-panel-provenance"
        aria-labelledby="dock-tab-provenance"
        hidden={dockTab !== 'provenance'}
      >
        <ProvenanceTab incident={incident} />
      </div>
      <div
        className="dock-panel"
        role="tabpanel"
        id="dock-panel-audit"
        aria-labelledby="dock-tab-audit"
        hidden={dockTab !== 'audit'}
      >
        <AuditTab incident={incident} />
      </div>
    </section>
  );
}
