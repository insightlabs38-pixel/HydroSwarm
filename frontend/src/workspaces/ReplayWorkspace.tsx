import { useMutation } from '@tanstack/react-query';
import type { IncidentView } from '../types';
import { replayIncident } from '../api/replay';
import { FAILURE_INJECTION_CATEGORIES, FAILURE_INJECTION_REASONS } from '../api/incident';
import { ApiError } from '../api/client';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { Timeline } from '../components/Timeline';
import { EmptyState } from '../components/common/EmptyState';
import { KeyValueGrid } from '../components/common/KeyValueGrid';

/**
 * ui-work.txt UI-8: deterministic replay and fail-closed demo states.
 *
 * "Event replay" (ui-work.txt 20) is the only kind of replay this
 * backend supports today: POST /incidents/{id}/replay returns the
 * incident's complete real audit-event history plus a real hash-chain
 * integrity check, but its `state` field is the incident's *current*
 * raw state, not a historical snapshot -- there is no per-step
 * map/chart frame to animate. Never fabricate one; this workspace only
 * ever shows the real event ledger and the real chain-verification
 * result. Failure injection reuses the existing `?failure=<category>`
 * mechanism (already wired end-to-end in api/incident.ts) via real
 * links, so an operator doesn't have to hand-edit the URL to see each
 * governed fail-closed category.
 */
export function ReplayWorkspace({ incident }: { incident: IncidentView }) {
  const verifyMutation = useMutation({
    mutationFn: () => replayIncident(incident.id),
  });

  return (
    <div className="replay-workspace">
      <Panel
        title="Event ledger"
        eyebrow={`${incident.audit.length} EVENTS`}
        className="replay-event-ledger"
      >
        {incident.audit.length === 0 ? (
          <EmptyState title="No audit events recorded for this incident." />
        ) : (
          <Timeline events={incident.audit} />
        )}
      </Panel>
      <aside className="replay-hash-chain" aria-label="Hash-chain verification">
        <Panel title="Verify hash chain" eyebrow="TAMPER-EVIDENT AUDIT">
          {incident.mode !== 'LIVE' ? (
            <EmptyState
              title="Verification is not available in this mode."
              detail="Hash-chain verification requires a live incident with a real audit ledger to check. This deterministic demo has no real ledger to verify."
            />
          ) : (
            <>
              <button
                type="button"
                onClick={() => verifyMutation.mutate()}
                disabled={verifyMutation.isPending}
              >
                {verifyMutation.isPending ? 'Verifying…' : 'Verify hash chain'}
              </button>
              {verifyMutation.isError && (
                <p className="supporting" role="alert">
                  {verifyMutation.error instanceof ApiError
                    ? verifyMutation.error.message
                    : 'Verification failed.'}
                </p>
              )}
              {verifyMutation.isSuccess && (
                <div className="inspector-stack">
                  <StatusBadge tone={verifyMutation.data.chainValid ? 'good' : 'danger'}>
                    {verifyMutation.data.chainValid ? 'CHAIN VALID' : 'CHAIN INVALID'}
                  </StatusBadge>
                  <KeyValueGrid
                    entries={[
                      {
                        key: 'events',
                        label: 'Events verified',
                        value: String(verifyMutation.data.events.length),
                      },
                      { key: 'ood', label: 'OOD level', value: verifyMutation.data.state.oodLevel },
                      {
                        key: 'samples',
                        label: 'Sample count',
                        value: String(verifyMutation.data.state.sampleCount),
                      },
                      {
                        key: 'exact-sims',
                        label: 'Exact simulations used',
                        value: String(verifyMutation.data.state.exactSimulationsUsed),
                      },
                      {
                        key: 'plans-verified',
                        label: 'Plans exactly verified',
                        value: String(verifyMutation.data.state.plansExactlyVerified),
                      },
                      {
                        key: 'cache-hits',
                        label: 'Exact simulation cache hits',
                        value: String(verifyMutation.data.state.exactSimulationCacheHits),
                      },
                      {
                        key: 'budget',
                        label: 'Remaining EPANET budget',
                        value: String(verifyMutation.data.state.remainingEpanetBudget),
                      },
                    ]}
                  />
                </div>
              )}
            </>
          )}
        </Panel>
      </aside>
      <Panel
        title="Failure-injection demonstration"
        eyebrow="DIAGNOSTICS"
        className="replay-diagnostics"
      >
        <p className="supporting">
          Every governed fail-closed category, deterministically reachable without a live backend in
          that state. Selecting one reloads this console in ERROR mode with the exact reason named
          -- never a false VERIFIED/LIVE state.
        </p>
        <ul className="failure-injection-list">
          {FAILURE_INJECTION_CATEGORIES.map((category) => (
            <li key={category}>
              <a href={`?failure=${category}`}>{category.replaceAll('_', ' ')}</a>
              <span className="supporting">{FAILURE_INJECTION_REASONS[category]}</span>
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
