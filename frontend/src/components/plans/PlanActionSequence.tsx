import type { Plan } from '../../types';
import { EmptyState } from '../common/EmptyState';

const ACTION_LABEL: Record<string, string> = {
  ISOLATE_ZONE: 'Isolate zone',
  CLOSE_PIPE: 'Close pipe',
  OPEN_PIPE: 'Open pipe',
  FLUSH_NODE: 'Flush node',
  MONITOR_NODE: 'Monitor node',
  COLLECT_SAMPLE: 'Collect sample',
  WAIT: 'Wait',
  END_PLAN: 'End plan',
};

/** ui-work.txt 13.4: "full action sequence/timing/targets" -- the
 * complete ordered plan, not just an action count. */
export function PlanActionSequence({ plan }: { plan: Plan }) {
  if (plan.actions.length === 0) {
    return <EmptyState title="This plan has no actions." />;
  }
  return (
    <ol className="action-sequence">
      {plan.actions.map((action, index) => (
        <li key={`${plan.id}-${index}`}>
          <span className="action-sequence-index">{index + 1}</span>
          <strong>{ACTION_LABEL[action.actionType] ?? action.actionType}</strong>
          {action.targetId && <span className="mono">{action.targetId}</span>}
          <span className="supporting">
            t+{action.startMinute}min
            {action.durationMinutes > 0 ? ` for ${action.durationMinutes}min` : ''}
            {action.flowRateLps !== null ? ` · ${action.flowRateLps} L/s` : ''}
          </span>
        </li>
      ))}
    </ol>
  );
}
