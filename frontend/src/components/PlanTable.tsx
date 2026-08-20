import { useConsoleStore } from '../store';
import type { Plan } from '../types';
import { StatusBadge } from './StatusBadge';
import { formatDisplayId } from '../displayId';

const tone = { REJECTED: 'danger', RECOMMENDED: 'good', VALID: 'info', PENDING: 'warn' } as const;

function rejectionText(plan: Plan): string | undefined {
  const verification = plan.verification;
  if (!verification) return undefined;
  if (verification.decision === 'REJECTED')
    return verification.rejectionCodes.join(', ') || 'rejected by exact simulation';
  if (verification.decision === 'ABSTAINED') return verification.abstentionReason ?? undefined;
  return undefined;
}

export function PlanTable({ plans }: { plans: Plan[] }) {
  const { selectedPlanId, selectPlan } = useConsoleStore();
  return (
    <div className="table-scroll">
      <table className="plan-table">
        <caption>Simulator-verified response plan comparison</caption>
        <thead>
          <tr>
            <th scope="col">Plan</th>
            <th scope="col">Exposure reduction</th>
            <th scope="col">Pressure violations</th>
            <th scope="col">Service availability</th>
            <th scope="col" className="col-actions">
              Actions
            </th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {plans.map((plan) => {
            const consequences = plan.verification?.consequences ?? null;
            return (
              <tr key={plan.id} className={selectedPlanId === plan.id ? 'selected-row' : ''}>
                <th scope="row">
                  <button
                    type="button"
                    className="table-plan-button"
                    onClick={() => selectPlan(plan.id)}
                    aria-pressed={selectedPlanId === plan.id}
                    title={plan.id}
                  >
                    {formatDisplayId(plan.id)} · {plan.name}
                  </button>
                </th>
                <td>
                  {plan.exposureReduction === null
                    ? 'Not evaluated'
                    : `${Math.round(plan.exposureReduction * 100)}%`}
                </td>
                <td>
                  {consequences === null
                    ? '—'
                    : consequences.pressureViolationMinutes === 0
                      ? '0 min — no modeled violation'
                      : `${consequences.pressureViolationMinutes} min — violation`}
                </td>
                <td>
                  {consequences === null
                    ? '—'
                    : `${(consequences.serviceAvailability * 100).toFixed(1)}%`}
                </td>
                <td className="col-actions">{plan.actions.length}</td>
                <td>
                  <StatusBadge tone={tone[plan.status]}>{plan.status}</StatusBadge>
                  {rejectionText(plan) && <small>{rejectionText(plan)}</small>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
