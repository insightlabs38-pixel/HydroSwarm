import { useConsoleStore } from '../store';
import type { Plan } from '../types';
import { StatusBadge } from './StatusBadge';

const tone = { REJECTED: 'danger', RECOMMENDED: 'good', VALID: 'info', PENDING: 'warn' } as const;

export function PlanTable({ plans }: { plans: Plan[] }) {
  const { selectedPlan, selectPlan } = useConsoleStore();
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
            <th scope="col">Actions</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {plans.map((plan) => (
            <tr key={plan.id} className={selectedPlan === plan.id ? 'selected-row' : ''}>
              <th scope="row">
                <button
                  type="button"
                  className="table-plan-button"
                  onClick={() => selectPlan(plan.id)}
                  aria-pressed={selectedPlan === plan.id}
                >
                  {plan.id} · {plan.name}
                </button>
              </th>
              <td>{Math.round(plan.exposureReduction * 100)}%</td>
              <td>
                {plan.pressureViolations === 0 ? '0 — safe' : `${plan.pressureViolations} — unsafe`}
              </td>
              <td>{(plan.serviceAvailability * 100).toFixed(1)}%</td>
              <td>{plan.actions}</td>
              <td>
                <StatusBadge tone={tone[plan.status]}>{plan.status}</StatusBadge>
                {plan.rejectionReason && <small>{plan.rejectionReason}</small>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
