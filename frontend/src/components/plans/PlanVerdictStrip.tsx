import type { Plan } from '../../types';
import { formatDisplayId } from '../../displayId';

function verdict(plan: Plan): string {
  return plan.verification?.decision ?? (plan.status === 'REJECTED' ? 'REJECTED' : 'UNVERIFIED');
}

export function PlanVerdictStrip({
  plans,
  selectedPlanId,
  onSelect,
}: {
  plans: Plan[];
  selectedPlanId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <section className="plan-verdict-strip" aria-label="Plan verification verdicts">
      {plans.map((plan) => {
        const verification = plan.verification;
        const consequences = verification?.consequences;
        const decision = verdict(plan);
        return (
          <button
            key={plan.id}
            type="button"
            className={`plan-verdict plan-verdict-${decision.toLowerCase()}`}
            aria-pressed={plan.id === selectedPlanId}
            onClick={() => onSelect(plan.id)}
            title={plan.id}
          >
            <span className="plan-verdict-name">
              {formatDisplayId(plan.id)} · {plan.name}
            </span>
            <strong>{decision}</strong>
            {verification?.verificationStatus && <span>{verification.verificationStatus}</span>}
            {consequences?.pressureMarginM !== null &&
              consequences?.pressureMarginM !== undefined && (
                <span>
                  pressure {consequences.pressureMarginM >= 0 ? '+' : ''}
                  {consequences.pressureMarginM.toFixed(1)} m
                </span>
              )}
            {verification?.rejectionCodes[0] && <span>{verification.rejectionCodes[0]}</span>}
          </button>
        );
      })}
    </section>
  );
}
