import type { Plan } from '../types';

interface Branch {
  key: string;
  label: string;
  exposure: number | null;
  service: number | null;
  pressureViolationMinutes: number | null;
  containmentMinutes: number | null;
  recommended: boolean;
}

function branchFromPlan(plan: Plan): Branch {
  const consequences = plan.verification?.consequences ?? null;
  return {
    key: plan.id,
    label: plan.name,
    exposure: plan.exposureReduction,
    service: consequences?.serviceAvailability ?? null,
    pressureViolationMinutes: consequences?.pressureViolationMinutes ?? null,
    containmentMinutes: consequences?.containmentTimeMinutes ?? null,
    recommended: plan.status === 'RECOMMENDED',
  };
}

export function Counterfactuals({ plans }: { plans: Plan[] }) {
  // A fixed, non-computed comparison baseline for taking no action -- no
  // backend endpoint computes a genuine no-response WNTR consequence yet
  // (same known gap as Plan.exposureReduction, see api.ts's viewFromApi
  // comment). This is REFERENCE BASELINE · NOT SIMULATED -- never
  // display unmeasured service/pressure quantities as computed values.
  const noResponse: Branch = {
    key: 'no-response',
    label: 'No response · NOT SIMULATED',
    exposure: null,
    service: null,
    pressureViolationMinutes: null,
    containmentMinutes: null,
    recommended: false,
  };
  // Every plan the incident actually has, in whatever order they arrived --
  // not a hard-coded "first two" (overnight-plan.txt Task 3.3: "hard-coded
  // recommended branch" and "ensure plan order can change without breaking
  // the UI"). "Recommended" styling follows the plan's real status, not
  // its position.
  const branches: Branch[] = [noResponse, ...plans.map(branchFromPlan)];
  return (
    <div className="branch-grid">      {branches.map((branch) => (
        <article key={branch.key} className={branch.recommended ? 'recommended-branch' : ''}>
          <h3>{branch.label}</h3>
          {branch.key !== 'no-response' && branch.exposure !== null && (
            <div className="spread-visual" aria-hidden="true">
              <i style={{ width: `${100 - branch.exposure * 100}%` }} />
            </div>
          )}
          {branch.key === 'no-response' && (
            <p className="supporting">
              Exposure reduction: 0% by definition. Service, pressure, and containment
              were not evaluated for the no-response baseline.
            </p>
          )}
          <dl>
            <div>
              <dt>Exposure reduced</dt>
              <dd>
                {branch.key === 'no-response'
                  ? '0% by definition'
                  : branch.exposure === null
                    ? 'Not evaluated'
                    : `${Math.round(branch.exposure * 100)}%`}
              </dd>
            </div>
            <div>
              <dt>Service</dt>
              <dd>{branch.service === null ? 'Not evaluated' : `${(branch.service * 100).toFixed(1)}%`}</dd>
            </div>
            <div>
              <dt>Pressure violations</dt>
              <dd>
                {branch.pressureViolationMinutes === null ? 'Not evaluated' : `${branch.pressureViolationMinutes} min`}
              </dd>
            </div>
            <div>
              <dt>Containment</dt>
              <dd>{branch.containmentMinutes === null ? 'Not applicable' : `${branch.containmentMinutes} min`}</dd>
            </div>
          </dl>
        </article>
      ))}
    </div>
  );
}
