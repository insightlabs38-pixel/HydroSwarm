import type { Plan } from '../types';

interface Branch {
  key: string;
  label: string;
  exposure: number;
  service: number;
  pressure: number;
  time: number;
  recommended: boolean;
}

export function Counterfactuals({ plans }: { plans: Plan[] }) {
  // A fixed, non-computed comparison baseline for taking no action -- no
  // backend endpoint computes a genuine no-response WNTR consequence yet
  // (same known gap as Plan.exposureReduction, see api.ts's viewFromApi
  // comment). Shown identically in every mode as an explicit reference
  // point, never as a substitute for a real per-incident plan.
  const noResponse: Branch = {
    key: 'no-response',
    label: 'No response',
    exposure: 0,
    service: 1,
    pressure: 0,
    time: 120,
    recommended: false,
  };
  // Every plan the incident actually has, in whatever order they arrived --
  // not a hard-coded "first two" (overnight-plan.txt Task 3.3: "hard-coded
  // recommended branch" and "ensure plan order can change without breaking
  // the UI"). "Recommended" styling follows the plan's real status, not
  // its position.
  const branches: Branch[] = [
    noResponse,
    ...plans.map((plan) => ({
      key: plan.id,
      label: plan.name,
      exposure: plan.exposureReduction,
      service: plan.serviceAvailability,
      pressure: plan.pressureViolations,
      time: plan.containmentMinutes,
      recommended: plan.status === 'RECOMMENDED',
    })),
  ];
  return (
    <div className="branch-grid">
      {branches.map((branch) => (
        <article key={branch.key} className={branch.recommended ? 'recommended-branch' : ''}>
          <h3>{branch.label}</h3>
          <div className="spread-visual" aria-hidden="true">
            <i style={{ width: `${100 - branch.exposure * 100}%` }} />
          </div>
          <dl>
            <div>
              <dt>Exposure reduced</dt>
              <dd>{Math.round(branch.exposure * 100)}%</dd>
            </div>
            <div>
              <dt>Service</dt>
              <dd>{(branch.service * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>Pressure risk</dt>
              <dd>{branch.pressure}</dd>
            </div>
            <div>
              <dt>Containment</dt>
              <dd>{branch.time} min</dd>
            </div>
          </dl>
        </article>
      ))}
    </div>
  );
}
