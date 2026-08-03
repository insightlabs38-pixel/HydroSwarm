import type { Plan } from '../types';

export function Counterfactuals({ plans }: { plans: Plan[] }) {
  const noResponse = { id: 'NO RESPONSE', exposure: 0, service: 1, pressure: 0, time: 120 };
  const branches = [
    noResponse,
    {
      id: 'PLAN A',
      exposure: plans[0].exposureReduction,
      service: plans[0].serviceAvailability,
      pressure: plans[0].pressureViolations,
      time: plans[0].containmentMinutes,
    },
    {
      id: 'PLAN B',
      exposure: plans[1].exposureReduction,
      service: plans[1].serviceAvailability,
      pressure: plans[1].pressureViolations,
      time: plans[1].containmentMinutes,
    },
  ];
  return (
    <div className="branch-grid">
      {branches.map((branch) => (
        <article key={branch.id} className={branch.id === 'PLAN B' ? 'recommended-branch' : ''}>
          <h3>{branch.id}</h3>
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
