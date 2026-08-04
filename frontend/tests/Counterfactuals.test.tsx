import { render, screen } from '@testing-library/react';
import { Counterfactuals } from '../src/components/Counterfactuals';
import type { Plan } from '../src/types';

// Regression coverage for a real bug found while investigating
// overnight-plan.txt Task 8.1: the component used to hard-code plans[0]/
// plans[1] as "PLAN A"/"PLAN B" and hard-code the recommended styling to
// whichever branch was literally named "PLAN B" -- exactly what Task 3.3
// ("hard-coded recommended branch", "ensure plan order can change without
// breaking the UI") already required removing, but this component was
// missed and had no test coverage catching it (the identifier-independence
// test's OLD_HARDCODED_IDENTIFIERS list checks for "Plan A"/"Plan B", which
// never matched this component's literal all-caps "PLAN A"/"PLAN B" text).

function plan(overrides: Partial<Plan>): Plan {
  return {
    id: 'plan-x',
    name: 'Plan X',
    exposureReduction: 0.5,
    pressureViolations: 0,
    serviceAvailability: 0.99,
    actions: 1,
    containmentMinutes: 10,
    status: 'VALID',
    ...overrides,
  };
}

test('renders real plan names, never the old hard-coded PLAN A / PLAN B labels', () => {
  render(
    <Counterfactuals
      plans={[
        plan({ id: 'iso', name: 'Isolate zone 4', status: 'REJECTED' }),
        plan({ id: 'flush', name: 'Controlled flush', status: 'RECOMMENDED' }),
      ]}
    />,
  );
  expect(screen.getByText('Isolate zone 4')).toBeInTheDocument();
  expect(screen.getByText('Controlled flush')).toBeInTheDocument();
  expect(screen.queryByText(/^PLAN A$/)).toBeNull();
  expect(screen.queryByText(/^PLAN B$/)).toBeNull();
});

test('does not crash with a single plan (plans[1] used to throw)', () => {
  render(<Counterfactuals plans={[plan({ id: 'only', name: 'Only plan' })]} />);
  expect(screen.getByText('Only plan')).toBeInTheDocument();
  expect(screen.getByText('No response')).toBeInTheDocument();
});

test('does not crash and renders correctly with zero plans', () => {
  render(<Counterfactuals plans={[]} />);
  expect(screen.getByText('No response')).toBeInTheDocument();
});

test('recommended styling follows plan.status, not plan order or position', () => {
  const { container } = render(
    <Counterfactuals
      plans={[
        // Recommended plan placed FIRST -- the old code only ever styled
        // whichever branch was literally named "PLAN B" (i.e. index 1).
        plan({ id: 'winner', name: 'Winner', status: 'RECOMMENDED' }),
        plan({ id: 'loser', name: 'Loser', status: 'REJECTED' }),
      ]}
    />,
  );
  const articles = Array.from(container.querySelectorAll('article'));
  const winner = articles.find((el) => el.textContent?.includes('Winner'));
  const loser = articles.find((el) => el.textContent?.includes('Loser'));
  expect(winner?.className).toContain('recommended-branch');
  expect(loser?.className).not.toContain('recommended-branch');
});

test('plan order can change without breaking which branch is marked recommended', () => {
  const winner = plan({ id: 'winner', name: 'Winner', status: 'RECOMMENDED' });
  const loser = plan({ id: 'loser', name: 'Loser', status: 'REJECTED' });

  const { container: first, unmount } = render(<Counterfactuals plans={[loser, winner]} />);
  const firstWinnerEl = Array.from(first.querySelectorAll('article')).find((el) =>
    el.textContent?.includes('Winner'),
  );
  expect(firstWinnerEl?.className).toContain('recommended-branch');
  unmount();

  const { container: second } = render(<Counterfactuals plans={[winner, loser]} />);
  const secondWinnerEl = Array.from(second.querySelectorAll('article')).find((el) =>
    el.textContent?.includes('Winner'),
  );
  expect(secondWinnerEl?.className).toContain('recommended-branch');
});
