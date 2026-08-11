import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { ModeBanner } from '../src/shell/ModeBanner';
import { demoIncident } from '../src/demoFixture';
import type { ReferenceController } from '../src/reference/useReferenceIncident';

function referenceController(pauseAction: ReferenceController['pauseAction']): ReferenceController {
  return {
    incident: null,
    isLoading: false,
    isError: false,
    errorMessage: null,
    milestoneIndex: 3,
    milestoneCount: 11,
    milestoneLabel: 'Evidence insufficient',
    narrative: '',
    isPaused: true,
    pauseReason: 'A recorded operator boundary requires deliberate replay.',
    pauseAction,
    pauseActionLabel: 'Old artifact wording',
    isPlaying: false,
    isAtEnd: false,
    next: vi.fn(),
    previous: vi.fn(),
    performPauseAction: vi.fn(),
    togglePlay: vi.fn(),
    reset: vi.fn(),
  };
}

test.each([
  ['COLLECT_REFERENCE_SAMPLE', 'Replay sample collection'],
  ['APPROVE_REFERENCE_PLAN', 'Replay operator approval'],
] as const)(
  '%s pause cannot be bypassed by Next and advances only through its replay action',
  async (pauseAction, label) => {
    const user = userEvent.setup();
    const reference = referenceController(pauseAction);
    render(<ModeBanner incident={{ ...demoIncident, mode: 'REFERENCE' }} reference={reference} />);

    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: label }));
    expect(reference.performPauseAction).toHaveBeenCalledTimes(1);
    expect(reference.next).not.toHaveBeenCalled();
  },
);
