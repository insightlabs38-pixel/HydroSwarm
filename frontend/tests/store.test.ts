import { act } from '@testing-library/react';
import { useConsoleStore } from '../src/store';

test('operator state transitions are explicit and reversible', () => {
  act(() => {
    useConsoleStore.getState().setPage('audit');
    useConsoleStore.getState().setTimeIndex(2);
    useConsoleStore.getState().selectPlan('C');
  });
  expect(useConsoleStore.getState()).toMatchObject({
    page: 'audit',
    timeIndex: 2,
    selectedPlan: 'C',
  });
  act(() => useConsoleStore.getState().setPage('overview'));
  expect(useConsoleStore.getState().page).toBe('overview');
});
