import { act } from '@testing-library/react';
import { useConsoleStore } from '../src/store';

test('operator state transitions are explicit and reversible', () => {
  act(() => {
    useConsoleStore.getState().setWorkspace('source');
    useConsoleStore.getState().setReplayIndex(2);
    useConsoleStore.getState().selectPlan('C');
  });
  expect(useConsoleStore.getState()).toMatchObject({
    workspace: 'source',
    replayIndex: 2,
    selectedPlanId: 'C',
  });
  act(() => useConsoleStore.getState().setWorkspace('incident'));
  expect(useConsoleStore.getState().workspace).toBe('incident');
});

test('selection setters cover node, link, plan, and audit sequence independently', () => {
  act(() => {
    useConsoleStore.getState().selectNode('J1');
    useConsoleStore.getState().selectLink('P1');
    useConsoleStore.getState().selectAuditSequence(3);
  });
  expect(useConsoleStore.getState()).toMatchObject({
    selectedNodeId: 'J1',
    selectedLinkId: 'P1',
    selectedAuditSequence: 3,
  });
  act(() => useConsoleStore.getState().selectNode(null));
  expect(useConsoleStore.getState().selectedNodeId).toBeNull();
});

test('layout toggles are independent of each other', () => {
  const before = useConsoleStore.getState();
  act(() => useConsoleStore.getState().toggleLeftRail());
  expect(useConsoleStore.getState().leftRailCollapsed).toBe(!before.leftRailCollapsed);
  expect(useConsoleStore.getState().inspectorCollapsed).toBe(before.inspectorCollapsed);
  expect(useConsoleStore.getState().dockCollapsed).toBe(before.dockCollapsed);
  act(() => useConsoleStore.getState().toggleLeftRail());
});
