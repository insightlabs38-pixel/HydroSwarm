import { render } from '@testing-library/react';
import * as echarts from 'echarts/core';
import { HydraulicChart } from '../src/components/HydraulicChart';
import { ParetoFrontier } from '../src/components/plans/ParetoFrontier';
import { demoIncident, demoParetoFrontier } from '../src/demoFixture';
import { useConsoleStore } from '../src/store';

/**
 * ui-work.txt UI-10 / 24: "prefers-reduced-motion plus existing app
 * toggle" -- the CSS `.reduced-motion` class only disables CSS
 * transitions/animations, not ECharts' own canvas-internal animation
 * loop, so the app's `reducedMotion` store toggle must also be threaded
 * into every `echarts.setOption` call. Real regression coverage for a
 * gap UI-10 found and closed (both charts previously ignored the toggle
 * entirely).
 */
test('HydraulicChart disables ECharts animation when reducedMotion is on, and enables it when off', () => {
  const initSpy = echarts.init as unknown as ReturnType<typeof vi.fn>;
  initSpy.mockClear();

  useConsoleStore.setState({ reducedMotion: true });
  const { unmount } = render(<HydraulicChart incident={demoIncident} />);
  const firstInstance = initSpy.mock.results[0]!.value;
  expect(firstInstance.setOption).toHaveBeenCalledWith(
    expect.objectContaining({ animation: false }),
  );
  unmount();

  initSpy.mockClear();
  useConsoleStore.setState({ reducedMotion: false });
  render(<HydraulicChart incident={demoIncident} />);
  const secondInstance = initSpy.mock.results[0]!.value;
  expect(secondInstance.setOption).toHaveBeenCalledWith(
    expect.objectContaining({ animation: true }),
  );
});

test('ParetoFrontier exposure-aware chart disables ECharts animation when reducedMotion is on', () => {
  const initSpy = echarts.init as unknown as ReturnType<typeof vi.fn>;
  initSpy.mockClear();

  useConsoleStore.setState({ reducedMotion: true });
  render(
    <ParetoFrontier entries={demoParetoFrontier} selectedPlanId={null} onSelectPlan={() => {}} />,
  );
  const instance = initSpy.mock.results[0]!.value;
  expect(instance.setOption).toHaveBeenCalledWith(expect.objectContaining({ animation: false }));
});
