import { useEffect, useMemo, useRef } from 'react';
import * as echarts from 'echarts/core';
import { ScatterChart } from 'echarts/charts';
import { GridComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { ParetoFrontierEntry } from '../../types';
import { useConsoleStore } from '../../store';
import { EmptyState } from '../common/EmptyState';

echarts.use([ScatterChart, GridComponent, TooltipComponent, CanvasRenderer]);

/**
 * ui-work.txt 15: "Use a 2D scatter plus table; no 3D chart." and
 * "Never visually merge [EXPOSURE_AWARE and HYDRAULIC_ONLY] as one
 * comparable frontier" -- rendered as two entirely separate
 * chart+table pairs, never one combined axis space, since the two
 * groups measure genuinely different things (real chemical exposure
 * vs. hydraulic-only Pydantic defaults).
 */
function ExposureAwareFrontier({
  entries,
  selectedPlanId,
  onSelectPlan,
}: {
  entries: ParetoFrontierEntry[];
  selectedPlanId: string | null;
  onSelectPlan: (planId: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const { reducedMotion } = useConsoleStore();

  useEffect(() => {
    if (!ref.current || entries.length === 0) return;
    const chart = echarts.init(ref.current, undefined, { renderer: 'canvas' });
    chart.setOption({
      backgroundColor: 'transparent',
      animation: !reducedMotion,
      textStyle: { color: '#b8ccd4' },
      tooltip: {
        trigger: 'item',
        formatter: (params: { data: [number, number, string] }) =>
          `${params.data[2]}<br/>mass consumed: ${params.data[0]} mg<br/>service: ${params.data[1]}%`,
      },
      grid: { left: 60, right: 24, top: 24, bottom: 40 },
      xAxis: {
        name: 'Contaminant mass consumed (mg)',
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: { color: '#8da9b4' },
        axisLabel: { color: '#8da9b4' },
        splitLine: { lineStyle: { color: '#18313d' } },
      },
      yAxis: {
        name: 'Service availability (%)',
        nameTextStyle: { color: '#8da9b4' },
        axisLabel: { color: '#8da9b4' },
        splitLine: { lineStyle: { color: '#18313d' } },
      },
      series: [
        {
          type: 'scatter',
          symbolSize: (data: [number, number, string, boolean, boolean]) => (data[4] ? 22 : 14),
          data: entries.map((entry) => [
            entry.consequences.contaminantMassConsumedMg,
            entry.consequences.serviceAvailability * 100,
            entry.label,
            entry.dominated,
            entry.planId === selectedPlanId,
          ]),
          itemStyle: {
            color: (params: { data: [number, number, string, boolean, boolean] }) =>
              params.data[3] ? '#607985' : params.data[4] ? '#6bd6dd' : '#f4b45f',
          },
        },
      ],
    });
    const clickHandler = (params: unknown) => {
      const data = (params as { data: [number, number, string, boolean, boolean] }).data;
      const entry = entries.find((item) => item.label === data[2]);
      if (entry) onSelectPlan(entry.planId);
    };
    chart.on('click', clickHandler);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(ref.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [entries, selectedPlanId, onSelectPlan, reducedMotion]);

  return (
    <>
      <div
        ref={ref}
        className="chart frontier-chart"
        role="img"
        aria-label="Exposure-aware verified response Pareto frontier: contaminant mass consumed versus service availability"
      />
      <div className="table-scroll">
        <table className="benchmark-table">
          <caption>Exposure-aware frontier (real measured chemical exposure)</caption>
          <thead>
            <tr>
              <th>Plan</th>
              <th>Mass consumed (mg)</th>
              <th>Service availability</th>
              <th>Pareto-efficient</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr
                key={entry.planId}
                className={entry.planId === selectedPlanId ? 'selected-row' : ''}
              >
                <th scope="row">
                  <button
                    type="button"
                    className="table-plan-button"
                    onClick={() => onSelectPlan(entry.planId)}
                  >
                    {entry.label}
                    {entry.isNoActionComparator ? ' (comparator)' : ''}
                  </button>
                </th>
                <td>{entry.consequences.contaminantMassConsumedMg}</td>
                <td>{(entry.consequences.serviceAvailability * 100).toFixed(1)}%</td>
                <td>{entry.dominated ? 'dominated' : 'yes'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function HydraulicOnlyFrontier({ entries }: { entries: ParetoFrontierEntry[] }) {
  return (
    <div className="table-scroll">
      <table className="benchmark-table">
        <caption>
          Hydraulic-only frontier (no chemical exposure model -- never comparable to the
          exposure-aware frontier above)
        </caption>
        <thead>
          <tr>
            <th>Plan</th>
            <th>Minimum pressure</th>
            <th>Service availability</th>
            <th>Numerically sensitive</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={entry.planId}>
              <th scope="row">{entry.label}</th>
              <td>{entry.consequences.minimumPressureM.toFixed(1)} m</td>
              <td>{(entry.consequences.serviceAvailability * 100).toFixed(1)}%</td>
              <td>{entry.consequences.numericallySensitive ? 'yes' : 'no'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ParetoFrontier({
  entries,
  selectedPlanId,
  onSelectPlan,
}: {
  entries: ParetoFrontierEntry[];
  selectedPlanId: string | null;
  onSelectPlan: (planId: string) => void;
}) {
  const exposureAware = useMemo(
    () => entries.filter((entry) => entry.group === 'EXPOSURE_AWARE'),
    [entries],
  );
  const hydraulicOnly = useMemo(
    () => entries.filter((entry) => entry.group === 'HYDRAULIC_ONLY'),
    [entries],
  );

  if (entries.length === 0) {
    return (
      <EmptyState title="No verified plans available to compare on the Pareto frontier yet." />
    );
  }

  return (
    <div className="inspector-stack">
      {exposureAware.length > 0 && (
        <ExposureAwareFrontier
          entries={exposureAware}
          selectedPlanId={selectedPlanId}
          onSelectPlan={onSelectPlan}
        />
      )}
      {hydraulicOnly.length > 0 && <HydraulicOnlyFrontier entries={hydraulicOnly} />}
    </div>
  );
}
