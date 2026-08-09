import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, LineChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { IncidentView } from '../types';
import { useConsoleStore } from '../store';

echarts.use([
  LineChart,
  BarChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  CanvasRenderer,
]);

/** Renders a real recorded hydraulic time series (DEMO_FALLBACK/REPLAY
 * only -- ui-work.txt 8.5) or, when only the latest live sensor readings
 * exist, an honest current-value snapshot instead of an invented series. */
export function HydraulicChart({ incident }: { incident: IncidentView }) {
  const ref = useRef<HTMLDivElement>(null);
  const { reducedMotion } = useConsoleStore();
  const series = incident.hydraulicSeries;
  const sensors = incident.nodes.flatMap((node) =>
    node.sensor ? [{ node, sensor: node.sensor }] : [],
  );

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current, undefined, { renderer: 'canvas' });
    if (series && series.length > 0) {
      chart.setOption({
        backgroundColor: 'transparent',
        animation: !reducedMotion,
        textStyle: { color: '#b8ccd4' },
        tooltip: { trigger: 'axis' },
        legend: {
          data: ['Pressure (m)', 'Concentration (mg/L)'],
          textStyle: { color: '#b8ccd4' },
        },
        grid: { left: 45, right: 18, top: 42, bottom: 28 },
        xAxis: {
          type: 'category',
          data: series.map((point) => point.time),
          axisLabel: { color: '#8da9b4' },
        },
        yAxis: {
          type: 'value',
          axisLabel: { color: '#8da9b4' },
          splitLine: { lineStyle: { color: '#18313d' } },
        },
        series: [
          {
            name: 'Pressure (m)',
            type: 'line',
            smooth: true,
            data: series.map((point) => point.pressureM),
            color: '#6bd6dd',
          },
          {
            name: 'Concentration (mg/L)',
            type: 'line',
            smooth: true,
            data: series.map((point) => point.concentrationMgL),
            color: '#f4b45f',
          },
        ],
      });
    } else if (sensors.length > 0) {
      chart.setOption({
        backgroundColor: 'transparent',
        animation: !reducedMotion,
        textStyle: { color: '#b8ccd4' },
        tooltip: { trigger: 'axis' },
        legend: {
          data: ['Pressure (m)', 'Concentration (mg/L)'],
          textStyle: { color: '#b8ccd4' },
        },
        grid: { left: 45, right: 18, top: 42, bottom: 28 },
        xAxis: {
          type: 'category',
          data: sensors.map(({ node, sensor }) => `${sensor.id} (${node.id})`),
          axisLabel: { color: '#8da9b4' },
        },
        yAxis: {
          type: 'value',
          axisLabel: { color: '#8da9b4' },
          splitLine: { lineStyle: { color: '#18313d' } },
        },
        series: [
          {
            name: 'Pressure (m)',
            type: 'bar',
            // ECharts' own documented convention: `null` renders as a real
            // gap (no bar drawn) at that index, never coerced to 0 --
            // UI-11.1 §2, a genuine 0 m reading must stay visually
            // distinct from "not measured".
            data: sensors.map(({ sensor }) => sensor.pressure),
            color: '#6bd6dd',
          },
          {
            name: 'Concentration (mg/L)',
            type: 'bar',
            data: sensors.map(({ sensor }) => sensor.concentration),
            color: '#f4b45f',
          },
        ],
      });
    }
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(ref.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [series, sensors, reducedMotion]);

  if (!series && sensors.length === 0) {
    return (
      <p className="supporting" role="status">
        No sensor readings available for this incident.
      </p>
    );
  }

  const equivalentText = series
    ? `Text equivalent: pressure ${series.map((p) => p.pressureM).join('→')} m; concentration ${series.map((p) => p.concentrationMgL).join('→')} mg/L between ${series[0]?.time} and ${series[series.length - 1]?.time}.`
    : `Text equivalent (latest snapshot, not a time series): ${sensors
        .map(
          ({ node, sensor }) =>
            `${sensor.id} (${node.id}) pressure ${sensor.pressure === null ? 'not measured' : `${sensor.pressure} m`}, concentration ${sensor.concentration === null ? 'not measured' : `${sensor.concentration} mg/L`}`,
        )
        .join('; ')}.`;

  return (
    <>
      <div
        ref={ref}
        className="chart"
        role="img"
        aria-label={
          series
            ? 'Recorded pressure and concentration time series'
            : 'Latest sensor pressure and concentration snapshot'
        }
      />
      <p className="chart-equivalent">{equivalentText}</p>
    </>
  );
}
