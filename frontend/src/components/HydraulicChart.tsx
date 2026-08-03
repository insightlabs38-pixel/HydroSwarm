import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

export function HydraulicChart() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current, undefined, { renderer: 'canvas' });
    chart.setOption({
      backgroundColor: 'transparent',
      textStyle: { color: '#b8ccd4' },
      tooltip: { trigger: 'axis' },
      legend: { data: ['Pressure (m)', 'Concentration (mg/L)'], textStyle: { color: '#b8ccd4' } },
      grid: { left: 45, right: 18, top: 42, bottom: 28 },
      xAxis: {
        type: 'category',
        data: ['08:00', '08:10', '08:20', '08:30', '08:40'],
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
          data: [31, 29, 27, 28, 30],
          color: '#6bd6dd',
        },
        {
          name: 'Concentration (mg/L)',
          type: 'line',
          smooth: true,
          data: [0, 0.12, 0.78, 0.44, 0.16],
          color: '#f4b45f',
        },
      ],
    });
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(ref.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, []);
  return (
    <>
      <div
        ref={ref}
        className="chart"
        role="img"
        aria-label="Pressure remains above 27 meters while concentration peaks at 0.78 milligrams per liter at 08:20"
      />
      <p className="chart-equivalent">
        Text equivalent: pressure 31→27→30 m; concentration 0→0.78→0.16 mg/L between 08:00 and
        08:40.
      </p>
    </>
  );
}
