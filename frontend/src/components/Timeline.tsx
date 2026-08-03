import { useEffect } from 'react';
import { useConsoleStore } from '../store';
import type { AuditEvent } from '../types';

export function Timeline({ events }: { events: AuditEvent[] }) {
  const { timeIndex, playing, speed, reducedMotion, setTimeIndex, togglePlaying, setSpeed } =
    useConsoleStore();
  useEffect(() => {
    if (!playing || reducedMotion) return;
    const timer = window.setInterval(
      () => setTimeIndex((timeIndex + 1) % events.length),
      1200 / speed,
    );
    return () => window.clearInterval(timer);
  }, [events.length, playing, reducedMotion, setTimeIndex, speed, timeIndex]);
  return (
    <div className="timeline">
      <div className="playback-controls">
        <button
          type="button"
          onClick={() => setTimeIndex(Math.max(0, timeIndex - 1))}
          aria-label="Previous event"
        >
          ←
        </button>
        <button type="button" onClick={togglePlaying} aria-pressed={playing}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          onClick={() => setTimeIndex(Math.min(events.length - 1, timeIndex + 1))}
          aria-label="Next event"
        >
          →
        </button>
        <label>
          Speed{' '}
          <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
          </select>
        </label>
      </div>
      <input
        aria-label="Incident timeline position"
        type="range"
        min={0}
        max={events.length - 1}
        value={timeIndex}
        onChange={(event) => setTimeIndex(Number(event.target.value))}
      />
      <div className="timeline-event" aria-live="polite">
        <span>{events[timeIndex]?.timestamp}</span>
        <strong>{events[timeIndex]?.type.replaceAll('_', ' ')}</strong>
        <p>{events[timeIndex]?.detail}</p>
      </div>
    </div>
  );
}
