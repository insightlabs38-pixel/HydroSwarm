import { useEffect } from 'react';
import { useConsoleStore } from '../store';
import type { AuditEvent } from '../types';

export function Timeline({ events }: { events: AuditEvent[] }) {
  const {
    replayIndex,
    replayPlaying,
    replaySpeed,
    reducedMotion,
    setReplayIndex,
    toggleReplayPlaying,
    setReplaySpeed,
  } = useConsoleStore();
  useEffect(() => {
    if (!replayPlaying || reducedMotion) return;
    const timer = window.setInterval(
      () => setReplayIndex((replayIndex + 1) % events.length),
      1200 / replaySpeed,
    );
    return () => window.clearInterval(timer);
  }, [events.length, replayPlaying, reducedMotion, setReplayIndex, replaySpeed, replayIndex]);
  return (
    <div className="timeline">
      <div className="playback-controls">
        <button
          type="button"
          onClick={() => setReplayIndex(Math.max(0, replayIndex - 1))}
          aria-label="Previous event"
        >
          ←
        </button>
        <button type="button" onClick={toggleReplayPlaying} aria-pressed={replayPlaying}>
          {replayPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          onClick={() => setReplayIndex(Math.min(events.length - 1, replayIndex + 1))}
          aria-label="Next event"
        >
          →
        </button>
        <label>
          Speed{' '}
          <select
            value={replaySpeed}
            onChange={(event) => setReplaySpeed(Number(event.target.value) as 0.5 | 1 | 2 | 4)}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </label>
      </div>
      <input
        aria-label="Incident timeline position"
        type="range"
        min={0}
        max={events.length - 1}
        value={replayIndex}
        onChange={(event) => setReplayIndex(Number(event.target.value))}
      />
      <div className="timeline-event" aria-live="polite">
        <span>{events[replayIndex]?.timestamp}</span>
        <strong>{events[replayIndex]?.type.replaceAll('_', ' ')}</strong>
        <p>{events[replayIndex]?.detail}</p>
      </div>
    </div>
  );
}
