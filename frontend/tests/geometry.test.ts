import { describe, expect, test } from 'vitest';
import { normalizeMapCoordinates } from '../src/geometry';

describe('normalizeMapCoordinates', () => {
  test('passes real geographic coordinates through unchanged', () => {
    const raw: [number, number][] = [
      [-80.01, 35.01],
      [-79.995, 35.015],
    ];
    expect(normalizeMapCoordinates(raw)).toEqual(raw);
  });

  test('rescales out-of-range engineering-unit coordinates into a valid MapLibre range', () => {
    // Real values from data/frozen/golden_network.inp, which crashed
    // MapLibre with "Invalid LngLat latitude value" before this fix --
    // raw Y goes up to 1450, far outside the valid [-90, 90] latitude range.
    const raw: [number, number][] = [
      [0, 0],
      [1000, 0],
      [2000, 0],
      [3000, 650],
      [1750, 1250],
      [2650, 1450],
    ];
    const result = normalizeMapCoordinates(raw);
    expect(result).toHaveLength(raw.length);
    for (const [lng, lat] of result) {
      expect(lng).toBeGreaterThanOrEqual(-180);
      expect(lng).toBeLessThanOrEqual(180);
      expect(lat).toBeGreaterThanOrEqual(-90);
      expect(lat).toBeLessThanOrEqual(90);
    }
  });

  test('preserves relative shape (uniform scale, not independently stretched per axis)', () => {
    // A 2:1 rectangle should stay a 2:1 rectangle after normalization.
    const raw: [number, number][] = [
      [0, 0],
      [2000, 0],
      [2000, 1000],
      [0, 1000],
    ];
    const [a, b, , d] = normalizeMapCoordinates(raw);
    const width = b[0] - a[0];
    const height = d[1] - a[1];
    expect(width / height).toBeCloseTo(2, 5);
  });

  test('a single node does not divide by zero', () => {
    const raw: [number, number][] = [[5000, 5000]];
    const result = normalizeMapCoordinates(raw);
    expect(result).toHaveLength(1);
    expect(Number.isFinite(result[0][0])).toBe(true);
    expect(Number.isFinite(result[0][1])).toBe(true);
  });

  test('empty input returns empty output', () => {
    expect(normalizeMapCoordinates([])).toEqual([]);
  });

  test('coordinates already spanning the whole safe range are left alone, not rescaled', () => {
    const raw: [number, number][] = [
      [-170, -80],
      [170, 80],
    ];
    expect(normalizeMapCoordinates(raw)).toEqual(raw);
  });
});
