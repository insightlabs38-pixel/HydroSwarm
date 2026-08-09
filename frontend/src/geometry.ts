/**
 * MapLibre GL requires valid geographic coordinates (longitude in
 * [-180, 180], latitude in [-90, 90]) -- it throws synchronously
 * ("Invalid LngLat latitude value") on anything outside that range. Real
 * EPANET .inp files commonly use arbitrary local/engineering-unit
 * coordinates (e.g. feet or meters from an arbitrary origin, often in the
 * thousands) with no real-world georeference at all, which is exactly
 * ui-work.txt 26's "primary network view must work from local geometry" --
 * this console never claimed the map requires true geographic placement.
 * Passing such coordinates straight through as [lng, lat] previously
 * crashed the map for any real backend incident built on a network like
 * that (found via UI-11's real-backend smoke test, not by reading the
 * code).
 */

const SAFE_LNG_RANGE = 170; // stays clear of the antimeridian
const SAFE_LAT_RANGE = 80; // stays clear of the poles / Mercator singularity
/** Half-width, in degrees, of the rescaled local window -- matches the
 * demoFixture's own real-world coordinate scale (a real utility network
 * spans roughly this many degrees), so normalized and real geographic
 * data end up at a similar zoom level. */
const LOCAL_WINDOW_HALF_WIDTH = 0.4;

/**
 * Normalizes a full network's node coordinates together (never one node
 * at a time -- every node must be rescaled by the same factor so the
 * network's real relative shape/angles are preserved, not distorted).
 * Coordinates already within valid geographic range are returned
 * unchanged -- this never overrides a genuinely georeferenced network.
 */
export function normalizeMapCoordinates(
  rawCoordinates: readonly (readonly [number, number])[],
): [number, number][] {
  if (rawCoordinates.length === 0) return [];

  let minLng = Infinity;
  let maxLng = -Infinity;
  let minLat = Infinity;
  let maxLat = -Infinity;
  for (const [lng, lat] of rawCoordinates) {
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
  }

  const withinSafeRange =
    minLng >= -SAFE_LNG_RANGE &&
    maxLng <= SAFE_LNG_RANGE &&
    minLat >= -SAFE_LAT_RANGE &&
    maxLat <= SAFE_LAT_RANGE;
  if (withinSafeRange) {
    return rawCoordinates.map(([lng, lat]) => [lng, lat]);
  }

  const spanX = maxLng - minLng || 1;
  const spanY = maxLat - minLat || 1;
  const scale = (LOCAL_WINDOW_HALF_WIDTH * 2) / Math.max(spanX, spanY);
  const centerX = (minLng + maxLng) / 2;
  const centerY = (minLat + maxLat) / 2;
  return rawCoordinates.map(([x, y]) => [(x - centerX) * scale, (y - centerY) * scale]);
}
