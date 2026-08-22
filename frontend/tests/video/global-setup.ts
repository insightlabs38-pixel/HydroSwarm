/**
 * Preflight for the demo-clip recording harness. Fails fast (before any
 * browser opens) unless a real HydroSwarm v0.2.0 release instance is
 * already running and reachable -- this harness must never fall back to
 * a mocked or dev-built app for final footage.
 */
const BASE_URL = process.env.HYDROSWARM_RECORDING_BASE_URL ?? 'http://127.0.0.1:8765';
const REQUIRED_VERSION = '0.2.0';

async function getJson(path: string): Promise<unknown> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`GET ${path} returned ${response.status}`);
  }
  return response.json();
}

export default async function globalSetup(): Promise<void> {
  let health: { version?: string; status?: string };
  try {
    health = (await getJson('/api/health')) as { version?: string; status?: string };
  } catch (error) {
    throw new Error(
      `Cannot reach HydroSwarm at ${BASE_URL}/api/health. Start the real published release ` +
        `first, e.g.:\n  docker compose -f docker-compose.release.yml up\n` +
        `Underlying error: ${(error as Error).message}`,
    );
  }
  if (health.version !== REQUIRED_VERSION) {
    throw new Error(
      `Refusing to record: ${BASE_URL}/api/health reports version "${health.version}", ` +
        `expected "${REQUIRED_VERSION}". This harness only records against the real published ` +
        `v0.2.0 release, never a dev/preview build.`,
    );
  }

  try {
    await getJson('/api/reference-demo');
  } catch (error) {
    throw new Error(
      `Reference Incident did not load from ${BASE_URL}/api/reference-demo -- cannot record ` +
        `REFERENCE clips. Underlying error: ${(error as Error).message}`,
    );
  }

  try {
    await getJson('/api/live-example-inputs');
  } catch (error) {
    throw new Error(
      `Live Example inputs did not load from ${BASE_URL}/api/live-example-inputs -- cannot ` +
        `record the LIVE clip. Underlying error: ${(error as Error).message}`,
    );
  }

  console.log(
    `[record-demo-clips] preflight OK -- ${BASE_URL} is real HydroSwarm v${health.version}, ` +
      `Reference Incident and Live Example inputs both load.`,
  );
}
