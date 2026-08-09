import { beforeEach, describe, expect, test, vi } from 'vitest';

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Error',
    json: async () => body,
  } as Response;
}

beforeEach(() => {
  vi.resetModules();
  window.history.pushState(null, '', '/');
});

describe('fetchIncident (no VITE_INCIDENT_ID configured, matching this test env)', () => {
  test('throws IncidentUnavailableError when the API is reachable but no incident is configured', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' })));
    const { fetchIncident, IncidentUnavailableError } = await import('../src/api/incident');
    await expect(fetchIncident()).rejects.toBeInstanceOf(IncidentUnavailableError);
  });
});

describe('fetchIncidentWithFallback', () => {
  test('falls back to DEMO_FALLBACK when the API is entirely unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
    const { fetchIncidentWithFallback } = await import('../src/api/incident');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('DEMO_FALLBACK');
    expect(incident.modeReason).toBeTruthy();
  });

  test('returns ERROR mode (not DEMO_FALLBACK) when the API is reachable but the incident cannot be resolved', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' })));
    const { fetchIncidentWithFallback } = await import('../src/api/incident');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('ERROR');
    expect(incident.modeReason).toMatch(/No active incident configured/);
  });

  test('never labels any fallback result as LIVE', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
    const { fetchIncidentWithFallback } = await import('../src/api/incident');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).not.toBe('LIVE');
  });
});

/** Minimal but structurally complete ApiIncidentView payload (overnight-plan.txt
 * Task 3.2), for exercising fetchIncident()/viewFromApi() against the real
 * GET /incidents/{id}/view contract. */
function apiIncidentViewFixture(): unknown {
  return {
    schema_version: 1,
    incident_id: 'test-incident-id',
    network_id: 'net-x',
    runtime_mode: 'FULL_HYBRID',
    data_mode: 'LIVE',
    controller_state: 'PLANNING',
    generated_at: '2026-08-03T08:40:00Z',
    provenance: {
      network_hash: 'n'.repeat(64),
      feature_schema_hash: 'f'.repeat(64),
      model_version: 'hydrocore-hybrid-v1',
      model_checkpoint_hash: 'm'.repeat(64),
      calibration_version: 'hydroswarm-calibration-v1',
      calibration_hash: 'c'.repeat(64),
      simulator: 'wntr',
      simulator_version: '1.2.0',
    },
    candidates: {
      node_probabilities: { J1: 0.7, J2: 0.3 },
      node_ids: ['J1', 'J2'],
      coverage_target: 0.9,
      measured_coverage: 0.91,
      calibrated: true,
    },
    disagreement_js: 0.05,
    ood_level: 'NORMAL',
    calibration_alpha: 0.1,
    nodes: [
      {
        node_id: 'J1',
        node_type: 'junction',
        elevation_m: 10,
        coordinates: [1, 2],
        probability: 0.7,
        concentration_mg_l: 0.3,
        candidate: true,
      },
      {
        node_id: 'J2',
        node_type: 'junction',
        elevation_m: 12,
        coordinates: [3, 4],
        probability: 0.3,
        concentration_mg_l: 0.1,
        candidate: true,
      },
    ],
    links: [{ link_id: 'P1', link_type: 'pipe', start_node: 'J1', end_node: 'J2' }],
    sensor_health: [
      {
        sensor_id: 'S1',
        node_id: 'J1',
        health: 'HEALTHY',
        quality: 0.98,
        observed_at: new Date().toISOString(),
        received_at: new Date().toISOString(),
        pressure_m: 28,
        concentration_mg_l: 0.3,
      },
    ],
    sample_recommendation: { node_id: 'J2', expected_information_gain: 0.4, alternatives: [] },
    evidence_history: [
      {
        round_index: 0,
        observation_count: 3,
        valid_concentration_count: 2,
        sensor_nodes: ['S1'],
        evidence_hash: 'e'.repeat(64),
      },
    ],
    plans: [
      {
        plan: {
          plan_id: 'plan-a',
          name: 'Isolate J2',
          actions: [
            {
              action_type: 'CLOSE_PIPE',
              target_id: 'P1',
              start_minute: 0,
              duration_minutes: 30,
              flow_rate_lps: null,
            },
          ],
        },
        verification: {
          decision: 'VERIFIED',
          simulator: 'wntr',
          simulator_version: '1.2.0',
          state_hash: 's'.repeat(64),
          consequences: {
            population_impacted: 10,
            contaminant_mass_consumed_mg: 5,
            volume_above_threshold_l: 2,
            contaminated_pipe_extent_m: 100,
            minimum_pressure_m: 20,
            pressure_violation_minutes: 0,
            unserved_demand_l: 0,
            service_availability: 0.99,
            operation_count: 1,
            containment_time_minutes: 12,
            exposure_evaluated: true,
            pressure_margin_m: 5,
            service_availability_margin: 0.04,
            numerically_sensitive: false,
          },
          worst_case_consequences: null,
          evaluation_provenance: null,
          rejection_codes: [],
          abstention_reason: null,
          verified_at: '2026-08-03T08:00:00Z',
          context_hash: 'ctx-plan-a',
          verification_status: 'CURRENT',
        },
      },
    ],
    selected_plan_id: null,
    recommended_plan_id: 'plan-a',
    counterfactual_consequences: {
      'plan-a': {
        population_impacted: 10,
        contaminant_mass_consumed_mg: 5,
        volume_above_threshold_l: 2,
        contaminated_pipe_extent_m: 100,
        minimum_pressure_m: 20,
        pressure_violation_minutes: 0,
        unserved_demand_l: 0,
        service_availability: 0.99,
        operation_count: 1,
        containment_time_minutes: 12,
        exposure_evaluated: true,
        pressure_margin_m: 5,
        service_availability_margin: 0.04,
        numerically_sensitive: false,
      },
    },
    explanations: [
      { intent: 'WHY_SOURCE', text: 'J1 is the leading source.', facts: {}, limitations: [] },
    ],
    audit_events: [
      {
        sequence: 1,
        timestamp: '2026-08-03T08:00:00',
        event_type: 'INCIDENT_CREATED',
        actor: 'OPERATOR',
        payload: {},
      },
    ],
    runtime_metrics_ms: { hydraulic_simulation: 12.5, neural_inference: 3.2 },
    simulator_budget: {
      exact_simulations_used: 1,
      plans_exactly_verified: 1,
      exact_simulation_cache_hits: 0,
      remaining_epanet_budget: 2,
    },
  };
}

describe('fetchIncident with an incident configured (VITE_INCIDENT_ID stubbed)', () => {
  test('maps a complete /view response into a LIVE IncidentView', async () => {
    vi.stubEnv('VITE_INCIDENT_ID', 'test-incident-id');
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (String(url).includes('/health')) return Promise.resolve(jsonResponse({ status: 'ok' }));
        return Promise.resolve(jsonResponse(apiIncidentViewFixture()));
      }),
    );
    const { fetchIncident } = await import('../src/api/incident');
    const incident = await fetchIncident();
    expect(incident.mode).toBe('LIVE');
    expect(incident.id).toBe('test-incident-id');
    expect(incident.nodes).toHaveLength(2);
    expect(incident.plans).toHaveLength(1);
    expect(incident.plans[0].status).toBe('RECOMMENDED');
    expect(incident.recommendedPlanId).toBe('plan-a');
    expect(incident.counterfactuals['plan-a'].serviceAvailability).toBe(0.99);
    expect(incident.provenance.calibrationHash).toBe('c'.repeat(64));
    // "pre-UI-11" fix B: the exact-simulation budget the /view response
    // reports is passed through unmodified, never recomputed or defaulted.
    expect(incident.simulatorBudget).toEqual({
      exactSimulationsUsed: 1,
      plansExactlyVerified: 1,
      exactSimulationCacheHits: 0,
      remainingEpanetBudget: 2,
    });
  });

  test('maps a null sample_recommendation to null, not a zero-filled placeholder', async () => {
    // core-issues.txt: represent unknown/absent values as null, never a
    // fabricated zero -- a genuinely absent recommendation must not become
    // { nodeId: '', informationGain: 0, ... }, which a consumer could
    // mistake for a real recommendation.
    vi.stubEnv('VITE_INCIDENT_ID', 'test-incident-id');
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (String(url).includes('/health')) return Promise.resolve(jsonResponse({ status: 'ok' }));
        return Promise.resolve(
          jsonResponse({ ...(apiIncidentViewFixture() as object), sample_recommendation: null }),
        );
      }),
    );
    const { fetchIncident } = await import('../src/api/incident');
    const incident = await fetchIncident();
    expect(incident.recommendedSample).toBeNull();
  });

  test('fetchIncidentWithFallback returns the live-mapped incident, not the demo fixture', async () => {
    vi.stubEnv('VITE_INCIDENT_ID', 'test-incident-id');
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (String(url).includes('/health')) return Promise.resolve(jsonResponse({ status: 'ok' }));
        return Promise.resolve(jsonResponse(apiIncidentViewFixture()));
      }),
    );
    const { fetchIncidentWithFallback } = await import('../src/api/incident');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('LIVE');
    expect(incident.id).toBe('test-incident-id');
  });

  test('falls back to DEMO_FALLBACK when the /view response is malformed', async () => {
    vi.stubEnv('VITE_INCIDENT_ID', 'test-incident-id');
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (String(url).includes('/health')) return Promise.resolve(jsonResponse({ status: 'ok' }));
        return Promise.resolve(jsonResponse({ not: 'a valid incident view' }));
      }),
    );
    const { fetchIncidentWithFallback } = await import('../src/api/incident');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('DEMO_FALLBACK');
  });
});

describe('failure injection (overnight-plan.txt Task 3.8)', () => {
  test.each([
    'missing_checkpoint',
    'corrupt_checkpoint_hash',
    'incompatible_feature_schema',
    'corrupt_calibration',
    'unknown_topology',
    'wntr_timeout',
    'incomplete_simulator_output',
    'severe_sensor_dropout',
    'no_valid_plan',
  ] as const)(
    '%s renders ERROR mode with an explicit, distinct reason -- never LIVE',
    async (category) => {
      window.history.pushState(null, '', `/?failure=${category}`);
      // Even an otherwise-healthy API must not override an injected failure.
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' })));
      const { fetchIncidentWithFallback } = await import('../src/api/incident');
      const incident = await fetchIncidentWithFallback();
      expect(incident.mode).toBe('ERROR');
      expect(incident.mode).not.toBe('LIVE');
      expect(incident.modeReason).toContain(category);
    },
  );

  test.each(['missing_checkpoint', 'no_valid_plan'] as const)(
    '%s: ERROR mode carries no demo plans, candidates, or sample recommendation',
    async (category) => {
      // core-issues.txt: ERROR mode must not render demo plans, candidates,
      // or operational recommendations -- previously this spread the whole
      // fabricated demoIncident fixture and only overrode `mode`, so a
      // full set of fake plans/candidates/recommendations rendered right
      // alongside the error banner.
      window.history.pushState(null, '', `/?failure=${category}`);
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' })));
      const { fetchIncidentWithFallback } = await import('../src/api/incident');
      const incident = await fetchIncidentWithFallback();
      expect(incident.mode).toBe('ERROR');
      expect(incident.plans).toEqual([]);
      expect(incident.candidates).toEqual([]);
      expect(incident.nodes).toEqual([]);
      expect(incident.recommendedSample).toBeNull();
      expect(incident.selectedPlanId).toBeNull();
      expect(incident.recommendedPlanId).toBeNull();
      expect(incident.benchmarks).toEqual([]);
      expect(incident.audit).toEqual([]);
      expect(incident.simulatorBudget).toBeNull();
    },
  );

  test('ERROR mode from an unresolvable incident also carries no demo content', async () => {
    vi.unstubAllEnvs(); // a prior test in this file may have stubbed VITE_INCIDENT_ID
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'ok' })));
    const { fetchIncidentWithFallback } = await import('../src/api/incident');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('ERROR');
    expect(incident.plans).toEqual([]);
    expect(incident.candidates).toEqual([]);
    expect(incident.recommendedSample).toBeNull();
  });

  test('an unrecognized failure category is ignored, not silently accepted', async () => {
    window.history.pushState(null, '', '/?failure=not_a_real_category');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
    const { fetchIncidentWithFallback } = await import('../src/api/incident');
    const incident = await fetchIncidentWithFallback();
    expect(incident.mode).toBe('DEMO_FALLBACK');
  });

  test('every failure category has a distinct reason string', async () => {
    const { FAILURE_INJECTION_CATEGORIES } = await import('../src/api/incident');
    window.history.pushState(null, '', '/');
    const reasons = new Set<string>();
    for (const category of FAILURE_INJECTION_CATEGORIES) {
      window.history.pushState(null, '', `/?failure=${category}`);
      vi.resetModules();
      vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline test')));
      const { fetchIncidentWithFallback } = await import('../src/api/incident');
      const incident = await fetchIncidentWithFallback();
      reasons.add(incident.modeReason ?? '');
    }
    expect(reasons.size).toBe(FAILURE_INJECTION_CATEGORIES.length);
  });
});
