import type { ConsequenceView, FrontierMode, ParetoFrontierEntry } from '../types';
import { request } from './client';

/** Mirrors hydroswarm.domain.schemas.ConsequenceMetrics -- shared shape
 * with api/incident.ts's ApiConsequenceMetrics, but kept local here
 * since the two modules have no other reason to depend on each other. */
interface ApiConsequenceMetrics {
  population_impacted: number;
  contaminant_mass_consumed_mg: number;
  volume_above_threshold_l: number;
  contaminated_pipe_extent_m: number;
  minimum_pressure_m: number;
  pressure_violation_minutes: number;
  unserved_demand_l: number;
  service_availability: number;
  operation_count: number;
  containment_time_minutes: number | null;
  exposure_evaluated: boolean;
  pressure_margin_m: number | null;
  service_availability_margin: number | null;
  numerically_sensitive: boolean;
}

/** Mirrors hydroswarm.api.state.ParetoFrontierEntryView (Pydantic)
 * field-for-field, as returned by GET /incidents/{id}/frontier
 * (ui-work.txt 9.4). */
interface ApiParetoFrontierEntry {
  plan_id: string;
  label: string;
  consequences: ApiConsequenceMetrics;
  mode: FrontierMode;
  dominated: boolean;
  is_no_action_comparator: boolean;
  group: 'EXPOSURE_AWARE' | 'HYDRAULIC_ONLY';
}

function consequenceFromApi(raw: ApiConsequenceMetrics): ConsequenceView {
  return {
    populationImpacted: raw.population_impacted,
    contaminantMassConsumedMg: raw.contaminant_mass_consumed_mg,
    volumeAboveThresholdL: raw.volume_above_threshold_l,
    contaminatedPipeExtentM: raw.contaminated_pipe_extent_m,
    minimumPressureM: raw.minimum_pressure_m,
    pressureViolationMinutes: raw.pressure_violation_minutes,
    unservedDemandL: raw.unserved_demand_l,
    serviceAvailability: raw.service_availability,
    operationCount: raw.operation_count,
    containmentTimeMinutes: raw.containment_time_minutes,
    exposureEvaluated: raw.exposure_evaluated,
    pressureMarginM: raw.pressure_margin_m,
    serviceAvailabilityMargin: raw.service_availability_margin,
    numericallySensitive: raw.numerically_sensitive,
  };
}

function entryFromApi(raw: ApiParetoFrontierEntry): ParetoFrontierEntry {
  return {
    planId: raw.plan_id,
    label: raw.label,
    consequences: consequenceFromApi(raw.consequences),
    mode: raw.mode,
    dominated: raw.dominated,
    isNoActionComparator: raw.is_no_action_comparator,
    group: raw.group,
  };
}

/**
 * Fetch the verified response Pareto frontier for this incident
 * (ui-work.txt 9.4). Only meaningful for a LIVE incident with a real
 * backend -- callers in other data modes should not call this.
 */
export async function fetchParetoFrontier(
  incidentId: string,
  mode: FrontierMode = 'posterior_weighted',
  signal?: AbortSignal,
): Promise<ParetoFrontierEntry[]> {
  const raw = await request<ApiParetoFrontierEntry[]>(
    `/incidents/${incidentId}/frontier?mode=${mode}`,
    signal,
  );
  return raw.map(entryFromApi);
}
