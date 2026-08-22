import { useCallback, useEffect, useRef, useState } from 'react';
import { approvePlan } from '../api/approval';
import {
  analyzeLiveExampleIncident,
  createLiveExampleIncident,
  fetchLiveExampleInputs,
  generateLiveExamplePlans,
  importLiveExampleNetwork,
  recommendLiveExampleSample,
  submitLiveExampleSample,
  verifyLiveExamplePlan,
  type LiveExampleInputs,
  type LiveExamplePlanSummary,
} from '../api/liveExampleFlow';
import { fetchEvidenceCertificate } from '../api/sampling';
import type { EvidenceCertificate } from '../types';
import { selectIncident } from '../incidentSelection';

/** submission.txt SUB-12.1 P1 #4: the "Run Live Example" judge path --
 * a real, sequential drive through the actual production API (network
 * import, incident creation, observation submission, analysis, sampling
 * recommendation, plan generation, exact WNTR verification, approval),
 * using known reference inputs so a judge needs neither their own EPANET
 * network nor real field telemetry. Every stage here is a real network
 * request; nothing is a fixture, a fake wait, or a hard-coded outcome.
 *
 * Mirrors the REFERENCE controller's two-pause shape *only when the real
 * deterministic evidence/sampling policy actually asks for a sample*: once
 * to let the judge trigger evidence collection, once more at the
 * human-approval boundary. The sample-collection pause is conditional --
 * see the real GET /incidents/{id}/evidence-certificate branch below --
 * because a real incident is not guaranteed to need another sample before
 * planning, and one is not guaranteed to have a useful candidate left to
 * offer. Forcing either case through the old unconditional
 * POST /samples/recommend call surfaced the real, correct
 * `marginal_value_below_threshold` / `no_accessible_sample` abstention as
 * a generic "Something went wrong" error; the branch below represents each
 * of those governed outcomes truthfully instead.
 */

export type LiveExampleStage =
  | 'idle'
  | 'importing_network'
  | 'creating_incident'
  | 'analyzing_initial'
  | 'awaiting_sample_collection'
  | 'submitting_sample'
  | 'reanalyzing'
  | 'generating_plans'
  | 'verifying_plans'
  | 'awaiting_approval'
  | 'approving'
  | 'complete'
  /** The real deterministic evidence/sampling policy stopped: no useful
   * sample remains, the sample budget is exhausted, or it abstained
   * outright -- and planning is therefore not permitted either. A real,
   * governed terminal state, not a failure. */
  | 'governed_stop'
  | 'error';

export interface LiveExampleController {
  stage: LiveExampleStage;
  errorMessage: string | null;
  incidentId: string | null;
  recommendedNode: string | null;
  expectedInformationGainBits: number | null;
  plans: LiveExamplePlanSummary[];
  verifiedPlan: LiveExamplePlanSummary | null;
  /** Set only when `stage === 'governed_stop'`: the real Evidence
   * Certificate that explains why planning is currently not permitted. */
  evidenceCertificate: EvidenceCertificate | null;
  /** Advances past the sample-collection pause: submits the real
   * WNTR-simulated concentration for whichever node was recommended. */
  collectSample: () => void;
  /** Advances past the human-approval boundary: records real operator
   * approval of the real verified plan through the real API. */
  approve: () => void;
  /** Restarts the whole flow from scratch (a fresh network import and a
   * fresh incident -- never reuses a partially-completed prior attempt). */
  restart: () => void;
}

const OPERATOR_ID = 'judge-live-example';

export function useLiveExampleFlow(enabled: boolean): LiveExampleController {
  const [stage, setStage] = useState<LiveExampleStage>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [incidentId, setIncidentId] = useState<string | null>(null);
  const [recommendedNode, setRecommendedNode] = useState<string | null>(null);
  const [expectedInformationGainBits, setExpectedInformationGainBits] = useState<number | null>(
    null,
  );
  const [plans, setPlans] = useState<LiveExamplePlanSummary[]>([]);
  const [verifiedPlan, setVerifiedPlan] = useState<LiveExamplePlanSummary | null>(null);
  const [evidenceCertificate, setEvidenceCertificate] = useState<EvidenceCertificate | null>(null);
  const [restartToken, setRestartToken] = useState(0);

  const inputsRef = useRef<LiveExampleInputs | null>(null);

  const fail = useCallback((error: unknown) => {
    setErrorMessage(error instanceof Error ? error.message : String(error));
    setStage('error');
  }, []);

  /** Shared by both the "no sampling needed" (EVIDENCE_SUFFICIENT) path
   * and the ordinary post-sample-collection path: real plan generation,
   * real exact WNTR/EPANET verification of each candidate, then the
   * human-approval pause. Returns false (having already called fail())
   * if verification produced no VERIFIED plan -- that remains a real
   * stop, never a forced approval. */
  const runPlanningAndVerification = useCallback(
    async (id: string): Promise<boolean> => {
      setStage('generating_plans');
      const generated = await generateLiveExamplePlans(id);
      setPlans(generated);

      setStage('verifying_plans');
      let firstVerified: LiveExamplePlanSummary | null = null;
      for (const plan of generated) {
        const { decision } = await verifyLiveExamplePlan(id, plan.planId);
        if (decision === 'VERIFIED' && !firstVerified) {
          firstVerified = plan;
        }
      }
      if (!firstVerified) {
        fail(
          new Error(
            'no generated plan was VERIFIED by exact WNTR verification -- correct behavior ' +
              'here is to stop, not to force an unverified plan through',
          ),
        );
        return false;
      }
      setVerifiedPlan(firstVerified);
      setStage('awaiting_approval');
      return true;
    },
    [fail],
  );

  /** Real branch point, driven by the real, already-authoritative
   * GET /incidents/{id}/evidence-certificate (the same certificate the
   * ordinary LIVE Sampling workspace renders -- see SamplingWorkspace.tsx
   * and hydroswarm.inference.evidence_certificate.build_evidence_certificate
   * on the backend):
   *  - EVIDENCE_SUFFICIENT: the planning gate is already satisfied; no
   *    sample-collection pause is truthful here, so skip straight to real
   *    plan generation/verification.
   *  - CONTINUE_SAMPLING: unchanged from before -- request the real
   *    sampling recommendation and pause for the judge to collect it.
   *  - Any STOP_* status: planning is not permitted. This is a real,
   *    governed terminal state (`governed_stop`), not an error -- no
   *    sample is fabricated and no plan is generated just to keep the
   *    demo moving. */
  const advanceFromEvidence = useCallback(
    async (id: string): Promise<void> => {
      const certificate = await fetchEvidenceCertificate(id);
      if (certificate.status === 'EVIDENCE_SUFFICIENT') {
        await runPlanningAndVerification(id);
        return;
      }
      if (certificate.status === 'CONTINUE_SAMPLING') {
        const recommendation = await recommendLiveExampleSample(id);
        setRecommendedNode(recommendation.nodeId);
        setExpectedInformationGainBits(recommendation.expectedInformationGain);
        setStage('awaiting_sample_collection');
        return;
      }
      // STOP_BUDGET_EXHAUSTED / STOP_NO_USEFUL_CANDIDATE / STOP_ABSTAIN.
      setEvidenceCertificate(certificate);
      setStage('governed_stop');
    },
    [runPlanningAndVerification],
  );

  // Guards against re-starting the flow on every stage transition: `run()`
  // below calls setStage() repeatedly as it progresses, and if `stage`
  // were an effect dependency (it originally was, as a "only start once
  // we're idle" guard), each of those own updates would re-trigger this
  // effect, whose cleanup would cancel the very run that scheduled the
  // update -- silently killing the flow after its first stage transition.
  // hasStartedRef -- not `stage` -- is what actually gates "have we
  // already kicked off a run for this enabled/restartToken generation".
  const hasStartedRef = useRef(false);

  useEffect(() => {
    if (!enabled || hasStartedRef.current) return;
    hasStartedRef.current = true;
    let cancelled = false;

    async function run() {
      try {
        setStage('importing_network');
        const inputs = await fetchLiveExampleInputs();
        inputsRef.current = inputs;
        const network = await importLiveExampleNetwork(inputs);
        if (cancelled) return;

        setStage('creating_incident');
        const newIncidentId = await createLiveExampleIncident(network.networkId, inputs);
        if (cancelled) return;
        setIncidentId(newIncidentId);

        setStage('analyzing_initial');
        await analyzeLiveExampleIncident(newIncidentId);
        if (cancelled) return;

        await advanceFromEvidence(newIncidentId);
      } catch (error) {
        if (!cancelled) fail(error);
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [enabled, restartToken, fail, advanceFromEvidence]);

  const collectSample = useCallback(() => {
    const inputs = inputsRef.current;
    if (!incidentId || !recommendedNode || !inputs || stage !== 'awaiting_sample_collection') {
      return;
    }
    const concentration = inputs.candidateSignaturesMgL[recommendedNode];
    if (typeof concentration !== 'number') {
      fail(
        new Error(`no real reference signature available for recommended node ${recommendedNode}`),
      );
      return;
    }

    (async () => {
      try {
        setStage('submitting_sample');
        await submitLiveExampleSample(incidentId, recommendedNode, concentration);

        setStage('reanalyzing');
        await analyzeLiveExampleIncident(incidentId);

        await runPlanningAndVerification(incidentId);
      } catch (error) {
        fail(error);
      }
    })();
  }, [incidentId, recommendedNode, stage, fail, runPlanningAndVerification]);

  const approve = useCallback(() => {
    if (!incidentId || !verifiedPlan || stage !== 'awaiting_approval') return;
    (async () => {
      try {
        setStage('approving');
        await approvePlan(incidentId, verifiedPlan.planId, OPERATOR_ID);
        selectIncident(incidentId);
        setStage('complete');
      } catch (error) {
        fail(error);
      }
    })();
  }, [incidentId, verifiedPlan, stage, fail]);

  const restart = useCallback(() => {
    inputsRef.current = null;
    hasStartedRef.current = false;
    setErrorMessage(null);
    setIncidentId(null);
    setRecommendedNode(null);
    setExpectedInformationGainBits(null);
    setPlans([]);
    setVerifiedPlan(null);
    setEvidenceCertificate(null);
    setStage('idle');
    setRestartToken((token) => token + 1);
  }, []);

  return {
    stage,
    errorMessage,
    incidentId,
    recommendedNode,
    expectedInformationGainBits,
    plans,
    verifiedPlan,
    evidenceCertificate,
    collectSample,
    approve,
    restart,
  };
}
