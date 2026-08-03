export type OodLevel = 'NORMAL' | 'CAUTION' | 'OUTSIDE_VALIDATED_RANGE';
export type PlanStatus = 'REJECTED' | 'RECOMMENDED' | 'VALID';

export interface NetworkNode {
  id: string;
  kind: 'junction' | 'reservoir' | 'tank';
  coordinates: [number, number];
  probability: number;
  concentration: number;
  candidate: boolean;
  sensor?: SensorState;
}

export interface NetworkLink {
  id: string;
  source: string;
  target: string;
  flow: number;
  concentration: number;
  action?: 'CLOSE' | 'FLUSH';
}

export interface SensorState {
  id: string;
  health: 'HEALTHY' | 'DRIFT' | 'MISSING';
  quality: number;
  ageMinutes: number;
  pressure: number;
  concentration: number;
}

export interface Candidate {
  nodeId: string;
  probability: number;
}

export interface Plan {
  id: string;
  name: string;
  exposureReduction: number;
  pressureViolations: number;
  serviceAvailability: number;
  actions: number;
  containmentMinutes: number;
  status: PlanStatus;
  rejectionReason?: string;
}

export interface AuditEvent {
  sequence: number;
  timestamp: string;
  type: string;
  actor: string;
  detail: string;
}

export interface Benchmark {
  metric: string;
  value: string;
  comparison: string;
  status: 'PASS' | 'WATCH';
}

export interface IncidentView {
  id: string;
  networkId: string;
  status: 'SAMPLING' | 'PLANNING' | 'APPROVAL' | 'CLOSED';
  source: 'api' | 'demo-fallback';
  offline: boolean;
  runtimeMs: number;
  modelVersion: string;
  ood: OodLevel;
  approvalPending: boolean;
  candidateCoverage: number;
  disagreement: number;
  nodes: NetworkNode[];
  links: NetworkLink[];
  candidates: Candidate[];
  recommendedSample: {
    nodeId: string;
    informationGain: number;
    delayMinutes: number;
    cost: number;
    rationale: string;
  };
  evidence: {
    before: Candidate[];
    after: Candidate[];
    uncertaintyReduction: number;
    nodesRemoved: number;
  };
  plans: Plan[];
  audit: AuditEvent[];
  benchmarks: Benchmark[];
  explanation: string;
}
