import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Primary operator workflow stages plus secondary utilities
 * (ui-work.txt 4, 10). This replaces the old flat page-tab model with
 * the mission-control left workflow rail's stage identity.
 */
export type Workspace =
  | 'incident'
  | 'source'
  | 'sampling'
  | 'response'
  | 'approval'
  | 'replay'
  | 'network'
  | 'validation'
  | 'authority'
  | 'benchmarks';

export const WORKSPACE_LABELS: Record<Workspace, string> = {
  incident: 'Incident',
  source: 'Source',
  sampling: 'Sampling',
  response: 'Response',
  approval: 'Approval',
  replay: 'Replay',
  network: 'Network',
  validation: 'Validation',
  authority: 'Model & Authority',
  benchmarks: 'Benchmarks',
};

export type DockTab =
  'timeline' | 'evidence' | 'hydraulics' | 'verification' | 'provenance' | 'audit';

export interface MapLayerState {
  assets: boolean;
  flow: boolean;
  concentration: boolean;
  candidates: boolean;
  sensors: boolean;
  sample: boolean;
  actions: boolean;
}

const defaultMapLayers: MapLayerState = {
  assets: true,
  flow: true,
  concentration: true,
  candidates: true,
  sensors: true,
  sample: true,
  actions: true,
};

interface ConsoleUiState {
  workspace: Workspace;
  selectedNodeId: string | null;
  selectedLinkId: string | null;
  selectedPlanId: string | null;
  selectedAuditSequence: number | null;
  mapLayers: MapLayerState;
  leftRailCollapsed: boolean;
  inspectorCollapsed: boolean;
  dockCollapsed: boolean;
  dockHeight: number;
  dockTab: DockTab;
  replayIndex: number;
  replayPlaying: boolean;
  replaySpeed: 0.5 | 1 | 2 | 4;
  reducedMotion: boolean;

  setWorkspace: (workspace: Workspace) => void;
  selectNode: (nodeId: string | null) => void;
  selectLink: (linkId: string | null) => void;
  selectPlan: (planId: string | null) => void;
  selectAuditSequence: (sequence: number | null) => void;
  setMapLayer: (layer: keyof MapLayerState, visible: boolean) => void;
  toggleLeftRail: () => void;
  toggleInspector: () => void;
  toggleDock: () => void;
  setDockHeight: (height: number) => void;
  setDockTab: (tab: DockTab) => void;
  setReplayIndex: (value: number) => void;
  toggleReplayPlaying: () => void;
  setReplaySpeed: (value: ConsoleUiState['replaySpeed']) => void;
  toggleReducedMotion: () => void;
}

/**
 * ui-work.txt 5: "Persist only layout preferences in localStorage" --
 * everything else here is ephemeral interaction/selection state, never
 * incident payloads, candidates, plan truth, verification truth,
 * approval, or authority certificates (ui-work.txt 10).
 */
export const useConsoleStore = create<ConsoleUiState>()(
  persist(
    (set) => ({
      workspace: 'incident',
      selectedNodeId: null,
      selectedLinkId: null,
      selectedPlanId: null,
      selectedAuditSequence: null,
      mapLayers: defaultMapLayers,
      leftRailCollapsed: false,
      inspectorCollapsed: false,
      dockCollapsed: false,
      dockHeight: 240,
      dockTab: 'timeline',
      replayIndex: 0,
      replayPlaying: false,
      replaySpeed: 1,
      reducedMotion: false,

      setWorkspace: (workspace) => set({ workspace }),
      selectNode: (selectedNodeId) => set({ selectedNodeId }),
      selectLink: (selectedLinkId) => set({ selectedLinkId }),
      selectPlan: (selectedPlanId) => set({ selectedPlanId }),
      selectAuditSequence: (selectedAuditSequence) => set({ selectedAuditSequence }),
      setMapLayer: (layer, visible) =>
        set((state) => ({ mapLayers: { ...state.mapLayers, [layer]: visible } })),
      toggleLeftRail: () => set((state) => ({ leftRailCollapsed: !state.leftRailCollapsed })),
      toggleInspector: () => set((state) => ({ inspectorCollapsed: !state.inspectorCollapsed })),
      toggleDock: () => set((state) => ({ dockCollapsed: !state.dockCollapsed })),
      setDockHeight: (dockHeight) => set({ dockHeight }),
      setDockTab: (dockTab) => set({ dockTab, dockCollapsed: false }),
      setReplayIndex: (replayIndex) => set({ replayIndex }),
      toggleReplayPlaying: () => set((state) => ({ replayPlaying: !state.replayPlaying })),
      setReplaySpeed: (replaySpeed) => set({ replaySpeed }),
      toggleReducedMotion: () => set((state) => ({ reducedMotion: !state.reducedMotion })),
    }),
    {
      name: 'hydroswarm-console-layout',
      partialize: (state) => ({
        leftRailCollapsed: state.leftRailCollapsed,
        inspectorCollapsed: state.inspectorCollapsed,
        dockCollapsed: state.dockCollapsed,
        dockHeight: state.dockHeight,
        dockTab: state.dockTab,
        reducedMotion: state.reducedMotion,
      }),
    },
  ),
);
