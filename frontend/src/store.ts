import { create } from 'zustand';

export type Page = 'overview' | 'audit' | 'validation' | 'benchmarks' | 'topology';

interface ConsoleState {
  page: Page;
  timeIndex: number;
  playing: boolean;
  speed: number;
  reducedMotion: boolean;
  selectedPlan: string;
  setPage: (page: Page) => void;
  setTimeIndex: (value: number) => void;
  togglePlaying: () => void;
  setSpeed: (value: number) => void;
  toggleReducedMotion: () => void;
  selectPlan: (id: string) => void;
}

export const useConsoleStore = create<ConsoleState>((set) => ({
  page: 'overview',
  timeIndex: 5,
  playing: false,
  speed: 1,
  reducedMotion: false,
  selectedPlan: 'B',
  setPage: (page) => set({ page }),
  setTimeIndex: (timeIndex) => set({ timeIndex }),
  togglePlaying: () => set((state) => ({ playing: !state.playing })),
  setSpeed: (speed) => set({ speed }),
  toggleReducedMotion: () => set((state) => ({ reducedMotion: !state.reducedMotion })),
  selectPlan: (selectedPlan) => set({ selectedPlan }),
}));
