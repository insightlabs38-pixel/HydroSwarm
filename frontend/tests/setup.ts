import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

// Most existing component tests exercise LIVE/DEMO_FALLBACK rendering
// downstream of a configured incident, not the SUB-5 first-launch gateway
// itself (see reference/FirstLaunchGateway.test.tsx for that). Stubbing a
// configured incident id here keeps that large existing suite's implicit
// assumption ("an incident is configured; the mocked fetch may succeed or
// fail per test") true globally, matching real deployments that set
// VITE_INCIDENT_ID at build time.
vi.stubEnv('VITE_INCIDENT_ID', 'test-configured-incident');

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverMock);
Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  value: vi.fn(() => ({ measureText: () => ({ width: 10 }) })),
});

vi.mock('maplibre-gl', () => {
  class MapMock {
    layers = new Set<string>();
    addControl() {}
    on(event: string, callback: () => void) {
      if (event === 'load') callback();
    }
    addSource() {}
    addLayer(layer: { id: string }) {
      this.layers.add(layer.id);
    }
    remove() {}
    isStyleLoaded() {
      return true;
    }
    getLayer(id: string) {
      return this.layers.has(id) ? { id } : undefined;
    }
    setLayoutProperty() {}
    setFilter() {}
    fitBounds() {}
    jumpTo() {}
    getSource() {
      return { setData: () => {} };
    }
    getCanvas() {
      return { style: {} };
    }
  }
  class LngLatBoundsMock {
    constructor(
      public sw: [number, number],
      public ne: [number, number],
    ) {}
    extend() {
      return this;
    }
  }
  return {
    default: { Map: MapMock, NavigationControl: class {}, LngLatBounds: LngLatBoundsMock },
  };
});

vi.mock('echarts/core', () => ({
  use: vi.fn(),
  init: vi.fn(() => ({ setOption: vi.fn(), resize: vi.fn(), dispose: vi.fn(), on: vi.fn() })),
}));
vi.mock('echarts/charts', () => ({ LineChart: {}, BarChart: {}, ScatterChart: {} }));
vi.mock('echarts/components', () => ({
  GridComponent: {},
  LegendComponent: {},
  TooltipComponent: {},
}));
vi.mock('echarts/renderers', () => ({ CanvasRenderer: {} }));
