import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';

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
  }
  return { default: { Map: MapMock, NavigationControl: class {} } };
});

vi.mock('echarts/core', () => ({
  use: vi.fn(),
  init: () => ({ setOption: vi.fn(), resize: vi.fn(), dispose: vi.fn() }),
}));
vi.mock('echarts/charts', () => ({ LineChart: {} }));
vi.mock('echarts/components', () => ({
  GridComponent: {},
  LegendComponent: {},
  TooltipComponent: {},
}));
vi.mock('echarts/renderers', () => ({ CanvasRenderer: {} }));
