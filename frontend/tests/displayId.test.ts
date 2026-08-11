import { expect, test } from 'vitest';
import { formatDisplayId } from '../src/displayId';

test('formats long IDs compactly without changing short IDs', () => {
  expect(formatDisplayId('5bf2cfe9d4184e049ae86e7831e402aa')).toBe('5bf2cfe9…02aa');
  expect(formatDisplayId('plan-A')).toBe('plan-A');
});
