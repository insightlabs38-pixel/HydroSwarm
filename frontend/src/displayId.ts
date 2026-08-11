/** Compact IDs for high-level operational surfaces. The underlying value is
 * never changed; callers retain it in a title/accessible description for
 * forensic inspection. */
export function formatDisplayId(value: string, prefix = 8, suffix = 4): string {
  if (value.length <= prefix + suffix + 1) return value;
  return `${value.slice(0, prefix)}…${value.slice(-suffix)}`;
}
