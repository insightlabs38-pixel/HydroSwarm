import { useState } from 'react';

export interface KeyValueEntry {
  key: string;
  label: string;
  value: string | null;
  /** True for long hash-like identifiers: rendered in monospace,
   * shortened for display, and copyable in full (ui-work.txt 14
   * Provenance: "Use shortened hash display but make full values
   * copyable."). */
  hash?: boolean;
}

function shorten(value: string): string {
  return value.length > 16 ? `${value.slice(0, 10)}…${value.slice(-4)}` : value;
}

function CopyableHash({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  const canCopy = typeof navigator !== 'undefined' && Boolean(navigator.clipboard);
  return (
    <span className="hash-value">
      <code title={value}>{shorten(value)}</code>
      {canCopy && (
        <button
          type="button"
          className="copy-hash-button"
          onClick={() => {
            void navigator.clipboard.writeText(value).then(() => {
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1500);
            });
          }}
          aria-label={`Copy full value: ${value}`}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      )}
    </span>
  );
}

export function KeyValueGrid({ entries }: { entries: KeyValueEntry[] }) {
  return (
    <dl className="key-value-grid">
      {entries.map((entry) => (
        <div key={entry.key}>
          <dt>{entry.label}</dt>
          <dd className={entry.hash ? 'mono' : ''}>
            {entry.value === null ? (
              '—'
            ) : entry.hash ? (
              <CopyableHash value={entry.value} />
            ) : (
              entry.value
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}
