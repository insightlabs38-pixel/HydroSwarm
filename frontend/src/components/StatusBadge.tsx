import type { ReactNode } from 'react';

interface Props {
  tone: 'good' | 'warn' | 'danger' | 'info';
  children: ReactNode;
  label?: string;
}

export function StatusBadge({ tone, children, label }: Props) {
  return (
    <span className={`status-badge status-${tone}`} aria-label={label}>
      <span className="status-icon" aria-hidden="true" />
      {children}
    </span>
  );
}
