import type { ReactNode } from 'react';

export function Panel({
  title,
  eyebrow,
  children,
  className = '',
}: {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`panel ${className}`}
      aria-labelledby={`panel-${title.replaceAll(' ', '-').toLowerCase()}`}
    >
      <header className="panel-header">
        <div>
          {eyebrow && <p className="eyebrow">{eyebrow}</p>}
          <h2 id={`panel-${title.replaceAll(' ', '-').toLowerCase()}`}>{title}</h2>
        </div>
      </header>
      {children}
    </section>
  );
}
