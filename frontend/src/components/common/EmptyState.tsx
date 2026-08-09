/**
 * A legitimate "nothing to show, and here is exactly why" state
 * (ui-work.txt 23: "Every major component needs: loading; empty; error;
 * suppressed/abstained where applicable"). Never used to paper over a
 * missing value with fabricated content -- the whole point is to say so
 * honestly instead.
 */
export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="empty-state" role="status">
      <p className="empty-state-title">{title}</p>
      {detail && <p className="supporting">{detail}</p>}
    </div>
  );
}
