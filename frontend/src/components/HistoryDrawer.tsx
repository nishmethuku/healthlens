import type { HistoryEntry } from "../history";

interface Props {
  open: boolean;
  entries: HistoryEntry[];
  onClose: () => void;
  onSelect: (entry: HistoryEntry) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}

function formatWhen(timestamp: number): string {
  return new Date(timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function HistoryDrawer({ open, entries, onClose, onSelect, onRemove, onClear }: Props) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        type="button"
        aria-label="Close history"
        onClick={onClose}
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
      />
      <aside
        role="dialog"
        aria-label="Question history"
        className="relative flex h-full w-full max-w-sm flex-col border-l border-white/10 bg-[#0a1424] shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-white/[0.08] px-5 py-4">
          <h2 className="font-display text-sm font-semibold uppercase tracking-[0.15em] text-slate-300">
            History
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-white/10 px-3 py-1 text-xs text-slate-400 hover:text-slate-200"
          >
            Close
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-3">
          {entries.length === 0 ? (
            <p className="px-2 py-6 text-sm leading-relaxed text-slate-500">
              Questions you ask will show up here. History is saved only in this browser.
            </p>
          ) : (
            <ul className="space-y-2">
              {entries.map((entry) => (
                <li
                  key={entry.id}
                  className="group flex items-start gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-1 hover:border-sky-400/25"
                >
                  <button
                    type="button"
                    onClick={() => onSelect(entry)}
                    className="min-w-0 flex-1 rounded-lg px-3 py-2 text-left"
                  >
                    <p className="line-clamp-2 text-sm leading-snug text-slate-200">{entry.question}</p>
                    <p className="mt-1 text-xs text-slate-500">
                      {formatWhen(entry.createdAt)} · {entry.sources.length}{" "}
                      {entry.sources.length === 1 ? "paper" : "papers"}
                    </p>
                  </button>
                  <button
                    type="button"
                    onClick={() => onRemove(entry.id)}
                    aria-label={`Delete "${entry.question}" from history`}
                    className="mr-1 mt-1 rounded-md px-2 py-1 text-xs text-slate-500 hover:text-red-300"
                  >
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {entries.length > 0 && (
          <div className="border-t border-white/[0.08] px-5 py-4">
            <button
              type="button"
              onClick={onClear}
              className="text-xs text-slate-500 hover:text-red-300"
            >
              Clear all history
            </button>
          </div>
        )}
      </aside>
    </div>
  );
}
