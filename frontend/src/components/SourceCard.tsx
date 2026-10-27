import type { Source } from "../api";

interface Props {
  source: Source;
  number: number;
  highlighted: boolean;
}

export function SourceCard({ source, number, highlighted }: Props) {
  return (
    <li
      id={`source-${number}`}
      className={`citation-card scroll-mt-24 rounded-xl border p-5 sm:p-6 ${
        highlighted
          ? "border-sky-400/60 bg-sky-400/[0.08]"
          : "border-white/[0.08] bg-white/[0.025]"
      }`}
    >
      <div className="flex gap-4">
        <span className="font-display flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-sky-500/20 to-indigo-500/20 text-xs font-bold text-sky-300">
          {number}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-[0.95rem] font-medium leading-snug text-slate-100">{source.title}</p>
          {source.snippet && (
            <p className="mt-2 text-sm leading-relaxed text-slate-400">{source.snippet}</p>
          )}
          <a
            href={`https://pubmed.ncbi.nlm.nih.gov/${source.pmid}/`}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 inline-flex items-center gap-1.5 text-sm text-sky-400 transition-colors hover:text-sky-300"
          >
            <span className="text-slate-500">PMID</span>
            <span className="font-mono">{source.pmid}</span>
            <svg
              className="h-3.5 w-3.5 opacity-60"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
              aria-hidden
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
              />
            </svg>
          </a>
        </div>
      </div>
    </li>
  );
}
