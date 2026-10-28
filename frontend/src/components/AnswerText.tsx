import type { ReactNode } from "react";

// Matches "[1]", "[2, 3]" and "[1,2,3]" — the same forms the backend's
// grounding check recognizes.
const CITATION_GROUP = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

interface Props {
  text: string;
  sourceCount: number;
  onCite: (sourceNumber: number) => void;
}

// Turns **bold** markers into <strong>; the model sometimes writes them.
function withBold(text: string, keyBase: number): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={`${keyBase}-${i}`} className="font-semibold text-white">
        {part.slice(2, -2)}
      </strong>
    ) : (
      part
    ),
  );
}

// Renders the answer with each [n] marker turned into a button that jumps to
// the matching source card. A marker pointing at a source that does not exist
// is shown struck through so a bad citation is visible instead of trusted.
export function AnswerText({ text, sourceCount, onCite }: Props) {
  const nodes: ReactNode[] = [];
  let last = 0;
  let key = 0;

  for (const match of text.matchAll(CITATION_GROUP)) {
    const start = match.index ?? 0;
    if (start > last) nodes.push(...withBold(text.slice(last, start), key++));

    for (const raw of match[1].split(",")) {
      const n = Number(raw.trim());
      if (n >= 1 && n <= sourceCount) {
        nodes.push(
          <button
            key={key++}
            type="button"
            onClick={() => onCite(n)}
            aria-label={`Go to source ${n}`}
            className="mx-0.5 inline-flex min-w-[1.4rem] -translate-y-px items-center justify-center rounded-md border border-emerald-200/50 bg-emerald-200/20 px-1 text-xs font-semibold text-emerald-100 transition-colors hover:bg-emerald-200/35"
          >
            {n}
          </button>,
        );
      } else {
        nodes.push(
          <span
            key={key++}
            title="This answer cites a source that was not retrieved"
            className="mx-0.5 inline-flex min-w-[1.4rem] items-center justify-center rounded-md border border-red-200/50 bg-red-300/20 px-1 text-xs font-semibold text-red-100 line-through"
          >
            {n}
          </span>,
        );
      }
    }
    last = start + match[0].length;
  }
  if (last < text.length) nodes.push(...withBold(text.slice(last), key));

  return <>{nodes}</>;
}
