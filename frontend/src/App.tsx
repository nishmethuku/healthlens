import { useRef, useState, type FormEvent } from "react";
import { sendFeedback, streamQuery, type Grounding, type Source } from "./api";
import { AnswerText } from "./components/AnswerText";
import { Background, LoadingState, LogoMark } from "./components/Decor";
import { HistoryDrawer } from "./components/HistoryDrawer";
import { SourceCard } from "./components/SourceCard";
import {
  addEntry,
  clearHistory,
  loadHistory,
  removeEntry,
  type HistoryEntry,
} from "./history";

const EXAMPLE_QUESTIONS = [
  "What are the latest treatments for type 2 diabetes?",
  "Does intermittent fasting improve cardiovascular health?",
  "How effective are statins for primary prevention of heart disease?",
  "Compare SGLT2 inhibitors and GLP-1 agonists for kidney outcomes",
];

function GroundingNotice({ grounding, total }: { grounding: Grounding; total: number }) {
  if (grounding.grounded) {
    return (
      <p className="mt-5 text-xs text-slate-500">
        Cites {grounding.cited.length} of {total} retrieved {total === 1 ? "paper" : "papers"}.
        Always read the sources before relying on a claim.
      </p>
    );
  }
  return (
    <div className="mt-5 rounded-xl border border-amber-400/25 bg-amber-950/25 px-4 py-3 text-sm leading-relaxed text-amber-100/85">
      {grounding.invalid.length > 0
        ? `This answer refers to source numbers that were not retrieved (${grounding.invalid.join(", ")}). `
        : "Some statements in this answer are not tied to a cited paper. "}
      Check the sources below before relying on it.
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [decompose, setDecompose] = useState(false);
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [grounding, setGrounding] = useState<Grounding | null>(null);
  const [flagged, setFlagged] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  function resetResult() {
    setAnswer(null);
    setSources([]);
    setGrounding(null);
    setFlagged(false);
    setWarning(null);
    setError(null);
    setQueryId(null);
    setFeedback(null);
    setHighlighted(null);
  }

  async function runQuery(raw: string) {
    const q = raw.trim();
    if (!q || loading) return;

    setQuestion(q);
    setLoading(true);
    resetResult();

    const controller = new AbortController();
    abortRef.current = controller;

    let fullAnswer = "";
    let finalSources: Source[] = [];
    let finalGrounding: Grounding | null = null;
    let failed = false;

    try {
      await streamQuery(q, decompose, controller.signal, {
        onSources: (s, id) => {
          finalSources = s;
          setSources(s);
          setQueryId(id);
        },
        onToken: (text) => {
          fullAnswer += text;
          setAnswer((prev) => (prev ?? "") + text);
        },
        onGrounding: (g) => {
          finalGrounding = g;
          setGrounding(g);
        },
        onFlagged: (warning) => {
          failed = true; // crisis-flagged questions are never saved to history
          setFlagged(true);
          setWarning(warning);
        },
        onError: (detail) => {
          failed = true;
          setError(detail);
        },
        onDone: () => {},
      });
    } catch (err) {
      failed = true;
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        setError(
          err instanceof Error ? err.message : "Something went wrong. Is the backend running?",
        );
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
    }

    if (!failed && fullAnswer) {
      const entry: HistoryEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        question: q,
        answer: fullAnswer,
        sources: finalSources,
        grounding: finalGrounding,
        createdAt: Date.now(),
      };
      setHistory(addEntry(history, entry));
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    void runQuery(question);
  }

  function stop() {
    abortRef.current?.abort();
  }

  function openHistoryEntry(entry: HistoryEntry) {
    resetResult();
    setQuestion(entry.question);
    setAnswer(entry.answer);
    setSources(entry.sources);
    setGrounding(entry.grounding);
    setHistoryOpen(false);
  }

  function citeClicked(n: number) {
    setHighlighted(n);
    const el = document.getElementById(`source-${n}`);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    window.setTimeout(() => setHighlighted((current) => (current === n ? null : current)), 1800);
  }

  async function submitFeedback(rating: "up" | "down") {
    if (!queryId || feedback) return;
    setFeedback(rating); // optimistic — this is a low-stakes UI signal, not a critical write
    await sendFeedback(queryId, rating, (answer ?? "").slice(0, 500));
  }

  async function copyAnswer() {
    if (!answer) return;
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard blocked (insecure context or denied) — nothing useful to show
    }
  }

  const hasResults = (answer !== null || sources.length > 0) && !flagged;
  const showSkeleton = loading && !hasResults && !error;
  const showExamples = !loading && !hasResults && !error && !flagged;

  return (
    <div className="relative min-h-screen text-white">
      <Background />

      <div className="relative mx-auto flex min-h-screen max-w-3xl flex-col px-6 pb-20 pt-6 sm:px-8 sm:pt-8">
        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => setHistoryOpen(true)}
            className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2 text-xs font-medium text-slate-300 backdrop-blur-sm transition-colors hover:border-sky-400/30 hover:text-white"
          >
            History{history.length > 0 ? ` (${history.length})` : ""}
          </button>
        </div>

        <header className="mb-12 mt-4 text-center sm:mb-14">
          <div className="mb-6 flex justify-center">
            <LogoMark />
          </div>
          <h1 className="font-display text-4xl font-bold tracking-tight sm:text-5xl">
            <span className="bg-gradient-to-r from-white via-sky-100 to-indigo-200 bg-clip-text text-transparent">
              HealthLens
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-md text-base leading-relaxed text-slate-400">
            Evidence-based medical answers grounded in PubMed research
          </p>
        </header>

        <main className="flex-1">
          <form onSubmit={handleSubmit} className="search-wrapper">
            <label htmlFor="question" className="sr-only">
              Medical question
            </label>
            <div className="relative">
              <div className="search-glow-ring pointer-events-none absolute -inset-[1px] rounded-[1.25rem] bg-gradient-to-r from-sky-500/60 via-indigo-500/60 to-violet-500/60 blur-[2px]" />
              <div className="relative flex flex-col gap-3 rounded-[1.2rem] border border-white/10 bg-white/[0.04] p-2 backdrop-blur-xl sm:flex-row sm:items-center sm:p-2.5">
                <input
                  id="question"
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a medical question…"
                  className="min-h-[3.25rem] flex-1 rounded-xl bg-transparent px-5 py-4 text-[1.05rem] text-white placeholder:text-slate-500 outline-none sm:min-h-[3.5rem]"
                  disabled={loading}
                  autoComplete="off"
                />
                {loading ? (
                  <button
                    type="button"
                    onClick={stop}
                    className="font-display min-h-[3.25rem] shrink-0 rounded-xl border border-white/15 bg-white/[0.06] px-8 py-3.5 text-sm font-semibold tracking-wide text-white transition-colors hover:bg-white/10 sm:min-h-[3.5rem] sm:px-10"
                  >
                    <span className="flex items-center justify-center gap-2">
                      <span
                        className="inline-block h-4 w-4 rounded-full border-2 border-white/30 border-t-white"
                        style={{ animation: "spin-slow 0.8s linear infinite" }}
                      />
                      Stop
                    </span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!question.trim()}
                    className="btn-gradient font-display min-h-[3.25rem] shrink-0 rounded-xl px-8 py-3.5 text-sm font-semibold tracking-wide text-white sm:min-h-[3.5rem] sm:px-10"
                  >
                    Ask
                  </button>
                )}
              </div>
            </div>

            <label className="mt-4 flex cursor-pointer items-start justify-center gap-2.5 text-xs text-slate-400">
              <input
                type="checkbox"
                checked={decompose}
                onChange={(e) => setDecompose(e.target.checked)}
                disabled={loading}
                className="mt-0.5 h-3.5 w-3.5 accent-sky-500"
              />
              <span>
                Complex question mode
                <span className="text-slate-500"> (splits it into parts before searching)</span>
              </span>
            </label>
          </form>

          <p className="mt-5 text-center text-xs leading-relaxed text-slate-500">
            Not medical advice. For emergencies, call{" "}
            <span className="text-slate-400">911</span> or{" "}
            <span className="text-slate-400">988</span>.
          </p>

          {showExamples && (
            <div className="mt-12">
              <p className="mb-4 text-center text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">
                Try asking
              </p>
              <div className="flex flex-wrap justify-center gap-2.5">
                {EXAMPLE_QUESTIONS.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => void runQuery(example)}
                    className="rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-left text-sm text-slate-300 transition-colors hover:border-sky-400/30 hover:text-white"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {showSkeleton && <LoadingState />}

          {error && !loading && (
            <div className="mt-14 rounded-2xl border border-red-400/20 bg-red-950/30 px-6 py-5 backdrop-blur-sm">
              <p className="font-display text-sm font-semibold text-red-300">Something went wrong</p>
              <p className="mt-2 text-sm leading-relaxed text-red-200/80">{error}</p>
            </div>
          )}

          {flagged && !loading && (
            <div className="mt-14 rounded-2xl border border-amber-400/25 bg-amber-950/25 px-6 py-6 backdrop-blur-sm">
              <div className="mb-2 flex items-center gap-2">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-400/15 text-amber-300">
                  !
                </span>
                <h2 className="font-display text-lg font-semibold text-amber-200">
                  Unable to process this question
                </h2>
              </div>
              <p className="text-sm leading-relaxed text-amber-100/85">{warning}</p>
            </div>
          )}

          {hasResults && (
            <section className="mt-14 space-y-10">
              <article className="rounded-2xl border border-white/[0.09] bg-white/[0.03] p-6 shadow-xl shadow-black/20 backdrop-blur-md sm:p-10">
                <div className="mb-6 flex items-center gap-3">
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/15 to-transparent" />
                  <span className="font-display text-xs font-semibold uppercase tracking-[0.2em] text-sky-400/80">
                    Answer
                  </span>
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/15 to-transparent" />
                </div>
                <div className="answer-text whitespace-pre-wrap">
                  <AnswerText text={answer ?? ""} sourceCount={sources.length} onCite={citeClicked} />
                  {loading && (
                    <span
                      className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-sky-400"
                      style={{ animation: "pulse-glow 1s ease-in-out infinite" }}
                      aria-hidden
                    />
                  )}
                </div>

                {!loading && grounding && (
                  <GroundingNotice grounding={grounding} total={sources.length} />
                )}

                {!loading && answer && (
                  <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-white/[0.06] pt-5">
                    {queryId && (
                      <>
                        <span className="text-xs text-slate-500">Was this answer helpful?</span>
                        <button
                          type="button"
                          onClick={() => void submitFeedback("up")}
                          disabled={feedback !== null}
                          aria-pressed={feedback === "up"}
                          aria-label="This answer was helpful"
                          className={`rounded-lg border p-1.5 transition-colors ${
                            feedback === "up"
                              ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-300"
                              : "border-white/10 bg-white/[0.03] text-slate-400 hover:text-slate-200"
                          } disabled:cursor-default`}
                        >
                          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z" />
                            <path strokeLinecap="round" strokeLinejoin="round" d="M7 22V11" />
                          </svg>
                        </button>
                        <button
                          type="button"
                          onClick={() => void submitFeedback("down")}
                          disabled={feedback !== null}
                          aria-pressed={feedback === "down"}
                          aria-label="This answer was not helpful"
                          className={`rounded-lg border p-1.5 transition-colors ${
                            feedback === "down"
                              ? "border-red-400/40 bg-red-400/10 text-red-300"
                              : "border-white/10 bg-white/[0.03] text-slate-400 hover:text-slate-200"
                          } disabled:cursor-default`}
                        >
                          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z" />
                            <path strokeLinecap="round" strokeLinejoin="round" d="M17 2v11" />
                          </svg>
                        </button>
                        {feedback && <span className="text-xs text-slate-500">Thanks for the feedback.</span>}
                      </>
                    )}
                    <button
                      type="button"
                      onClick={() => void copyAnswer()}
                      className="ml-auto rounded-lg border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-slate-400 transition-colors hover:text-slate-200"
                    >
                      {copied ? "Copied" : "Copy answer"}
                    </button>
                  </div>
                )}
              </article>

              {sources.length > 0 && (
                <div>
                  <div className="mb-5 flex items-center justify-between">
                    <h2 className="font-display text-sm font-semibold uppercase tracking-[0.15em] text-slate-400">
                      Cited sources
                    </h2>
                    <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400">
                      {sources.length} {sources.length === 1 ? "paper" : "papers"}
                    </span>
                  </div>
                  <ul className="grid gap-4">
                    {sources.map((src, i) => (
                      <SourceCard
                        key={src.pmid}
                        source={src}
                        number={i + 1}
                        highlighted={highlighted === i + 1}
                      />
                    ))}
                  </ul>
                </div>
              )}
            </section>
          )}
        </main>

        <footer className="mt-20 text-center text-xs text-slate-600">
          Powered by PubMed · Hybrid retrieval · Groq LLM
        </footer>
      </div>

      <HistoryDrawer
        open={historyOpen}
        entries={history}
        onClose={() => setHistoryOpen(false)}
        onSelect={openHistoryEntry}
        onRemove={(id) => setHistory((current) => removeEntry(current, id))}
        onClear={() => setHistory(clearHistory())}
      />
    </div>
  );
}
