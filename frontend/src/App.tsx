import { useRef, useState, type FormEvent } from "react";
import { sendFeedback, streamQuery, type Grounding, type Source } from "./api";
import { AnswerText } from "./components/AnswerText";
import {
  Background,
  HeroArt,
  IconBook,
  IconQuote,
  IconShield,
  LoadingState,
  LogoMark,
  PulseLine,
} from "./components/Decor";
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

const FEATURES = [
  {
    icon: <IconBook />,
    title: "Grounded in PubMed",
    text: "Answers are written only from the abstracts retrieved for your question.",
  },
  {
    icon: <IconQuote />,
    title: "Every claim cited",
    text: "Citations link to the papers, and each answer is checked for sources that do not exist.",
  },
  {
    icon: <IconShield />,
    title: "Safety first",
    text: "Crisis-related questions are blocked and pointed to 911 and 988.",
  },
];

function GroundingNotice({ grounding, total }: { grounding: Grounding; total: number }) {
  if (grounding.grounded) {
    return (
      <p className="mt-5 text-xs text-white/75">
        Cites {grounding.cited.length} of {total} retrieved {total === 1 ? "paper" : "papers"}.
        Always read the sources before relying on a claim.
      </p>
    );
  }
  return (
    <div className="mt-5 rounded-2xl border border-amber-200/50 bg-amber-300/15 px-4 py-3 text-sm leading-relaxed text-amber-50">
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
  // Once there is something to read, shrink the headline so the answer is on screen.
  const compact = hasResults || loading || Boolean(error) || flagged;

  function goToQuestion() {
    window.scrollTo({ top: 0, behavior: "smooth" });
    document.getElementById("question")?.focus();
  }

  function goToSafety() {
    document.getElementById("safety")?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const navItem =
    "rounded-full px-4 py-1.5 text-xs font-medium text-white/90 transition-colors hover:bg-white/20 hover:text-white sm:px-5";

  return (
    <div className="relative min-h-screen text-white">
      <Background />

      <div className="relative mx-auto max-w-5xl px-3 py-4 sm:px-8 sm:py-10">
        <div className="glass-frame rounded-[1.75rem] p-5 sm:rounded-[2.25rem] sm:p-8 lg:p-10">
          <header className="flex items-center justify-between gap-3">
            <LogoMark />

            <nav
              aria-label="Primary"
              className="glass-pill hidden items-center gap-1 rounded-full p-1 sm:flex"
            >
              <button
                type="button"
                onClick={goToQuestion}
                className="rounded-full bg-white/30 px-4 py-1.5 text-xs font-semibold text-white sm:px-5"
              >
                Search
              </button>
              <button type="button" onClick={() => setHistoryOpen(true)} className={navItem}>
                History{history.length > 0 ? ` (${history.length})` : ""}
              </button>
              <button type="button" onClick={goToSafety} className={navItem}>
                Safety
              </button>
            </nav>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setHistoryOpen(true)}
                className="glass-pill rounded-full px-4 py-2 text-xs font-medium text-white sm:hidden"
              >
                History{history.length > 0 ? ` (${history.length})` : ""}
              </button>
              <span className="glass-pill hidden rounded-full px-4 py-2 text-xs font-medium text-white md:inline-block">
                Emergency? Call 911 or 988
              </span>
            </div>
          </header>

          <PulseLine />

          <section className={compact ? "pt-8 sm:pt-10" : "pt-10 sm:pt-14"}>
            {compact ? (
              <h1 className="font-display text-[clamp(1.7rem,4.2vw,2.75rem)] font-light leading-[1.05] tracking-[-0.03em] text-white">
                Medical answers, grounded in research.
              </h1>
            ) : (
              <div className="grid gap-8 md:grid-cols-[1.6fr_1fr] md:items-end">
                <h1 className="font-display text-[clamp(2.6rem,9vw,5.75rem)] font-light leading-[0.95] tracking-[-0.035em] text-white">
                  <span className="block">Medical answers</span>
                  <span className="block sm:ml-[12%]">grounded in</span>
                  <span className="block sm:ml-[3%]">research.</span>
                </h1>
                <div>
                  <HeroArt />
                  <p className="border-white/40 text-base leading-relaxed text-white/90 md:mt-4 md:border-l md:pl-6">
                    Ask a health question and get an answer written from PubMed abstracts, with every
                    claim tied to the paper it came from.
                  </p>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="search-wrapper mt-8 sm:mt-10">
              <label htmlFor="question" className="sr-only">
                Medical question
              </label>
              <div className="search-field glass flex flex-col gap-2 rounded-[1.75rem] p-2 sm:flex-row sm:items-center sm:rounded-full">
                <input
                  id="question"
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a medical question…"
                  className="min-h-[3.25rem] flex-1 rounded-full bg-transparent px-5 py-3 text-[1.05rem] text-white outline-none placeholder:text-white/70 sm:px-6"
                  disabled={loading}
                  autoComplete="off"
                />
                {loading ? (
                  <button
                    type="button"
                    onClick={stop}
                    className="glass-pill font-display min-h-[3.25rem] shrink-0 rounded-full px-8 text-sm font-semibold text-white transition-colors hover:bg-white/25 sm:px-10"
                  >
                    <span className="flex items-center justify-center gap-2">
                      <span
                        className="inline-block h-4 w-4 rounded-full border-2 border-white/40 border-t-white"
                        style={{ animation: "spin-slow 0.8s linear infinite" }}
                      />
                      Stop
                    </span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!question.trim()}
                    className="btn-mint font-display min-h-[3.25rem] shrink-0 rounded-full px-8 text-sm font-semibold sm:px-10"
                  >
                    Ask
                  </button>
                )}
              </div>

              <label className="mt-4 flex cursor-pointer items-start gap-2.5 px-2 text-xs text-white/85 sm:px-5">
                <input
                  type="checkbox"
                  checked={decompose}
                  onChange={(e) => setDecompose(e.target.checked)}
                  disabled={loading}
                  className="mt-0.5 h-3.5 w-3.5 accent-emerald-300"
                />
                <span>
                  Complex question mode
                  <span className="text-white/70"> (splits it into parts before searching)</span>
                </span>
              </label>
            </form>

            {showExamples && (
              <div className="mt-8">
                <p className="mb-3 px-2 text-xs font-semibold uppercase tracking-[0.15em] text-white/80 sm:px-5">
                  Try asking
                </p>
                <div className="flex flex-wrap gap-2.5">
                  {EXAMPLE_QUESTIONS.map((example) => (
                    <button
                      key={example}
                      type="button"
                      onClick={() => void runQuery(example)}
                      className="glass-pill rounded-full px-4 py-2 text-left text-sm text-white transition-colors hover:bg-white/28"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </section>

          <main>
            {showSkeleton && <LoadingState />}

            {error && !loading && (
              <div className="glass-strong mt-10 rounded-3xl border-red-200/50 px-6 py-5">
                <p className="font-display text-sm font-semibold text-red-100">Something went wrong</p>
                <p className="mt-2 text-sm leading-relaxed text-red-50/90">{error}</p>
              </div>
            )}

            {flagged && !loading && (
              <div className="glass-strong mt-10 rounded-3xl border-amber-200/50 px-6 py-6">
                <div className="mb-2 flex items-center gap-2">
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-amber-200/25 text-amber-100">
                    !
                  </span>
                  <h2 className="font-display text-lg font-semibold text-amber-50">
                    Unable to process this question
                  </h2>
                </div>
                <p className="text-sm leading-relaxed text-amber-50/95">{warning}</p>
              </div>
            )}

            {hasResults && (
              <section className="mt-10 space-y-8">
                <article className="glass-strong rounded-3xl p-6 sm:p-9">
                  <div className="mb-6 flex items-center gap-3">
                    <span className="font-display text-xs font-semibold uppercase tracking-[0.2em] text-emerald-200">
                      Answer
                    </span>
                    <div className="h-px flex-1 bg-white/25" />
                  </div>
                  <div className="answer-text whitespace-pre-wrap">
                    <AnswerText text={answer ?? ""} sourceCount={sources.length} onCite={citeClicked} />
                    {loading && (
                      <span
                        className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-emerald-200"
                        style={{ animation: "pulse-glow 1s ease-in-out infinite" }}
                        aria-hidden
                      />
                    )}
                  </div>

                  {!loading && grounding && (
                    <GroundingNotice grounding={grounding} total={sources.length} />
                  )}

                  {!loading && answer && (
                    <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-white/20 pt-5">
                      {queryId && (
                        <>
                          <span className="text-xs text-white/80">Was this answer helpful?</span>
                          <button
                            type="button"
                            onClick={() => void submitFeedback("up")}
                            disabled={feedback !== null}
                            aria-pressed={feedback === "up"}
                            aria-label="This answer was helpful"
                            className={`rounded-full border p-2 transition-colors ${
                              feedback === "up"
                                ? "border-emerald-200 bg-emerald-200/25 text-emerald-50"
                                : "border-white/30 bg-white/10 text-white/85 hover:bg-white/20"
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
                            className={`rounded-full border p-2 transition-colors ${
                              feedback === "down"
                                ? "border-red-200 bg-red-300/25 text-red-50"
                                : "border-white/30 bg-white/10 text-white/85 hover:bg-white/20"
                            } disabled:cursor-default`}
                          >
                            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z" />
                              <path strokeLinecap="round" strokeLinejoin="round" d="M17 2v11" />
                            </svg>
                          </button>
                          {feedback && <span className="text-xs text-white/80">Thanks for the feedback.</span>}
                        </>
                      )}
                      <button
                        type="button"
                        onClick={() => void copyAnswer()}
                        className="glass-pill ml-auto rounded-full px-4 py-1.5 text-xs text-white transition-colors hover:bg-white/28"
                      >
                        {copied ? "Copied" : "Copy answer"}
                      </button>
                    </div>
                  )}
                </article>

                {sources.length > 0 && (
                  <div>
                    <div className="mb-4 flex items-center justify-between px-1">
                      <h2 className="font-display text-sm font-semibold uppercase tracking-[0.15em] text-white">
                        Cited sources
                      </h2>
                      <span className="glass-pill rounded-full px-3 py-1 text-xs text-white">
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

          {showExamples && (
            <section aria-label="How HealthLens works" className="mt-10 grid gap-4 sm:grid-cols-3">
              {FEATURES.map((feature) => (
                <div key={feature.title} className="glass rounded-3xl p-6">
                  {feature.icon}
                  <h2 className="font-display mt-5 text-lg font-medium tracking-tight text-white">
                    {feature.title}
                  </h2>
                  <p className="mt-2 text-sm leading-relaxed text-white/85">{feature.text}</p>
                </div>
              ))}
            </section>
          )}

          <footer id="safety" className="mt-10 border-t border-white/25 pt-6 text-xs leading-relaxed text-white/80">
            <p>
              <span className="font-semibold text-white">Not medical advice.</span> HealthLens is for
              education and research only and does not provide diagnosis or treatment. For emergencies
              call 911. If you are in crisis, call or text 988.
            </p>
            <p className="mt-2 text-white/65">Powered by PubMed · Hybrid retrieval · Groq LLM</p>
          </footer>
        </div>
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
