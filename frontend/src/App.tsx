import { useMemo, useState, type FormEvent } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
// Baked into the public JS bundle at build time — deters casual abuse only,
// not a real secret. See frontend/.env.example for the full caveat.
const API_KEY = import.meta.env.VITE_HEALTHLENS_API_KEY ?? "";

interface Source {
  title: string;
  pmid: string;
}

interface QueryResponse {
  answer: string | null;
  sources: Source[];
  flagged: boolean;
  warning: string | null;
  detail?: string;
}

function Background() {
  const particles = useMemo(
    () =>
      Array.from({ length: 28 }, (_, i) => ({
        id: i,
        left: `${(i * 17 + 7) % 100}%`,
        size: 1 + (i % 3),
        delay: `${(i * 0.7) % 12}s`,
        duration: `${14 + (i % 8)}s`,
      })),
    []
  );

  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden>
      <div className="absolute inset-0 bg-[#060d1a]" />
      <div
        className="absolute inset-0 opacity-[0.35]"
        style={{
          backgroundImage:
            "radial-gradient(circle at 1px 1px, rgba(148,163,184,0.12) 1px, transparent 0)",
          backgroundSize: "40px 40px",
        }}
      />
      <div className="orb-1 absolute -left-32 top-[-10%] h-[520px] w-[520px] rounded-full bg-teal-500/20 blur-[120px]" />
      <div className="orb-2 absolute -right-24 top-[20%] h-[480px] w-[480px] rounded-full bg-indigo-600/25 blur-[130px]" />
      <div className="orb-3 absolute bottom-[-15%] left-[30%] h-[560px] w-[560px] rounded-full bg-sky-600/15 blur-[140px]" />
      {particles.map((p) => (
        <span
          key={p.id}
          className="particle absolute bottom-0 rounded-full bg-sky-300/30"
          style={{
            left: p.left,
            width: p.size,
            height: p.size,
            animationDelay: p.delay,
            animationDuration: p.duration,
          }}
        />
      ))}
    </div>
  );
}

function LoadingState() {
  return (
    <div className="mt-16 space-y-8">
      <div className="flex flex-col items-center gap-5">
        <div className="relative h-14 w-14">
          <div
            className="absolute inset-0 rounded-full border-2 border-white/10"
            style={{ animation: "spin-slow 1.2s linear infinite" }}
          />
          <div
            className="absolute inset-0 rounded-full border-2 border-transparent border-t-sky-400 border-r-indigo-400"
            style={{ animation: "spin-slow 0.9s linear infinite reverse" }}
          />
          <div className="absolute inset-[18px] rounded-full bg-sky-400/20" style={{ animation: "pulse-glow 2s ease-in-out infinite" }} />
        </div>
        <div className="text-center">
          <p className="font-display text-sm font-medium text-white/80">
            Searching PubMed literature
          </p>
          <p className="mt-1 text-xs text-white/40">
            Retrieving abstracts · Ranking sources · Generating answer
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 backdrop-blur-sm">
          <div className="mb-5 h-3 w-24 rounded-full skeleton-shimmer" />
          <div className="space-y-3">
            <div className="h-3 w-full rounded-full skeleton-shimmer" />
            <div className="h-3 w-[92%] rounded-full skeleton-shimmer" />
            <div className="h-3 w-[85%] rounded-full skeleton-shimmer" />
            <div className="h-3 w-[70%] rounded-full skeleton-shimmer" />
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {[1, 2].map((n) => (
            <div
              key={n}
              className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-5"
            >
              <div className="mb-3 h-2.5 w-16 rounded-full skeleton-shimmer" />
              <div className="h-3 w-full rounded-full skeleton-shimmer" />
              <div className="mt-2 h-3 w-3/4 rounded-full skeleton-shimmer" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function LogoMark() {
  return (
    <div className="relative flex h-14 w-14 items-center justify-center">
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-sky-400/30 via-indigo-500/20 to-violet-500/30 blur-sm" />
      <div className="relative flex h-14 w-14 items-center justify-center rounded-2xl border border-white/15 bg-gradient-to-br from-[#0f2744] to-[#0a1628] shadow-lg shadow-black/30">
        <span className="font-display bg-gradient-to-br from-sky-300 via-white to-indigo-300 bg-clip-text text-xl font-bold tracking-tight text-transparent">
          HL
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [flagged, setFlagged] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;

    setLoading(true);
    setAnswer(null);
    setSources([]);
    setFlagged(false);
    setWarning(null);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/api/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
        body: JSON.stringify({ question: q }),
      });

      const data: QueryResponse = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Request failed");
      }

      if (data.flagged) {
        setFlagged(true);
        setWarning(data.warning);
      } else {
        setAnswer(data.answer);
        setSources(data.sources || []);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong. Is the backend running?";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const hasResults = answer && !loading && !flagged;

  return (
    <div className="relative min-h-screen text-white">
      <Background />

      <div className="relative mx-auto flex min-h-screen max-w-3xl flex-col px-6 pb-20 pt-12 sm:px-8 sm:pt-16">
        <header className="mb-14 text-center sm:mb-16">
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
                <button
                  type="submit"
                  disabled={loading || !question.trim()}
                  className="btn-gradient font-display min-h-[3.25rem] shrink-0 rounded-xl px-8 py-3.5 text-sm font-semibold tracking-wide text-white sm:min-h-[3.5rem] sm:px-10"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <span
                        className="inline-block h-4 w-4 rounded-full border-2 border-white/30 border-t-white"
                        style={{ animation: "spin-slow 0.8s linear infinite" }}
                      />
                      Searching
                    </span>
                  ) : (
                    "Ask"
                  )}
                </button>
              </div>
            </div>
          </form>

          <p className="mt-5 text-center text-xs leading-relaxed text-slate-500">
            Not medical advice. For emergencies, call{" "}
            <span className="text-slate-400">911</span> or{" "}
            <span className="text-slate-400">988</span>.
          </p>

          {loading && <LoadingState />}

          {error && !loading && (
            <div className="mt-14 rounded-2xl border border-red-400/20 bg-red-950/30 px-6 py-5 backdrop-blur-sm">
              <p className="font-display text-sm font-semibold text-red-300">
                Something went wrong
              </p>
              <p className="mt-2 text-sm leading-relaxed text-red-200/80">
                {error}
              </p>
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
              <p className="text-sm leading-relaxed text-amber-100/85">
                {warning}
              </p>
            </div>
          )}

          {hasResults && (
            <section className="mt-14 space-y-10">
              <article className="rounded-2xl border border-white/[0.09] bg-white/[0.03] p-8 shadow-xl shadow-black/20 backdrop-blur-md sm:p-10">
                <div className="mb-6 flex items-center gap-3">
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/15 to-transparent" />
                  <span className="font-display text-xs font-semibold uppercase tracking-[0.2em] text-sky-400/80">
                    Answer
                  </span>
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/15 to-transparent" />
                </div>
                <div className="answer-text whitespace-pre-wrap">{answer}</div>
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
                      <li
                        key={src.pmid}
                        className="citation-card rounded-xl border border-white/[0.08] bg-white/[0.025] p-5 sm:p-6"
                      >
                        <div className="flex gap-4">
                          <span className="font-display flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-sky-500/20 to-indigo-500/20 text-xs font-bold text-sky-300">
                            {i + 1}
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="text-[0.95rem] font-medium leading-snug text-slate-100">
                              {src.title}
                            </p>
                            <a
                              href={`https://pubmed.ncbi.nlm.nih.gov/${src.pmid}/`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="mt-3 inline-flex items-center gap-1.5 text-sm text-sky-400 transition-colors hover:text-sky-300"
                            >
                              <span className="text-slate-500">PMID</span>
                              <span className="font-mono">{src.pmid}</span>
                              <svg
                                className="h-3.5 w-3.5 opacity-60"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                                strokeWidth={2}
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
    </div>
  );
}
