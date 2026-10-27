import { useMemo } from "react";

export function Background() {
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

export function LoadingState() {
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

export function LogoMark() {
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
