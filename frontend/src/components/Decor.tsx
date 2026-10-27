// Soft bokeh discs, positioned by hand so the scene is identical on every load.
const BOKEH = [
  { left: "8%", top: "14%", size: 90, opacity: 0.16 },
  { left: "22%", top: "62%", size: 140, opacity: 0.12 },
  { left: "41%", top: "8%", size: 60, opacity: 0.18 },
  { left: "58%", top: "74%", size: 110, opacity: 0.14 },
  { left: "74%", top: "20%", size: 150, opacity: 0.12 },
  { left: "88%", top: "56%", size: 80, opacity: 0.18 },
  { left: "12%", top: "86%", size: 70, opacity: 0.15 },
  { left: "93%", top: "88%", size: 120, opacity: 0.12 },
];

// Morning-ocean scene: sky blue into teal, with soft light and blurred color
// behind the glass panels. Pure CSS, no image files.
export function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden" aria-hidden>
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(165deg, #2b86b8 0%, #2a93b4 22%, #1c8a98 48%, #226f9d 74%, #153f6d 100%)",
        }}
      />

      {/* Diagonal shaft of light */}
      <div className="absolute -top-48 left-[34%] h-[150%] w-[26%] rotate-[18deg] bg-gradient-to-b from-white/22 via-white/8 to-transparent blur-3xl" />

      {/* Blurred color pools: mint, warm gold, violet */}
      <div className="orb-1 absolute -left-32 top-[-8%] h-[540px] w-[540px] rounded-full bg-emerald-300/30 blur-[120px]" />
      <div className="orb-2 absolute -right-28 top-[18%] h-[500px] w-[500px] rounded-full bg-indigo-400/28 blur-[130px]" />
      <div className="orb-3 absolute bottom-[-14%] left-[28%] h-[560px] w-[560px] rounded-full bg-amber-200/25 blur-[140px]" />

      {BOKEH.map((b, i) => (
        <span
          key={i}
          className="absolute rounded-full bg-white blur-[3px]"
          style={{ left: b.left, top: b.top, width: b.size, height: b.size, opacity: b.opacity }}
        />
      ))}

      {/* Glass capsules and cells floating around the frame */}
      <div
        className="glass-capsule drift absolute left-[1%] top-[12%] hidden h-[250px] w-[100px] lg:block"
        style={{ "--tilt": "-24deg" } as React.CSSProperties}
      />
      <div
        className="glass-capsule drift-slow absolute bottom-[10%] right-[1.5%] hidden h-[200px] w-[80px] lg:block"
        style={{ "--tilt": "28deg" } as React.CSSProperties}
      />
      <div className="glass-sphere drift-slow absolute right-[3%] top-[16%] hidden h-24 w-24 lg:block" />
      <div className="glass-sphere drift absolute bottom-[22%] left-[2%] hidden h-16 w-16 lg:block" />
      <div className="glass-sphere drift absolute left-[9%] top-[64%] hidden h-9 w-9 lg:block" />

      {/* Darkens the edges so the frame floats */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 40%, transparent 55%, rgba(4, 24, 48, 0.38) 100%)",
        }}
      />
    </div>
  );
}

export function LoadingState() {
  return (
    <div className="mt-14 space-y-8">
      <div className="flex flex-col items-center gap-5">
        <div className="relative h-14 w-14">
          <div
            className="absolute inset-0 rounded-full border-2 border-white/20"
            style={{ animation: "spin-slow 1.2s linear infinite" }}
          />
          <div
            className="absolute inset-0 rounded-full border-2 border-transparent border-t-white border-r-emerald-200"
            style={{ animation: "spin-slow 0.9s linear infinite reverse" }}
          />
          <div
            className="absolute inset-[18px] rounded-full bg-white/30"
            style={{ animation: "pulse-glow 2s ease-in-out infinite" }}
          />
        </div>
        <div className="text-center">
          <p className="font-display text-sm font-medium text-white">Searching PubMed literature</p>
          <p className="mt-1 text-xs text-white/70">
            Retrieving abstracts · Ranking sources · Generating answer
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <div className="glass-strong rounded-3xl p-8">
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
            <div key={n} className="glass-strong rounded-2xl p-5">
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
    <div className="flex items-center gap-3">
      <div className="glass-pill flex h-9 w-9 items-center justify-center rounded-xl">
        <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8} aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" d="M2 12h4.5l2-5.5 3.5 11 2.5-8 1.5 2.5H22" />
        </svg>
      </div>
      <span className="font-display text-lg font-medium tracking-tight text-white">HealthLens</span>
    </div>
  );
}

// Heartbeat line used as the divider under the header. The faint line is always
// visible; a mint segment travels along it.
export function PulseLine() {
  const d = "M0 14 H340 q12 -8 24 0 H392 l8 3 l8 -15 l8 24 l8 -12 H470 q16 -9 32 0 H1000";
  return (
    <svg
      className="mt-6 block h-7 w-full sm:mt-8"
      viewBox="0 0 1000 28"
      preserveAspectRatio="none"
      aria-hidden
    >
      <path d={d} fill="none" stroke="rgba(255,255,255,0.40)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      <path
        className="ecg-glow"
        d={d}
        fill="none"
        stroke="#8df0b0"
        strokeWidth={2.5}
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
        style={{ filter: "drop-shadow(0 0 6px rgba(141,240,176,0.9))" }}
      />
    </svg>
  );
}

// Crisp glass capsule and cells shown beside the hero text.
export function HeroArt() {
  return (
    <div className="relative mx-auto hidden h-44 w-full max-w-[14rem] md:block" aria-hidden>
      <div
        className="glass-capsule drift absolute left-[22%] top-0 h-40 w-16"
        style={{ "--tilt": "24deg" } as React.CSSProperties}
      />
      <div className="glass-sphere drift-slow absolute right-[12%] top-[8%] h-16 w-16" />
      <div className="glass-sphere drift absolute bottom-0 right-[34%] h-9 w-9" />
      <div className="glass-sphere drift-slow absolute bottom-[10%] left-[4%] h-6 w-6" />
    </div>
  );
}

// Thin-line icons for the feature cards.
export function IconBook() {
  return (
    <svg className="h-9 w-9 text-white/85" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.1} aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.5C10.5 5.2 8 4.5 4.5 4.5v13c3.5 0 6 .7 7.5 2m0-12.5c1.5-1.3 4-2 7.5-2v13c-3.5 0-6 .7-7.5 2m0-13v13" />
    </svg>
  );
}

export function IconQuote() {
  return (
    <svg className="h-9 w-9 text-white/85" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.1} aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 6h14a1 1 0 011 1v8a1 1 0 01-1 1h-7l-4 3.5V16H5a1 1 0 01-1-1V7a1 1 0 011-1z" />
      <path strokeLinecap="round" d="M8 10h8M8 13h5" />
    </svg>
  );
}

export function IconShield() {
  return (
    <svg className="h-9 w-9 text-white/85" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.1} aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3l7.5 2.8v5.7c0 4.6-3.1 8.2-7.5 9.5-4.4-1.3-7.5-4.9-7.5-9.5V5.8L12 3z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2.2 2.2L15.5 10" />
    </svg>
  );
}
