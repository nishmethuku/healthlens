import type { Grounding, Source } from "./api";

export interface HistoryEntry {
  id: string;
  question: string;
  answer: string;
  sources: Source[];
  grounding: Grounding | null;
  createdAt: number;
}

const STORAGE_KEY = "healthlens.history.v1";
const MAX_ENTRIES = 30;

// History lives only in this browser's localStorage and is never sent anywhere.
// Every access is wrapped because storage can be blocked or full (private
// windows, strict browser settings), and history is a convenience, not a requirement.
export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

function persist(entries: HistoryEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // storage unavailable or full — keep working without persistence
  }
}

export function addEntry(entries: HistoryEntry[], entry: HistoryEntry): HistoryEntry[] {
  const next = [entry, ...entries.filter((e) => e.question !== entry.question)].slice(0, MAX_ENTRIES);
  persist(next);
  return next;
}

export function removeEntry(entries: HistoryEntry[], id: string): HistoryEntry[] {
  const next = entries.filter((e) => e.id !== id);
  persist(next);
  return next;
}

export function clearHistory(): HistoryEntry[] {
  persist([]);
  return [];
}

function isEntry(value: unknown): value is HistoryEntry {
  if (typeof value !== "object" || value === null) return false;
  const e = value as Record<string, unknown>;
  return (
    typeof e.id === "string" &&
    typeof e.question === "string" &&
    typeof e.answer === "string" &&
    typeof e.createdAt === "number" &&
    Array.isArray(e.sources) &&
    e.sources.every(
      (s) =>
        typeof s === "object" &&
        s !== null &&
        typeof (s as Source).title === "string" &&
        typeof (s as Source).pmid === "string",
    )
  );
}

export function exportHistory(entries: HistoryEntry[]): string {
  return JSON.stringify({ app: "healthlens", version: 1, entries }, null, 2);
}

/** Read a file produced by exportHistory. Returns null when the file is not usable. */
export function parseImport(text: string): HistoryEntry[] | null {
  try {
    const data = JSON.parse(text);
    const list = Array.isArray(data) ? data : data?.entries;
    if (!Array.isArray(list)) return null;
    const valid = list.filter(isEntry).map((e) => ({ ...e, grounding: e.grounding ?? null }));
    return valid.length > 0 ? valid : null;
  } catch {
    return null;
  }
}

export function mergeImported(current: HistoryEntry[], imported: HistoryEntry[]): HistoryEntry[] {
  const seen = new Set(current.map((e) => e.question));
  const extra = imported.filter((e) => !seen.has(e.question));
  const next = [...current, ...extra].sort((a, b) => b.createdAt - a.createdAt).slice(0, MAX_ENTRIES);
  persist(next);
  return next;
}

export function answerWithSources(answer: string, sources: Source[]): string {
  if (sources.length === 0) return answer;
  const lines = sources.map(
    (s, i) => `[${i + 1}] ${s.title} https://pubmed.ncbi.nlm.nih.gov/${s.pmid}/`,
  );
  return `${answer}\n\nSources:\n${lines.join("\n")}`;
}
