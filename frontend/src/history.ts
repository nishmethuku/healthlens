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
