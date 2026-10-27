import { beforeEach, describe, expect, it } from "vitest";
import { addEntry, clearHistory, loadHistory, removeEntry, type HistoryEntry } from "./history";

function entry(id: string, question = `question ${id}`): HistoryEntry {
  return { id, question, answer: "a", sources: [], grounding: null, createdAt: 0 };
}

beforeEach(() => localStorage.clear());

describe("history", () => {
  it("saves newest first and reloads from storage", () => {
    let entries = addEntry([], entry("1"));
    entries = addEntry(entries, entry("2"));
    expect(entries.map((e) => e.id)).toEqual(["2", "1"]);
    expect(loadHistory().map((e) => e.id)).toEqual(["2", "1"]);
  });

  it("replaces an earlier entry for the same question instead of duplicating it", () => {
    let entries = addEntry([], entry("1", "same"));
    entries = addEntry(entries, entry("2", "same"));
    expect(entries.map((e) => e.id)).toEqual(["2"]);
  });

  it("keeps at most 30 entries", () => {
    let entries: HistoryEntry[] = [];
    for (let i = 0; i < 35; i++) entries = addEntry(entries, entry(String(i)));
    expect(entries).toHaveLength(30);
    expect(entries[0].id).toBe("34");
  });

  it("removes one entry and clears everything", () => {
    const entries = addEntry(addEntry([], entry("1")), entry("2"));
    expect(removeEntry(entries, "1").map((e) => e.id)).toEqual(["2"]);
    expect(clearHistory()).toEqual([]);
    expect(loadHistory()).toEqual([]);
  });

  it("treats corrupt storage as empty history", () => {
    localStorage.setItem("healthlens.history.v1", "{not json");
    expect(loadHistory()).toEqual([]);
  });
});
