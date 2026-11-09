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

import { answerWithSources, exportHistory, mergeImported, parseImport } from "./history";

describe("export, import and copy", () => {
  const entry = {
    id: "1",
    question: "q one",
    answer: "a [1]",
    sources: [{ title: "Paper", pmid: "123" }],
    grounding: null,
    createdAt: 5,
  };

  it("round-trips an export", () => {
    expect(parseImport(exportHistory([entry]))).toEqual([entry]);
  });

  it("rejects files that are not history", () => {
    expect(parseImport("not json")).toBeNull();
    expect(parseImport(JSON.stringify({ entries: [{ foo: 1 }] }))).toBeNull();
  });

  it("merges without duplicating questions", () => {
    const other = { ...entry, id: "2", question: "q two", createdAt: 9 };
    const merged = mergeImported([entry], [entry, other]);
    expect(merged.map((e) => e.question)).toEqual(["q two", "q one"]);
  });

  it("adds PubMed links when copying", () => {
    const text = answerWithSources("Answer [1]", entry.sources);
    expect(text).toContain("https://pubmed.ncbi.nlm.nih.gov/123/");
  });
});
