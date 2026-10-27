import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const encoder = new TextEncoder();

function streamOf(events: string[]) {
  const body = new ReadableStream({
    start(controller) {
      for (const e of events) controller.enqueue(encoder.encode(e));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

const ANSWER_EVENTS = [
  'event: sources\ndata: {"sources": [{"title": "Statins and LDL", "pmid": "111", "snippet": "A randomized trial."}], "query_id": "q1"}\n\n',
  'event: token\ndata: {"text": "Statins lower LDL cholesterol in adults [1]."}\n\n',
  'event: grounding\ndata: {"cited": [1], "invalid": [], "claim_sentences": 1, "cited_sentences": 1, "coverage": 1, "grounded": true}\n\n',
  "event: done\ndata: {}\n\n",
];

beforeEach(() => {
  localStorage.clear();
  window.matchMedia = vi.fn().mockReturnValue({ matches: true }) as unknown as typeof window.matchMedia;
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("shows the glass landing page with nav, examples, and feature cards", () => {
    render(<App />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toContain("Medical answers");
    expect(screen.getByRole("navigation", { name: "Primary" })).toBeTruthy();
    expect(screen.getByText("Try asking")).toBeTruthy();
    expect(screen.getByText("Grounded in PubMed")).toBeTruthy();
    expect(screen.getByText("Every claim cited")).toBeTruthy();
    expect(screen.getByText("Safety first")).toBeTruthy();
    expect(document.querySelector(".glass-frame")).not.toBeNull();
    expect(document.querySelector(".ecg-glow")).not.toBeNull();
  });

  it("answers a question, links the citation to its source, and saves history", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => streamOf(ANSWER_EVENTS)));
    render(<App />);

    await userEvent.type(screen.getByLabelText("Medical question"), "Do statins work?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(screen.getByText(/Statins lower LDL cholesterol/)).toBeTruthy());
    expect(screen.getByText("Statins and LDL")).toBeTruthy();
    expect(screen.getByText("A randomized trial.")).toBeTruthy();

    // Once there is an answer the big hero collapses to the compact headline.
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
      "Medical answers, grounded in research.",
    );
    expect(screen.queryByText("Try asking")).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Go to source 1" }));
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();

    const saved = JSON.parse(localStorage.getItem("healthlens.history.v1") ?? "[]");
    expect(saved).toHaveLength(1);
    expect(saved[0].question).toBe("Do statins work?");
  });

  it("shows the server's error message when the request fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "PubMed could not be reached. Please try again shortly." }), {
          status: 503,
        }),
      ),
    );
    render(<App />);
    await userEvent.type(screen.getByLabelText("Medical question"), "Anything");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(screen.getByText(/PubMed could not be reached/)).toBeTruthy());
    expect(localStorage.getItem("healthlens.history.v1")).toBeNull();
  });
});
