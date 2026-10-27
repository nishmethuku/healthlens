import { afterEach, describe, expect, it, vi } from "vitest";
import { streamQuery, type StreamHandlers } from "./api";

const encoder = new TextEncoder();

function sseResponse(chunks: string[], status = 200) {
  const body = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status });
}

function recorder() {
  const log: string[] = [];
  const handlers: StreamHandlers = {
    onSources: (s, id) => log.push(`sources:${s.length}:${id}`),
    onToken: (t) => log.push(`token:${t}`),
    onGrounding: (g) => log.push(`grounding:${g.grounded}`),
    onFlagged: (w) => log.push(`flagged:${w}`),
    onError: (d) => log.push(`error:${d}`),
    onDone: () => log.push("done"),
  };
  return { log, handlers };
}

afterEach(() => vi.unstubAllGlobals());

describe("streamQuery", () => {
  it("parses events even when they are split across network chunks", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([
          'event: sources\ndata: {"sources": [{"title": "T", "pmid": "1"}], "query_id": "q1"}\n\nevent: tok',
          'en\ndata: {"text": "Hi "}\n\nevent: token\ndata: {"text": "there"}\n\n',
          'event: grounding\ndata: {"cited": [1], "invalid": [], "claim_sentences": 1, "cited_sentences": 1, "coverage": 1, "grounded": true}\n\nevent: done\ndata: {}\n\n',
        ]),
      ),
    );
    const { log, handlers } = recorder();
    await streamQuery("q", false, new AbortController().signal, handlers);
    expect(log).toEqual(["sources:1:q1", "token:Hi ", "token:there", "grounding:true", "done"]);
  });

  it("skips a malformed event and keeps reading", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse(["event: token\ndata: {broken\n\n", 'event: token\ndata: {"text": "ok"}\n\n']),
      ),
    );
    const { log, handlers } = recorder();
    await streamQuery("q", false, new AbortController().signal, handlers);
    expect(log).toEqual(["token:ok"]);
  });

  it("sends the question and the decompose flag with the API key header", async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse([]));
    vi.stubGlobal("fetch", fetchMock);
    await streamQuery("What is X?", true, new AbortController().signal, recorder().handlers);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/api\/v1\/query\/stream$/);
    expect(JSON.parse(init.body)).toEqual({ question: "What is X?", decompose: true });
    expect(init.headers).toHaveProperty("X-API-Key");
  });

  it("throws the server's detail message on an error response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Missing or invalid API key" }), { status: 401 }),
      ),
    );
    await expect(
      streamQuery("q", false, new AbortController().signal, recorder().handlers),
    ).rejects.toThrow("Missing or invalid API key");
  });
});
