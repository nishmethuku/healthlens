const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
// Baked into the public JS bundle at build time — deters casual abuse only,
// not a real secret. See frontend/.env.example for the full caveat.
const API_KEY = import.meta.env.VITE_HEALTHLENS_API_KEY ?? "";

export interface Source {
  title: string;
  pmid: string;
  snippet?: string;
}

export interface Grounding {
  cited: number[];
  invalid: number[];
  claim_sentences: number;
  cited_sentences: number;
  coverage: number;
  grounded: boolean;
}

export interface StreamHandlers {
  onSources: (sources: Source[], queryId: string) => void;
  onToken: (text: string) => void;
  onGrounding: (grounding: Grounding) => void;
  onFlagged: (warning: string | null) => void;
  onError: (detail: string) => void;
  onDone: () => void;
}

// Native EventSource can't do POST + custom headers, so SSE is parsed by hand
// from a fetch() ReadableStream. Events are "event: X\ndata: Y\n\n" blocks;
// data chunks can split across reads, so unparsed text is buffered between
// reader.read() calls until a full "\n\n"-terminated event is available.
async function consumeSSE(res: Response, handlers: StreamHandlers) {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("Streaming is not supported in this browser.");
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sepIndex: number;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);

      let eventType = "message";
      let data = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event: ")) eventType = line.slice(7);
        else if (line.startsWith("data: ")) data = line.slice(6);
      }
      if (!data) continue;

      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue; // ignore a malformed event instead of aborting the whole answer
      }

      switch (eventType) {
        case "sources":
          handlers.onSources(parsed.sources ?? [], parsed.query_id ?? "");
          break;
        case "token":
          handlers.onToken(parsed.text ?? "");
          break;
        case "grounding":
          handlers.onGrounding(parsed as Grounding);
          break;
        case "flagged":
          handlers.onFlagged(parsed.warning ?? null);
          break;
        case "error":
          handlers.onError(parsed.detail ?? "Something went wrong.");
          break;
        case "done":
          handlers.onDone();
          break;
      }
    }
  }
}

export async function streamQuery(
  question: string,
  decompose: boolean,
  signal: AbortSignal,
  handlers: StreamHandlers,
) {
  const res = await fetch(`${API_URL}/api/v1/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
    body: JSON.stringify({ question, decompose }),
    signal,
  });

  if (!res.ok) {
    let detail = "Request failed";
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      // response body wasn't JSON (e.g. a 401 from a proxy) — keep the default
    }
    throw new Error(detail);
  }

  await consumeSSE(res, handlers);
}

export async function sendFeedback(queryId: string, rating: "up" | "down", answerPreview: string) {
  try {
    await fetch(`${API_URL}/api/v1/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
      body: JSON.stringify({ query_id: queryId, rating, answer_preview: answerPreview }),
    });
  } catch {
    // best-effort — don't surface a feedback-submission failure to the user
  }
}
