# Security

This document describes what HealthLens protects, how it is defended, how each defense is
tested, and, just as important, what is not solved. To report a vulnerability, open a
private security advisory on this repository (Security tab, "Report a vulnerability")
instead of a public issue.

## What needs protecting

| Asset | Why it matters |
|---|---|
| Groq API key | Anyone holding it can spend the owner's quota, so a leak becomes a bill |
| HealthLens API key | Gates the endpoints that call the language model |
| Users' questions | Health questions can reveal medical conditions, so they are sensitive even without a name |
| Answer integrity | A wrong or manipulated answer about health can cause harm |
| Availability and cost | Each answer costs tokens, so abuse becomes a bill |

## Trust boundaries

```
Browser (untrusted)  --HTTPS-->  API (HealthLens)  --HTTPS-->  Groq (third party)
                                       |
                                       +--HTTPS-->  PubMed (third party, content is untrusted)
```

Everything crossing a boundary is treated as untrusted: the request body and headers, the
PubMed abstracts (which can contain text written to manipulate a model), and the model's
output (which can cite papers that do not exist).

## Threats and defenses

Status: **Done** means implemented and covered by a test. **Partial** means reduced but not
eliminated. **Open** means known and not addressed.

| Threat | Defense | Status |
|---|---|---|
| Guessing the API key by timing | Keys are compared with `hmac.compare_digest`, so response time does not reveal a correct prefix | Done |
| Missing or wrong key | Fail closed: with no key configured, every request is rejected with 401, and rejections are logged without the key | Done |
| Unsafe production configuration | The app refuses to start in production if the API key is unset, or if allowed origins are empty, a wildcard, or not https | Done |
| Cross-site requests from other origins | CORS allows only the configured origins and a short header list; credentialed requests are disabled because no cookies are used | Done |
| Browser-side attacks on the API's responses | Every response carries `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, a `default-src 'none'` policy, `Cache-Control: no-store`, and HSTS in production | Done |
| Script injection into the web app | The production page carries a Content-Security-Policy that allows scripts only from its own origin and connections only to its own origin and the API | Partial (set by a meta tag, which cannot express `frame-ancestors`) |
| Log forging and header abuse through the request ID | A caller's `X-Request-ID` is kept only if it is 1 to 64 plain characters; otherwise a new one is generated | Done |
| Information disclosure through `/metrics` | In production the endpoint needs a separate bearer token and returns 404 without it; it is closed if no token is configured | Done |
| Information disclosure through API docs | Interactive docs and the OpenAPI schema are disabled in production | Done |
| Cost abuse and denial of service | Per-client rate limit on query, stream, and feedback endpoints; question length capped at 2000 characters; feedback fields bounded and the feedback log capped at 5 MB | Partial (see gaps) |
| One user exhausting everyone's rate limit behind a proxy | The client address is read from the right of `X-Forwarded-For` using a configured number of trusted proxy hops, so forged left-hand entries are ignored | Done in code, **not yet verified on the live host** |
| Prompt injection through abstracts or the question | The system prompt marks both as untrusted data, answers are limited to the retrieved abstracts, and a grounding check flags citations to sources that were never retrieved | Partial (a model can still be manipulated) |
| Fabricated or unsupported claims | Every answer is checked for invalid citations and for the share of sentences that carry a citation; failures are surfaced to the user | Partial (does not prove a paper supports a sentence) |
| Harm to a user in crisis | Crisis-related questions are blocked and pointed to 911 and 988 | Partial (a fixed pattern list misses indirect phrasing) |
| Leaked secrets in source control | History was scanned and holds no keys; `.env` files are ignored; CI scans every push for secrets | Done |
| Known-vulnerable dependencies | CI runs `pip-audit` and `npm audit` on every change and weekly; Dependabot opens update pull requests | Done |
| Container compromise | The image runs as a non-root user and ships only the runtime environment | Done |

## What the tests check

`backend/tests/test_security.py` covers constant-time comparison, request-ID cleaning,
client-address selection with forged headers, security headers, production-only headers,
the metrics token, production configuration rules, feedback input limits, and the log size
cap. Rate limiting (the per-minute cap, per-client buckets, and forged `X-Forwarded-For` entries) is covered in the same file, and authentication in `test_auth.py`.

## Known gaps

These are real limits. Anyone evaluating the project should hear them from here first.

1. **The API key is not a secret.** The website is a public single-page app, so the key it
   sends is inside the JavaScript anyone can download. It deters casual and automated abuse of
   the language-model endpoint; it does not stop a determined person. There is no per-user
   identity.
2. **Rate limits are per process and in memory.** With more than one instance, each keeps its
   own counters, and a restart resets them.
3. **The proxy hop count is unverified.** `TRUSTED_PROXY_HOPS` defaults to 0, which is safe
   but means a host's shared proxy address may be used as the rate-limit key. Set it only after
   confirming how many proxies sit in front of the app.
4. **A language model can be manipulated.** Prompt-injection defenses reduce the risk; they do
   not remove it. The citation check cannot tell whether a cited paper truly supports a claim.
5. **Third parties see the questions.** Groq receives each question and the retrieved
   abstracts, and PubMed receives search terms. Users are not told this in the interface yet.
6. **No external review.** No penetration test, privacy review, or clinical review has been
   done.

## How identity would be added

The honest fix for gap 1 is real sign-in:

1. Add OpenID Connect sign-in with an identity provider (for example Microsoft Entra ID or
   Amazon Cognito) using the authorization code flow with PKCE.
2. Have the API validate a short-lived access token on every request: signature against the
   provider's rotating public keys, issuer, audience, expiry, and required scope.
3. Rate limit and bill per user instead of per address, and keep the model key only on the
   server.
4. Keep an audit log of sign-ins and rejected tokens, and a way to revoke a user.
5. For an anonymous tier, add a bot check before issuing a short-lived session.
