import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { backoffDelay, parseRetryAfter } from "../src/http.js";
import { connect, HeapDBError } from "../src/index.js";
import { apiError, client, json, KEY, mockFetch, URL_ } from "./helpers.js";

const ids = () => json({ ids: [1] });
const ev = { entity: "e", time: 1, type: "t" };

describe("retries with fake timers", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0.5);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("retries 503 with exponential backoff + jitter, then succeeds", async () => {
    const m = mockFetch(apiError(503, "unavailable"), apiError(503, "unavailable"), ids());
    const db = connect(URL_, { fetch: m.fetch, retryBaseDelayMs: 100, retryMaxDelayMs: 10_000 });
    const p = db.insert(ev);
    await vi.advanceTimersByTimeAsync(0);
    expect(m.calls).toHaveLength(1);
    // attempt 0: random(0.5) * min(10000, 100 * 2^0) = 50 ms
    await vi.advanceTimersByTimeAsync(49);
    expect(m.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(m.calls).toHaveLength(2);
    // attempt 1: 0.5 * 200 = 100 ms
    await vi.advanceTimersByTimeAsync(99);
    expect(m.calls).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(m.calls).toHaveLength(3);
    await expect(p).resolves.toBe(1);
  });

  it("honours Retry-After (seconds) on 429", async () => {
    const m = mockFetch(apiError(429, "rate_limited", "slow down", { "Retry-After": "2" }), ids());
    const db = connect(URL_, { fetch: m.fetch });
    const p = db.insert(ev);
    await vi.advanceTimersByTimeAsync(1999);
    expect(m.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(m.calls).toHaveLength(2);
    await expect(p).resolves.toBe(1);
  });

  it("honours Retry-After as an HTTP date", async () => {
    vi.setSystemTime(new Date("2026-10-20T00:00:00Z"));
    const m = mockFetch(
      apiError(503, "unavailable", "x", { "Retry-After": "Tue, 20 Oct 2026 00:00:03 GMT" }),
      ids(),
    );
    const db = connect(URL_, { fetch: m.fetch });
    const p = db.insert(ev);
    await vi.advanceTimersByTimeAsync(2999);
    expect(m.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    await expect(p).resolves.toBe(1);
  });

  it("gives up when Retry-After exceeds retryMaxDelayMs", async () => {
    const m = mockFetch(apiError(429, "rate_limited", "x", { "Retry-After": "120" }), ids());
    const db = connect(URL_, { fetch: m.fetch, retryMaxDelayMs: 30_000 });
    const err = await db.insert(ev).catch((e: unknown) => e);
    expect(m.calls).toHaveLength(1);
    expect(err).toMatchObject({ code: "rate_limited", status: 429, retryAfter: 120 });
  });

  it("retries network errors and stops after `retries`", async () => {
    const m = mockFetch(new TypeError("fetch failed"));
    const db = connect(URL_, { fetch: m.fetch, retries: 2, retryBaseDelayMs: 10 });
    const p = db.insert(ev).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(1000);
    const err = await p;
    expect(m.calls).toHaveLength(3);
    expect(err).toBeInstanceOf(HeapDBError);
    expect(err).toMatchObject({ code: "network_error", status: 0 });
    expect((err as Error).message).toContain("fetch failed");
  });

  it("times out a hung attempt and retries it", async () => {
    let n = 0;
    const fetch = (_u: string, init: RequestInit): Promise<Response> => {
      n++;
      if (n === 1) {
        return new Promise((_res, rej) => {
          init.signal?.addEventListener("abort", () =>
            rej(new DOMException("aborted", "AbortError")),
          );
        });
      }
      return Promise.resolve(ids());
    };
    const db = connect(URL_, { fetch, timeoutMs: 1000, retryBaseDelayMs: 10 });
    const p = db.insert(ev);
    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(10);
    await expect(p).resolves.toBe(1);
    expect(n).toBe(2);
  });

  it("reports timeout when retries are exhausted", async () => {
    const fetch = (_u: string, init: RequestInit): Promise<Response> =>
      new Promise((_res, rej) => {
        init.signal?.addEventListener("abort", () => rej(new Error("aborted")));
      });
    const db = connect(URL_, { fetch, timeoutMs: 500, retries: 0 });
    const p = db.info().catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(500);
    expect(await p).toMatchObject({ code: "timeout", status: 0 });
  });

  it("caller abort stops a pending backoff", async () => {
    const m = mockFetch(apiError(503, "unavailable"));
    const db = connect(URL_, { fetch: m.fetch, retryBaseDelayMs: 10_000 });
    const ctrl = new AbortController();
    const p = db.insert(ev, { signal: ctrl.signal }).catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(10);
    ctrl.abort();
    expect(await p).toMatchObject({ code: "aborted" });
    expect(m.calls).toHaveLength(1);
  });
});

describe("no retry", () => {
  it("never retries monthly_quota_exceeded", async () => {
    const { db, calls } = client(
      [
        apiError(429, "monthly_quota_exceeded", "monthly request quota reached", {
          "Retry-After": "86400",
        }),
        ids(),
      ],
      { retryMaxDelayMs: 10 ** 9 },
    );
    const err = (await db.insert(ev).catch((e: unknown) => e)) as HeapDBError;
    expect(calls).toHaveLength(1);
    expect(err).toBeInstanceOf(HeapDBError);
    expect(err.code).toBe("monthly_quota_exceeded");
    expect(err.status).toBe(429);
    expect(err.retryAfter).toBe(86400);
    expect(err.isQuotaExceeded).toBe(true);
  });

  it.each([400, 401, 403, 404, 409, 413, 500, 502])("does not retry HTTP %i", async (status) => {
    const { db, calls } = client([apiError(status, `code_${status}`), ids()]);
    await expect(db.insert(ev)).rejects.toMatchObject({ status, code: `code_${status}` });
    expect(calls).toHaveLength(1);
  });

  it("retries: 0 disables retries", async () => {
    const { db, calls } = client([apiError(503, "unavailable"), ids()], { retries: 0 });
    await expect(db.insert(ev)).rejects.toMatchObject({ status: 503 });
    expect(calls).toHaveLength(1);
  });

  it("does not retry client-side validation errors", async () => {
    const { db, calls } = client([ids()]);
    await expect(
      db.insert({ entity: "e", time: Number.POSITIVE_INFINITY, type: "t" }),
    ).rejects.toMatchObject({
      code: "invalid_argument",
    });
    expect(calls).toHaveLength(0);
  });
});

describe("error mapping", () => {
  it("maps {error:{code,message}} with status and request id", async () => {
    const { db } = client([
      json({ error: { code: "not_found", message: "unknown effect 'x'" } }, 404, {
        "x-request-id": "req_1",
      }),
    ]);
    const err = (await db
      .topCauses({ entity: "e", effect: "x", at: 1 })
      .catch((e) => e)) as HeapDBError;
    expect(err).toBeInstanceOf(HeapDBError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("HeapDBError");
    expect(err.code).toBe("not_found");
    expect(err.status).toBe(404);
    expect(err.requestId).toBe("req_1");
    expect(err.retryAfter).toBeNull();
    expect(err.message).toBe("POST /query/top-causes: unknown effect 'x'");
    expect(String(err)).toBe(
      "HeapDBError [not_found 404]: POST /query/top-causes: unknown effect 'x'",
    );
  });

  it("falls back to http_<status> for non-JSON error bodies", async () => {
    const { db } = client([new Response("<html>Bad Gateway</html>", { status: 502 })]);
    await expect(db.info()).rejects.toMatchObject({ code: "http_502", status: 502 });
  });

  it("flags non-JSON success bodies", async () => {
    const { db } = client([new Response("not json", { status: 200 })]);
    await expect(db.info()).rejects.toMatchObject({ code: "invalid_response" });
  });

  it("maps 401/403 auth errors", async () => {
    const { db } = client([apiError(401, "unauthorized", "invalid API key")]);
    await expect(db.info()).rejects.toMatchObject({ code: "unauthorized", status: 401 });
  });
});

describe("key redaction", () => {
  it("scrubs the key from server error messages", async () => {
    const { db } = client([apiError(403, "forbidden", `key ${KEY} lacks admin scope`)]);
    const err = (await db.registerEventType("t").catch((e) => e)) as HeapDBError;
    expect(err.message).not.toContain(KEY);
    expect(err.message).toContain("hdb_live_****");
    expect(String(err)).not.toContain(KEY);
    expect(JSON.stringify(err)).not.toContain(KEY);
  });

  it("scrubs the key from network error messages", async () => {
    const { db } = client([new TypeError(`connect ECONNREFUSED for Bearer ${KEY}`)], {
      retries: 0,
    });
    const err = (await db.info().catch((e) => e)) as HeapDBError;
    expect(err.code).toBe("network_error");
    expect(err.message).not.toContain(KEY);
  });

  it("never logs the key", async () => {
    const spies = (["log", "info", "warn", "error", "debug"] as const).map((k) =>
      vi.spyOn(console, k).mockImplementation(() => {}),
    );
    const { db } = client([apiError(503, "unavailable"), ids()]);
    await db.insert(ev);
    for (const s of spies) expect(s).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});

describe("helpers", () => {
  it("parseRetryAfter", () => {
    expect(parseRetryAfter(null)).toBeNull();
    expect(parseRetryAfter("")).toBeNull();
    expect(parseRetryAfter("3")).toBe(3);
    expect(parseRetryAfter("1.5")).toBe(1.5);
    expect(parseRetryAfter("garbage")).toBeNull();
    const now = Date.parse("2026-10-20T00:00:00Z");
    expect(parseRetryAfter("Tue, 20 Oct 2026 00:00:10 GMT", now)).toBe(10);
    expect(parseRetryAfter("Mon, 19 Oct 2026 00:00:00 GMT", now)).toBe(0);
  });

  it("backoffDelay is full jitter capped at max", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.999999);
    expect(backoffDelay(0, 100, 1000)).toBeCloseTo(100, 2);
    expect(backoffDelay(3, 100, 1000)).toBeCloseTo(800, 2);
    expect(backoffDelay(10, 100, 1000)).toBeCloseTo(1000, 2);
    vi.spyOn(Math, "random").mockReturnValue(0);
    expect(backoffDelay(5, 100, 1000)).toBe(0);
    vi.restoreAllMocks();
  });
});
