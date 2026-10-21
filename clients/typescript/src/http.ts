import { HeapDBError, redact } from "./errors.js";
import type { FetchLike, RequestOptions } from "./types.js";

export interface TransportConfig {
  base: string;
  apiKey: string;
  fetch: FetchLike;
  timeoutMs: number;
  retries: number;
  retryBaseDelayMs: number;
  retryMaxDelayMs: number;
  userAgent: string;
  headers: Record<string, string>;
}

export type Query = Record<string, string | number | boolean | undefined | null>;

/** Statuses retried with backoff (besides network errors and timeouts). */
const RETRY_STATUS = new Set([429, 503]);
/** 429 codes that must never be retried. */
const NO_RETRY_CODES = new Set(["monthly_quota_exceeded"]);

/** Headers the client controls; caller-supplied copies are dropped. */
const RESERVED_HEADERS = new Set(["authorization", "content-type", "accept", "user-agent"]);

/** Parse `Retry-After` (delta-seconds or HTTP-date) into seconds. */
export function parseRetryAfter(value: string | null, now = Date.now()): number | null {
  if (value === null) return null;
  const v = value.trim();
  if (v === "") return null;
  if (/^\d+(\.\d+)?$/.test(v)) return Number(v);
  const t = Date.parse(v);
  if (Number.isNaN(t)) return null;
  return Math.max(0, (t - now) / 1000);
}

/** Full-jitter exponential backoff: uniform in [0, min(max, base * 2^attempt)]. */
export function backoffDelay(attempt: number, baseMs: number, maxMs: number): number {
  const ceiling = Math.min(maxMs, baseMs * 2 ** attempt);
  return Math.random() * ceiling;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortedError(signal));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortedError(signal));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function abortedError(signal: AbortSignal | undefined): HeapDBError {
  return new HeapDBError({
    code: "aborted",
    message: "request aborted",
    cause: signal?.reason,
  });
}

export class Transport {
  constructor(private readonly cfg: TransportConfig) {}

  get base(): string {
    return this.cfg.base;
  }

  async request<T>(
    method: "GET" | "POST" | "PUT" | "DELETE",
    path: string,
    body?: unknown,
    query?: Query,
    opts: RequestOptions = {},
  ): Promise<T> {
    const url = this.buildUrl(path, query);
    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries(this.cfg.headers)) {
      if (!RESERVED_HEADERS.has(k.toLowerCase())) headers[k] = v;
    }
    Object.assign(headers, {
      Accept: "application/json",
      Authorization: `Bearer ${this.cfg.apiKey}`,
      "User-Agent": this.cfg.userAgent,
    });
    let payload: string | undefined;
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
    const timeoutMs = opts.timeoutMs ?? this.cfg.timeoutMs;

    for (let attempt = 0; ; attempt++) {
      let err: HeapDBError;
      try {
        return await this.once<T>(method, url, headers, payload, timeoutMs, opts.signal);
      } catch (e) {
        if (!(e instanceof HeapDBError)) throw e;
        err = e;
      }
      const delay = this.retryDelay(err, attempt);
      if (delay === null) throw err;
      await sleep(delay, opts.signal);
    }
  }

  /** ms to wait before retrying `err`, or null to give up. */
  private retryDelay(err: HeapDBError, attempt: number): number | null {
    if (attempt >= this.cfg.retries) return null;
    const transient =
      err.code === "network_error" ||
      err.code === "timeout" ||
      (RETRY_STATUS.has(err.status) && !NO_RETRY_CODES.has(err.code));
    if (!transient) return null;
    if (err.retryAfter !== null) {
      const ms = err.retryAfter * 1000;
      // A server asking us to wait longer than we are willing to: surface it now.
      if (ms > this.cfg.retryMaxDelayMs) return null;
      return ms;
    }
    return backoffDelay(attempt, this.cfg.retryBaseDelayMs, this.cfg.retryMaxDelayMs);
  }

  private buildUrl(path: string, query?: Query): string {
    let url = this.cfg.base + path;
    if (query) {
      const parts: string[] = [];
      for (const [k, v] of Object.entries(query)) {
        if (v === undefined || v === null) continue;
        parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
      }
      if (parts.length) url += `?${parts.join("&")}`;
    }
    return url;
  }

  private async once<T>(
    method: string,
    url: string,
    headers: Record<string, string>,
    payload: string | undefined,
    timeoutMs: number,
    signal: AbortSignal | undefined,
  ): Promise<T> {
    if (signal?.aborted) throw abortedError(signal);
    const ctrl = new AbortController();
    let timedOut = false;
    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            timedOut = true;
            ctrl.abort();
          }, timeoutMs)
        : undefined;
    const onAbort = () => ctrl.abort(signal?.reason);
    signal?.addEventListener("abort", onAbort, { once: true });

    let res: Response;
    let text: string;
    try {
      const init: RequestInit = { method, headers, signal: ctrl.signal };
      if (payload !== undefined) init.body = payload;
      res = await this.cfg.fetch(url, init);
      text = await res.text();
    } catch (e) {
      if (timedOut) {
        throw new HeapDBError({
          code: "timeout",
          message: `${method} ${this.describe(url)} timed out after ${timeoutMs} ms`,
        });
      }
      if (signal?.aborted) throw abortedError(signal);
      const detail = e instanceof Error ? e.message : String(e);
      throw new HeapDBError({
        code: "network_error",
        message: `${method} ${this.describe(url)} failed: ${this.scrub(detail)}`,
      });
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }

    const requestId = res.headers.get("x-request-id");
    let data: unknown;
    if (text !== "") {
      try {
        data = JSON.parse(text);
      } catch {
        data = undefined;
      }
    }

    if (res.ok) {
      if (text !== "" && data === undefined) {
        throw new HeapDBError({
          code: "invalid_response",
          message: `${method} ${this.describe(url)}: response is not JSON`,
          status: res.status,
          requestId,
        });
      }
      return (data ?? {}) as T;
    }

    const errObj = (data as { error?: { code?: unknown; message?: unknown } } | undefined)?.error;
    const code =
      typeof errObj?.code === "string" && errObj.code !== "" ? errObj.code : `http_${res.status}`;
    const serverMsg =
      typeof errObj?.message === "string" && errObj.message !== ""
        ? errObj.message
        : text.slice(0, 200) || res.statusText || "request failed";
    throw new HeapDBError({
      code,
      status: res.status,
      message: this.scrub(`${method} ${this.describe(url)}: ${serverMsg}`),
      retryAfter: parseRetryAfter(res.headers.get("retry-after")),
      requestId,
    });
  }

  /** Path relative to the database base, for messages. */
  private describe(url: string): string {
    const rel = url.startsWith(this.cfg.base) ? url.slice(this.cfg.base.length) : url;
    return rel === "" ? "/" : rel;
  }

  private scrub(text: string): string {
    return redact(text, this.cfg.apiKey);
  }
}
