/** Error codes produced by the client itself (server codes pass through as-is). */
export type ClientErrorCode =
  | "invalid_url"
  | "invalid_argument"
  | "network_error"
  | "timeout"
  | "aborted"
  | "invalid_response"
  | "no_fetch";

export interface HeapDBErrorInit {
  code: string;
  message: string;
  status?: number;
  retryAfter?: number | null;
  requestId?: string | null;
  insertedIds?: number[];
  cause?: unknown;
}

/**
 * Every failure surfaces as a `HeapDBError`.
 *
 * - `code`: the server's `error.code` (e.g. `monthly_quota_exceeded`, `rate_limited`,
 *   `not_found`), or a client code (`network_error`, `timeout`, `invalid_url`, ...).
 * - `status`: HTTP status, or 0 when no response was received.
 * - `retryAfter`: seconds from the `Retry-After` header, if any.
 *
 * Messages never contain the API key.
 */
export class HeapDBError extends Error {
  override readonly name = "HeapDBError";
  readonly code: string;
  readonly status: number;
  readonly retryAfter: number | null;
  readonly requestId: string | null;
  /** For `insertMany`: ids of the batches that were committed before the failure. */
  readonly insertedIds: number[] | undefined;

  constructor(init: HeapDBErrorInit) {
    super(init.message, init.cause === undefined ? undefined : { cause: init.cause });
    this.code = init.code;
    this.status = init.status ?? 0;
    this.retryAfter = init.retryAfter ?? null;
    this.requestId = init.requestId ?? null;
    this.insertedIds = init.insertedIds;
  }

  /** True for the monthly request quota (429 `monthly_quota_exceeded`), which is never retried. */
  get isQuotaExceeded(): boolean {
    return this.code === "monthly_quota_exceeded";
  }

  override toString(): string {
    return `${this.name} [${this.code}${this.status ? ` ${this.status}` : ""}]: ${this.message}`;
  }
}

/** Replace every occurrence of `secret` in `text`. */
export function redact(text: string, secret: string | undefined): string {
  if (!secret) return text;
  return text.split(secret).join(maskKey(secret));
}

/** `hdb_live_abc...xyz` -> `hdb_live_****`. */
export function maskKey(key: string): string {
  const m = /^([a-z]+_[a-z]+_)/i.exec(key);
  return `${m ? m[1] : ""}****`;
}
