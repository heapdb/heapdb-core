import { type ConnectOptions, connect, type HeapDB } from "../src/index.js";

export const KEY = "hdb_live_s3cr3tK3yABCDEFGHIJKLMNOPQRSTUVWXYZ0123";
export const DB = "db_7Kq2abcdefghijklmn";
export const URL_ = `heapdb://${KEY}@api.heapdb.com/${DB}`;
export const BASE = `https://api.heapdb.com/v1/databases/${DB}`;

export interface Call {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

type Reply = Response | Error | ((call: Call) => Response | Promise<Response>);

export function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

export function apiError(
  status: number,
  code: string,
  message = code,
  headers: Record<string, string> = {},
): Response {
  return json({ error: { code, message } }, status, headers);
}

/** A fetch mock that replays `replies` in order (the last one repeats). */
export function mockFetch(...replies: Reply[]) {
  const calls: Call[] = [];
  let i = 0;
  const fetch = async (input: string, init: RequestInit): Promise<Response> => {
    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries((init.headers ?? {}) as Record<string, string>)) {
      headers[k.toLowerCase()] = v;
    }
    const call: Call = {
      url: input,
      method: init.method ?? "GET",
      headers,
      body: typeof init.body === "string" ? JSON.parse(init.body) : undefined,
    };
    calls.push(call);
    const r = replies[Math.min(i++, replies.length - 1)];
    if (r === undefined) throw new Error("no reply configured");
    if (r instanceof Error) throw r;
    if (typeof r === "function") return r(call);
    return r.clone();
  };
  return { fetch, calls };
}

export function client(replies: Reply[], opts: ConnectOptions = {}): { db: HeapDB; calls: Call[] } {
  const m = mockFetch(...replies);
  const db = connect(URL_, { retryBaseDelayMs: 1, retryMaxDelayMs: 50, ...opts, fetch: m.fetch });
  return { db, calls: m.calls };
}
