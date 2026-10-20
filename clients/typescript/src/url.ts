import { HeapDBError, maskKey } from "./errors.js";

export interface ParsedUrl {
  apiKey: string;
  host: string;
  database: string;
}

const DB_RE = /^[A-Za-z0-9_-]+$/;
const HOST_RE = /^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)(:\d{1,5})?$/;

/** Describe a URL without its secret, for error messages. */
export function redactUrl(url: string): string {
  const m = /^([a-z][a-z0-9+.-]*:\/\/)([^@/]*)@/i.exec(url);
  if (!m) return url.length > 80 ? `${url.slice(0, 80)}...` : url;
  return `${m[1]}${maskKey(decodeSafe(m[2] ?? ""))}@${url.slice(m[0].length)}`;
}

function decodeSafe(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

function invalid(url: string, why: string): HeapDBError {
  return new HeapDBError({
    code: "invalid_url",
    message: `invalid HeapDB URL (${why}): ${redactUrl(url)}; expected heapdb://<api_key>@<host>/<database_id>`,
  });
}

/** Parse `heapdb://<api_key>@<host>[:port]/<database_id>`. */
export function parseConnectionUrl(url: string): ParsedUrl {
  if (typeof url !== "string" || url.trim() === "") {
    throw new HeapDBError({
      code: "invalid_url",
      message:
        "no HeapDB URL: pass connect(url) or set HEAPDB_URL (heapdb://<api_key>@<host>/<database_id>)",
    });
  }
  const s = url.trim();
  const scheme = /^([a-z][a-z0-9+.-]*):\/\//i.exec(s);
  if (!scheme) throw invalid(s, "missing scheme");
  if (scheme[1]?.toLowerCase() !== "heapdb") throw invalid(s, `scheme must be heapdb://`);
  const rest = s.slice(scheme[0].length);
  const at = rest.lastIndexOf("@");
  if (at < 0) throw invalid(s, "missing API key");
  const apiKey = decodeSafe(rest.slice(0, at));
  if (apiKey === "" || /[\s/:]/.test(apiKey)) throw invalid(s, "bad API key");
  const hostAndPath = rest.slice(at + 1).replace(/[?#].*$/, "");
  const slash = hostAndPath.indexOf("/");
  if (slash < 0) throw invalid(s, "missing database id");
  const host = hostAndPath.slice(0, slash);
  if (!host || !HOST_RE.test(host)) throw invalid(s, "bad host");
  const database = hostAndPath.slice(slash + 1).replace(/\/+$/, "");
  if (!database) throw invalid(s, "missing database id");
  if (!DB_RE.test(database)) throw invalid(s, "bad database id");
  return { apiKey, host, database };
}

/** Compute the Data API base URL for a parsed connection URL. */
export function dataApiBase(parsed: ParsedUrl, baseUrl?: string): string {
  let origin: string;
  if (baseUrl !== undefined) {
    if (!/^https?:\/\/[^/]+/i.test(baseUrl)) {
      throw new HeapDBError({
        code: "invalid_url",
        message: `invalid baseUrl ${JSON.stringify(baseUrl)}: expected http(s)://host[:port][/prefix]`,
      });
    }
    origin = baseUrl.replace(/\/+$/, "");
  } else {
    origin = `https://${parsed.host}`;
  }
  return `${origin}/v1/databases/${encodeURIComponent(parsed.database)}`;
}

/** Read `HEAPDB_URL` when a `process.env` (Node, Bun, Deno with compat) exists. */
export function envUrl(): string | undefined {
  try {
    const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
    const v = proc?.env?.HEAPDB_URL;
    return v && v.trim() !== "" ? v : undefined;
  } catch {
    // Deno without --allow-env throws on access.
    return undefined;
  }
}
