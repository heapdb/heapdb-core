import { afterEach, describe, expect, it, vi } from "vitest";
import { connect, HeapDBError, parseConnectionUrl } from "../src/index.js";
import { dataApiBase, redactUrl } from "../src/url.js";
import { DB, KEY, mockFetch, URL_ } from "./helpers.js";

describe("parseConnectionUrl", () => {
  it("parses key, host and database", () => {
    expect(parseConnectionUrl(URL_)).toEqual({ apiKey: KEY, host: "api.heapdb.com", database: DB });
  });

  it("accepts ports, trailing slash, whitespace and percent-encoded keys", () => {
    expect(parseConnectionUrl(" heapdb://hdb_live_a%2Bb@localhost:8080/db_x/ ")).toEqual({
      apiKey: "hdb_live_a+b",
      host: "localhost:8080",
      database: "db_x",
    });
    expect(parseConnectionUrl("HEAPDB://k@[::1]:8080/db").host).toBe("[::1]:8080");
  });

  it("ignores query and fragment", () => {
    expect(parseConnectionUrl("heapdb://k@h.example/db_1?x=1#f").database).toBe("db_1");
  });

  it.each([
    ["", "no HeapDB URL"],
    ["   ", "no HeapDB URL"],
    ["api.heapdb.com/db_x", "missing scheme"],
    ["https://k@api.heapdb.com/db_x", "scheme must be heapdb"],
    ["heapdb://api.heapdb.com/db_x", "missing API key"],
    ["heapdb://@api.heapdb.com/db_x", "bad API key"],
    ["heapdb://k k@api.heapdb.com/db_x", "bad API key"],
    ["heapdb://k@api.heapdb.com", "missing database id"],
    ["heapdb://k@api.heapdb.com/", "missing database id"],
    ["heapdb://k@/db_x", "bad host"],
    ["heapdb://k@bad_host!/db_x", "bad host"],
    ["heapdb://k@api.heapdb.com/db/x", "bad database id"],
    ["heapdb://k@api.heapdb.com/db x", "bad database id"],
  ])("rejects %j", (url, why) => {
    let err: unknown;
    try {
      parseConnectionUrl(url);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(HeapDBError);
    expect((err as HeapDBError).code).toBe("invalid_url");
    expect((err as HeapDBError).message).toContain(why);
  });

  it("never echoes the key in parse errors", () => {
    const bad = `heapdb://${KEY}@api.heapdb.com/db/with/slash`;
    expect(() => parseConnectionUrl(bad)).toThrow(HeapDBError);
    try {
      parseConnectionUrl(bad);
    } catch (e) {
      expect(String(e)).not.toContain(KEY);
      expect((e as Error).message).toContain("hdb_live_****");
    }
  });

  it("redactUrl masks the userinfo", () => {
    expect(redactUrl(URL_)).toBe(`heapdb://hdb_live_****@api.heapdb.com/${DB}`);
    expect(redactUrl("heapdb://plainkey@h/db")).toBe("heapdb://****@h/db");
  });
});

describe("dataApiBase", () => {
  const p = { apiKey: "k", host: "api.heapdb.com", database: "db_x" };
  it("defaults to https://<host>/v1/databases/<db>", () => {
    expect(dataApiBase(p)).toBe("https://api.heapdb.com/v1/databases/db_x");
  });
  it("uses baseUrl for self-hosted servers", () => {
    expect(dataApiBase(p, "http://localhost:8080")).toBe("http://localhost:8080/v1/databases/db_x");
    expect(dataApiBase(p, "https://proxy.internal/heapdb/")).toBe(
      "https://proxy.internal/heapdb/v1/databases/db_x",
    );
  });
  it("rejects a non-http baseUrl", () => {
    expect(() => dataApiBase(p, "localhost:8080")).toThrow(/invalid baseUrl/);
  });
});

describe("connect", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("reads HEAPDB_URL when no url is given", () => {
    vi.stubEnv("HEAPDB_URL", URL_);
    const db = connect({ fetch: mockFetch().fetch });
    expect(db.database).toBe(DB);
    expect(db.baseUrl).toBe(`https://api.heapdb.com/v1/databases/${DB}`);
  });

  it("an explicit url wins over HEAPDB_URL", () => {
    vi.stubEnv("HEAPDB_URL", URL_);
    const db = connect("heapdb://k@other.example/db_other", { fetch: mockFetch().fetch });
    expect(db.database).toBe("db_other");
  });

  it("fails clearly without url or HEAPDB_URL", () => {
    vi.stubEnv("HEAPDB_URL", "");
    expect(() => connect()).toThrow(/no HeapDB URL/);
  });

  it("validates options", () => {
    expect(() => connect(URL_, { retries: -1 })).toThrow(/retries/);
    expect(() => connect(URL_, { timeoutMs: -5 })).toThrow(/timeoutMs/);
  });

  it("does not expose the key via toString, JSON or inspect", async () => {
    const db = connect(URL_, { fetch: mockFetch().fetch });
    expect(String(db)).toBe(`HeapDB(heapdb://hdb_live_****@api.heapdb.com/${DB})`);
    expect(JSON.stringify(db)).not.toContain(KEY);
    const { inspect } = await import("node:util");
    expect(inspect(db)).not.toContain(KEY);
    expect(inspect(db, { showHidden: true, depth: 10 })).not.toContain(KEY);
    expect(Object.values(db).join(" ")).not.toContain(KEY);
  });
});
