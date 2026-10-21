import { describe, expect, it } from "vitest";
import { HEAPDB_SQL, HEAPDB_SQL_VERSION, HeapDBError, heaps } from "../src/index.js";
import { apiError, BASE, client, json } from "./helpers.js";

const T0 = 1_760_000_000;

const rankedWire = {
  rank: 1,
  id: "CHK/12",
  key: 0.4,
  score: "Infinity",
  heat: 1.5,
  base: 1,
  attrs: { kind: "bug" },
  version: 3,
};
const itemWire = {
  id: "CHK/12",
  base: 1,
  activity: 0.4,
  key: 0.4,
  attrs: { kind: "bug" },
  version: 1,
};
const infoWire = {
  name: "acme/checkout",
  half_life: 3600,
  landmark: T0,
  meta: { team: "pay" },
  size: 2,
};

describe("heaps over the server (mocked fetch)", () => {
  it("creates heaps with PUT and percent-encodes names", async () => {
    const { db, calls } = client([json(infoWire)]);
    const info = await heaps(db).createHeap("acme/checkout", 3600, {
      landmark: new Date(T0 * 1000),
      meta: { team: "pay" },
    });
    expect(info).toEqual({
      name: "acme/checkout",
      halfLife: 3600,
      landmark: T0,
      meta: { team: "pay" },
      size: 2,
    });
    expect(calls[0]!.method).toBe("PUT");
    expect(calls[0]!.url).toBe(`${BASE}/heaps/acme%2Fcheckout`);
    expect(calls[0]!.body).toEqual({ half_life: 3600, landmark: T0, meta: { team: "pay" } });
  });

  it("pushes batches and decodes items", async () => {
    const { db, calls } = client([
      json({ items: [itemWire, { ...itemWire, id: "b", version: 2 }] }),
    ]);
    const out = await heaps(db).pushMany("w", [
      { id: "CHK/12", weight: 1.5, t: T0, attrs: { kind: "bug" } },
      { id: "b" },
    ]);
    expect(out.map((i) => [i.id, i.version])).toEqual([
      ["CHK/12", 1],
      ["b", 2],
    ]);
    expect(calls[0]!.url).toBe(`${BASE}/heaps/w/push`);
    expect(calls[0]!.body).toEqual({
      items: [{ id: "CHK/12", weight: 1.5, time: T0, attrs: { kind: "bug" } }, { id: "b" }],
    });
  });

  it("push, setBase, setAttrs send one-item batches", async () => {
    const { db, calls } = client([json({ items: [itemWire] })]);
    const h = heaps(db);
    await h.push("w", "a", 2, { t: T0 });
    await h.setBase("w", "a", 4);
    await h.setAttrs("w", "a", { risk: 2 }, { merge: true });
    expect(calls.map((c) => [c.url.slice(BASE.length), c.body])).toEqual([
      ["/heaps/w/push", { items: [{ id: "a", weight: 2, time: T0 }] }],
      ["/heaps/w/base", { items: [{ id: "a", base: 4 }] }],
      ["/heaps/w/attrs", { items: [{ id: "a", attrs: { risk: 2 }, merge: true }] }],
    ]);
  });

  it("topK, pop, get, remove, rebase", async () => {
    const { db, calls } = client([
      json({ items: [rankedWire] }),
      json({ item: rankedWire }),
      json({ item: null }),
      json({ ...rankedWire, rank: 0 }),
      json({ removed: true }),
      json(infoWire),
    ]);
    const h = heaps(db);
    const filter = { all: [{ attr: "kind", op: "in" as const, value: ["bug"] }] };
    const top = await h.topK("w", { k: 3, now: T0, filter });
    expect(top[0]).toEqual({
      rank: 1,
      id: "CHK/12",
      key: 0.4,
      score: Number.POSITIVE_INFINITY,
      heat: 1.5,
      base: 1,
      attrs: { kind: "bug" },
      version: 3,
    });
    expect((await h.pop("w", { filter }))!.id).toBe("CHK/12");
    expect(await h.pop("w")).toBeNull();
    expect((await h.get("w", "CHK/12", { now: T0 }))!.rank).toBe(0);
    expect(await h.remove("w", "CHK/12")).toBe(true);
    expect((await h.rebase("w", { now: T0, halfLife: 60 })).halfLife).toBe(3600);
    expect(calls.map((c) => [c.method, c.url.slice(BASE.length), c.body])).toEqual([
      ["POST", "/heaps/w/top", { k: 3, now: T0, filter }],
      ["POST", "/heaps/w/pop", { filter }],
      ["POST", "/heaps/w/pop", {}],
      ["GET", `/heaps/w/items/CHK%2F12?now=${T0}`, undefined],
      ["DELETE", "/heaps/w/items/CHK%2F12", undefined],
      ["POST", "/heaps/w/rebase", { now: T0, half_life: 60 }],
    ]);
  });

  it("maps absence to null and keeps other errors", async () => {
    const { db } = client([
      apiError(404, "unknown_heap"),
      apiError(404, "unknown_item"),
      apiError(400, "invalid_argument", "items[1]: weight must be finite and > 0"),
      json({ heaps: [infoWire] }),
      json({ dropped: false }),
    ]);
    const h = heaps(db);
    expect(await h.heap("nope")).toBeNull();
    expect(await h.get("w", "nope")).toBeNull();
    const err = await h
      .pushMany("w", [{ id: "a" }, { id: "b", weight: 0 }])
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(HeapDBError);
    expect((err as HeapDBError).code).toBe("invalid_argument");
    expect((await h.listHeaps())[0]!.name).toBe("acme/checkout");
    expect(await h.dropHeap("w")).toBe(false);
  });

  it("rejects bad targets and arguments without a request", async () => {
    expect(() => heaps({} as never)).toThrow(/HeapDB connection or a client/);
    const { db, calls } = client([json({})]);
    await expect(heaps(db).push("", "a")).rejects.toThrow(/heap must be a non-empty string/);
    await expect(heaps(db).topK("w", { now: Number.NaN })).rejects.toThrow(
      /now must be Unix seconds/,
    );
    expect(calls).toHaveLength(0);
  });
});

describe("generated SQL", () => {
  it("ships sql/postgres/heapdb.sql", () => {
    expect(HEAPDB_SQL).toContain("CREATE SCHEMA IF NOT EXISTS heapdb");
    expect(HEAPDB_SQL).toContain(`VALUES (true, '${HEAPDB_SQL_VERSION}')`);
  });
});
