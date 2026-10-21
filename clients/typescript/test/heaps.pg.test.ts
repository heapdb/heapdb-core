/**
 * The Postgres transport against a real Postgres. Runs only when
 * HEAPDB_TEST_POSTGRES_URL is set (a server URL); creates and drops a scratch
 * database.
 */
import pg from "pg";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { HEAPDB_SQL_VERSION, type HeapDBError, heaps, installHeapSql } from "../src/index.js";

const url = process.env.HEAPDB_TEST_POSTGRES_URL;
const T0 = 1_760_000_000;

function withDb(u: string, db: string): string {
  const x = new URL(u);
  x.pathname = `/${db}`;
  return x.toString();
}

describe.skipIf(!url)("heaps over Postgres (HEAPDB_TEST_POSTGRES_URL)", () => {
  const name = `heapdb_ts_${process.pid}_${Date.now() % 1e6}`;
  let pool: pg.Pool;

  beforeAll(async () => {
    const admin = new pg.Client({ connectionString: url });
    await admin.connect();
    await admin.query(`CREATE DATABASE ${name}`);
    await admin.end();
    pool = new pg.Pool({ connectionString: withDb(url!, name), max: 8 });
    expect(await installHeapSql(pool)).toBe(HEAPDB_SQL_VERSION);
    expect(await installHeapSql(pool)).toBe(HEAPDB_SQL_VERSION); // idempotent
  });

  afterAll(async () => {
    await pool?.end();
    const admin = new pg.Client({ connectionString: url });
    await admin.connect();
    await admin.query(`DROP DATABASE IF EXISTS ${name} WITH (FORCE)`);
    await admin.end();
  });

  it("runs the doc flow", async () => {
    const h = heaps(pool);
    const info = await h.createHeap("acme/checkout", 3 * 86400, {
      landmark: T0,
      meta: { team: "pay" },
    });
    expect(info).toEqual({
      name: "acme/checkout",
      halfLife: 259200,
      landmark: T0,
      meta: { team: "pay" },
      size: 0,
    });
    expect(await h.createHeap("acme/checkout", 3 * 86400)).toEqual(info);
    await expect(h.createHeap("acme/checkout", 60)).rejects.toMatchObject({
      code: "heap_conflict",
    });
    const it1 = await h.push("acme/checkout", "CHK-12", 1.5, {
      t: T0,
      attrs: { kind: "bug", required_skills: ["ts"] },
    });
    expect(it1.version).toBe(1);
    expect(it1.key).toBeCloseTo(Math.log(1.5), 12);
    await h.push("acme/checkout", "CHK-13", 1, {
      t: T0,
      attrs: { kind: "task", labels: ["needs-human"] },
    });
    const b = await h.setBase("acme/checkout", "CHK-12", 4);
    expect(b.base).toBe(4);
    const merged = await h.setAttrs("acme/checkout", "CHK-13", { risk: 2 }, { merge: true });
    expect(merged.attrs).toEqual({ kind: "task", labels: ["needs-human"], risk: 2 });

    const top = await h.topK("acme/checkout", {
      k: 5,
      now: T0 + 3 * 86400,
      filter: {
        all: [
          { attr: "kind", op: "in", value: ["bug", "task"] },
          { attr: "required_skills", op: "subset_of", value: ["ts", "react"] },
        ],
      },
    });
    expect(top.map((r) => r.id)).toEqual(["CHK-12", "CHK-13"]);
    expect(top[0]!.score).toBeCloseTo(4 * 1.5 * 0.5, 9);
    expect(top[0]!.heat).toBeCloseTo(0.75, 9);

    const popped = await h.pop("acme/checkout", {
      now: T0,
      filter: { not: { attr: "labels", op: "contains", value: "needs-human" } },
    });
    expect(popped!.id).toBe("CHK-12");
    expect(await h.get("acme/checkout", "CHK-12")).toBeNull();
    expect((await h.get("acme/checkout", "CHK-13", { now: T0 }))!.rank).toBe(0);
    const before = await h.topK("acme/checkout", { now: T0 + 1000 });
    await h.rebase("acme/checkout", { now: T0 + 500 });
    const after = await h.topK("acme/checkout", { now: T0 + 1000 });
    expect(after[0]!.score).toBeCloseTo(before[0]!.score, 12);
    expect(await h.remove("acme/checkout", "CHK-13")).toBe(true);
    expect((await h.listHeaps()).map((x) => x.name)).toEqual(["acme/checkout"]);
    expect(await h.dropHeap("acme/checkout")).toBe(true);
    expect(await h.heap("acme/checkout")).toBeNull();
  });

  it("pushMany is atomic, now defaults to the database clock", async () => {
    const h = heaps(pool);
    await h.createHeap("q", 60);
    const err = await h
      .pushMany("q", [{ id: "a" }, { id: "b", weight: 0 }])
      .catch((e: unknown) => e as HeapDBError);
    expect((err as HeapDBError).code).toBe("invalid_argument");
    expect((await h.heap("q"))!.size).toBe(0);
    const out = await h.pushMany("q", [{ id: "a" }, { id: "b", weight: 2 }, { id: "a" }]);
    expect(out.map((i) => [i.id, i.version])).toEqual([
      ["a", 1],
      ["b", 1],
      ["a", 2],
    ]);
    const top = await h.topK("q");
    expect(top.map((r) => r.id)).toEqual(["a", "b"]);
    expect(top[0]!.heat).toBeGreaterThan(1.9);
    await expect(h.push("missing", "a")).rejects.toMatchObject({
      code: "unknown_heap",
      status: 404,
    });
    await expect(
      h.topK("q", { filter: { attr: "x", op: "like", value: 1 } as never }),
    ).rejects.toMatchObject({
      code: "invalid_argument",
    });
  });

  it("concurrent pops never share an item", async () => {
    const h = heaps(pool);
    await h.createHeap("race", 3600, { landmark: T0 });
    await h.pushMany(
      "race",
      Array.from({ length: 200 }, (_, i) => ({ id: `i${i}`, weight: 1 + (i % 7), t: T0 })),
    );
    const got: string[] = [];
    await Promise.all(
      Array.from({ length: 8 }, async () => {
        for (;;) {
          const r = await h.pop("race", { now: T0 });
          if (!r) {
            if ((await h.heap("race"))!.size === 0) return;
            continue;
          }
          got.push(r.id);
        }
      }),
    );
    expect(new Set(got).size).toBe(got.length);
    expect(got).toHaveLength(200);
  });
});
