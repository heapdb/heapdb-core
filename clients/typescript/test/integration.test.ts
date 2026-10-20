/**
 * Live test against a real heapdb-server. Runs only when HEAPDB_TEST_URL is set,
 * e.g. HEAPDB_TEST_URL=heapdb://<key>@localhost:8080/<db> HEAPDB_TEST_BASE_URL=http://localhost:8080
 * Use a throwaway database: it registers an effect and inserts events.
 */
import { describe, expect, it } from "vitest";
import { connect, kernels } from "../src/index.js";

const url = process.env.HEAPDB_TEST_URL;
const baseUrl = process.env.HEAPDB_TEST_BASE_URL;

describe.skipIf(!url)("integration (HEAPDB_TEST_URL)", () => {
  const H = 3600;
  const entity = `ts-it-${Date.now()}`;

  it("round-trips the quickstart", async () => {
    const db = connect(url, baseUrl ? { baseUrl } : {});
    const info = await db.info();
    expect(info.databaseId).toBe(db.database);

    await db.registerEffect("bloating", {
      kernel: kernels.lognormal(2 * H, { sigma: 0.6 }),
      priors: { beans: [3, 1], dairy: [2, 1] },
    });
    const ids = await db.insertMany([
      { entity, time: 8 * H, type: "meal", features: ["oats", "dairy"] },
      { entity, time: 13 * H, type: "meal", features: ["beans", "onion"], weight: 1.5 },
    ]);
    expect(ids).toHaveLength(2);

    const res = await db.topCauses({ entity, effect: "bloating", at: 15 * H, k: 3, explain: true });
    expect(res.causes.length).toBeGreaterThan(0);
    expect(res.causes[0]!.event.id).toBe(ids[1]);
    expect(res.causes[0]!.explanation).not.toBeNull();

    const fb = await db.feedback({
      entity,
      effect: "bloating",
      at: 15 * H,
      confirmedEventId: ids[1]!,
      rejectedIds: [ids[0]!],
    });
    expect(fb.confirmed).toBe(ids[1]);

    const links = await db.linkReport({ effect: "bloating", entity });
    expect(links.rows.some((r) => r.feature === "beans")).toBe(true);

    const counts = await db.forgetEntity(entity);
    expect(Object.keys(counts).length).toBeGreaterThan(0);
  });
});
