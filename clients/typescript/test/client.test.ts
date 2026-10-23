import { describe, expect, it } from "vitest";
import { HeapDBError, kernels, VERSION } from "../src/index.js";
import { BASE, client, json, KEY } from "./helpers.js";

const ok = () => json({ ok: true });

const explanationWire = {
  dt: 5400,
  kernel_value: 0.845,
  weight: 0.8,
  link_strength: 0.67,
  features: [{ feature: "dairy", strength: 0.5, alpha: 2, beta: 2 }],
  warnings: ["'coffee', 'dairy' always co-occur"],
  rules: [{ name: "small", multiplier: 0.3 }],
  joint_causes: [
    {
      features: ["coffee", "dairy"],
      strength: 0.83,
      joint: { prior_alpha: 1, prior_beta: 4, pos: 4.8, neg: 0, global_pos: 1, global_neg: 0.2 },
      inseparable: true,
      name: "coffee+dairy",
    },
  ],
  entity: "me",
  prospective: false,
  rule_multiplier: 0.3,
  score: 0.4528,
};

const eventWire = {
  id: 23,
  entity: "me",
  time: 484200,
  type: "meal",
  features: ["coffee", "dairy"],
  weight: 0.8,
  attrs: { portion_size: 2 },
};

describe("headers", () => {
  it("sends bearer auth, JSON content type and user agent", async () => {
    const { db, calls } = client([ok()], {
      userAgent: "my-agent/1.2",
      headers: { "X-Trace": "t" },
    });
    await db.registerEventType("meal");
    const h = calls[0]!.headers;
    expect(h.authorization).toBe(`Bearer ${KEY}`);
    expect(h["content-type"]).toBe("application/json");
    expect(h.accept).toBe("application/json");
    expect(h["user-agent"]).toBe(`heapdb-ts/${VERSION} my-agent/1.2`);
    expect(h["x-trace"]).toBe("t");
  });

  it("does not send a content type on GET/DELETE", async () => {
    const { db, calls } = client([json({ database_id: "db", plan: "free" })]);
    await db.info();
    expect(calls[0]!.headers["content-type"]).toBeUndefined();
    expect(calls[0]!.body).toBeUndefined();
  });

  it("caller headers cannot override Authorization", async () => {
    const { db, calls } = client([ok()], {
      headers: { authorization: "Bearer other", "User-Agent": "x" },
    });
    await db.registerEventType("meal");
    expect(calls[0]!.headers.authorization).toBe(`Bearer ${KEY}`);
  });
});

describe("schema endpoints", () => {
  it("registerEffect -> POST /effects (snake_case, kernel spec)", async () => {
    const { db, calls } = client([ok()]);
    await db.registerEffect("bloating", {
      kernel: kernels.lognormal(7200, { sigma: 0.6 }),
      priors: { beans: [3, 1], dairy: [2, 1] },
      defaultPrior: [1, 4],
      causeTypes: new Set(["meal"]),
      credit: "proportional",
      globalPseudocount: 5,
      leak: 0.05,
    });
    expect(calls[0]!.method).toBe("POST");
    expect(calls[0]!.url).toBe(`${BASE}/effects`);
    expect(calls[0]!.body).toEqual({
      name: "bloating",
      kernel: { type: "lognormal", params: { median: 7200, sigma: 0.6 } },
      priors: { beans: [3, 1], dairy: [2, 1] },
      default_prior: [1, 4],
      cause_types: ["meal"],
      credit: "proportional",
      global_pseudocount: 5,
      leak: 0.05,
    });
  });

  it("registerEffect sends combine, learning_rate and l2 for logistic effects", async () => {
    const { db, calls } = client([ok()]);
    await db.registerEffect("incident", {
      kernel: kernels.window(0, 7200),
      combine: "logistic",
      learningRate: 0.5,
      l2: 2,
    });
    expect(calls[0]!.body).toMatchObject({
      name: "incident",
      combine: "logistic",
      learning_rate: 0.5,
      l2: 2,
    });
    expect(calls[0]!.body).not.toHaveProperty("credit");
  });

  it("registerEffect omits unset options and accepts literal kernel specs", async () => {
    const { db, calls } = client([ok()]);
    await db.registerEffect("incident", {
      kernel: { type: "exponential", params: { half_life: 3600 } },
    });
    expect(calls[0]!.body).toEqual({
      name: "incident",
      kernel: { type: "exponential", params: { half_life: 3600 } },
    });
  });

  it("kernel helpers", () => {
    expect(kernels.exponential(3600)).toEqual({ type: "exponential", params: { half_life: 3600 } });
    expect(kernels.exponential(60, { horizonHalfLives: 5 })).toEqual({
      type: "exponential",
      params: { half_life: 60, horizon_half_lives: 5 },
    });
    expect(kernels.window(0, 600)).toEqual({
      type: "window",
      params: { min_lag: 0, max_lag: 600 },
    });
    expect(kernels.gamma(2, 1800, { tailStds: 4 })).toEqual({
      type: "gamma",
      params: { shape: 2, scale: 1800, tail_stds: 4 },
    });
    expect(kernels.lognormal(10, { tailSigmas: 3 }).params).toEqual({ median: 10, tail_sigmas: 3 });
  });

  it("registerEventType -> POST /event-types with declarative weight", async () => {
    const { db, calls } = client([ok()]);
    await db.registerEventType("pr", { kind: "log1p", attr: "lines_changed" });
    expect(calls[0]!.url).toBe(`${BASE}/event-types`);
    expect(calls[0]!.body).toEqual({
      type: "pr",
      weight: { kind: "log1p", attr: "lines_changed" },
    });
  });

  it("addRule -> POST /rules with declarative predicate", async () => {
    const { db, calls } = client([ok(), ok()]);
    await db.addRule("fraud", "small", {
      multiplier: 0.3,
      types: ["txn"],
      predicate: { attr: "amount", op: "lt", value: 5 },
    });
    expect(calls[0]!.url).toBe(`${BASE}/rules`);
    expect(calls[0]!.body).toEqual({
      effect: "fraud",
      name: "small",
      multiplier: 0.3,
      types: ["txn"],
      predicate: { attr: "amount", op: "lt", value: 5 },
    });
    await db.addRule("incident", "docs_only", { multiplier: 0, predicate: null });
    expect(calls[1]!.body).toEqual({ effect: "incident", name: "docs_only", multiplier: 0 });
  });

  it("validates required arguments before any request", async () => {
    const { db, calls } = client([ok()]);
    await expect(db.registerEffect("", { kernel: kernels.exponential(1) })).rejects.toMatchObject({
      code: "invalid_argument",
    });
    // @ts-expect-error kernel is required
    await expect(db.registerEffect("x", {})).rejects.toThrow(/kernel/);
    // @ts-expect-error multiplier is required
    await expect(db.addRule("e", "r", {})).rejects.toThrow(/multiplier/);
    expect(calls).toHaveLength(0);
  });
});

describe("events", () => {
  it("insert -> POST /events with one event, returns id", async () => {
    const { db, calls } = client([json({ ids: [42] })]);
    const id = await db.insert({
      entity: "me",
      time: new Date(13 * 3600 * 1000),
      type: "meal",
      features: new Set(["beans", "onion"]),
      weight: 1.5,
      attrs: { portion_size: 2, nested: { a: [1, "x"] } },
      externalId: "meal-1",
    });
    expect(id).toBe(42);
    expect(calls[0]!.url).toBe(`${BASE}/events`);
    expect(calls[0]!.body).toEqual({
      events: [
        {
          entity: "me",
          time: 46800,
          type: "meal",
          features: ["beans", "onion"],
          weight: 1.5,
          attrs: { portion_size: 2, nested: { a: [1, "x"] } },
          external_id: "meal-1",
        },
      ],
    });
  });

  it("insert accepts a single feature string and defaults features to []", async () => {
    const { db, calls } = client([json({ ids: [1] }), json({ ids: [2] })]);
    await db.insert({ entity: "e", time: 1.5, type: "t", features: "x" });
    await db.insert({ entity: "e", time: 2, type: "t" });
    expect((calls[0]!.body as { events: unknown[] }).events[0]).toEqual({
      entity: "e",
      time: 1.5,
      type: "t",
      features: ["x"],
    });
    expect((calls[1]!.body as { events: { features: string[] }[] }).events[0]!.features).toEqual(
      [],
    );
  });

  it("insert rejects bad times and missing fields locally", async () => {
    const { db, calls } = client([json({ ids: [1] })]);
    await expect(db.insert({ entity: "e", time: Number.NaN, type: "t" })).rejects.toThrow(
      /event.time/,
    );
    await expect(db.insert({ entity: "e", time: new Date("nope"), type: "t" })).rejects.toThrow(
      /event.time/,
    );
    await expect(db.insert({ entity: "", time: 1, type: "t" })).rejects.toThrow(/entity/);
    await expect(db.insert({ entity: "e", time: 1, type: "" })).rejects.toThrow(/type/);
    expect(calls).toHaveLength(0);
  });

  it("insert fails on a malformed response", async () => {
    const { db } = client([json({})]);
    await expect(db.insert({ entity: "e", time: 1, type: "t" })).rejects.toMatchObject({
      code: "invalid_response",
    });
  });
});

describe("batching", () => {
  const mk = (n: number) =>
    Array.from({ length: n }, (_, i) => ({ entity: "e", time: i, type: "t", features: ["f"] }));
  const echoIds = (call: { body: unknown }) => {
    const evs = (call.body as { events: { time: number }[] }).events;
    return json({ ids: evs.map((e) => e.time + 1000) });
  };

  it("splits into requests of at most 1000 events, ids in order", async () => {
    const { db, calls } = client([echoIds]);
    const ids = await db.insertMany(mk(2501));
    expect(calls.map((c) => (c.body as { events: unknown[] }).events.length)).toEqual([
      1000, 1000, 501,
    ]);
    expect(ids).toHaveLength(2501);
    expect(ids[0]).toBe(1000);
    expect(ids[2500]).toBe(3500);
  });

  it("exactly 1000 is one request; empty input makes none", async () => {
    const { db, calls } = client([echoIds]);
    await db.insertMany(mk(1000));
    expect(calls).toHaveLength(1);
    expect(await db.insertMany([])).toEqual([]);
    expect(calls).toHaveLength(1);
  });

  it("honours batchSize and accepts any iterable", async () => {
    const { db, calls } = client([echoIds]);
    function* gen() {
      yield* mk(5);
    }
    expect(await db.insertMany(gen(), { batchSize: 2 })).toEqual([1000, 1001, 1002, 1003, 1004]);
    expect(calls).toHaveLength(3);
  });

  it("rejects batchSize outside 1..1000", async () => {
    const { db } = client([echoIds]);
    await expect(db.insertMany(mk(1), { batchSize: 1001 })).rejects.toThrow(/batchSize/);
    await expect(db.insertMany(mk(1), { batchSize: 0 })).rejects.toThrow(/batchSize/);
  });

  it("validates every event before sending anything", async () => {
    const { db, calls } = client([echoIds]);
    const evs = [...mk(1500), { entity: "", time: 1, type: "t" }];
    await expect(db.insertMany(evs)).rejects.toThrow(/entity/);
    expect(calls).toHaveLength(0);
  });

  it("reports ids already committed when a later batch fails", async () => {
    const { db } = client(
      [echoIds, json({ error: { code: "storage_quota_exceeded", message: "full" } }, 413)],
      { retries: 0 },
    );
    const err = await db.insertMany(mk(1500)).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(HeapDBError);
    expect((err as HeapDBError).code).toBe("storage_quota_exceeded");
    expect((err as HeapDBError).status).toBe(413);
    expect((err as HeapDBError).insertedIds).toHaveLength(1000);
    expect((err as HeapDBError).message).toContain("after 1000 of 1500");
  });
});

describe("queries", () => {
  it("topCauses (single) -> POST /query/top-causes, decodes camelCase result", async () => {
    const { db, calls } = client([
      json({
        entity: "me",
        effect: "bloating",
        at: 54000,
        causes: [{ rank: 1, event: eventWire, score: 0.4528, explanation: explanationWire }],
        stats: {
          segments_in_window: 4,
          segments_scanned: 2,
          segments_pruned: 2,
          events_scanned: 7,
          union_fallbacks: 0,
        },
      }),
    ]);
    const res = await db.topCauses({
      entity: "me",
      effect: "bloating",
      at: 54000,
      k: 3,
      within: 43200,
      explain: true,
    });
    expect(calls[0]!.url).toBe(`${BASE}/query/top-causes`);
    expect(calls[0]!.body).toEqual({
      entity: "me",
      effect: "bloating",
      at: 54000,
      k: 3,
      within: 43200,
      explain: true,
    });
    expect(res.stats).toEqual({
      segmentsInWindow: 4,
      segmentsScanned: 2,
      segmentsPruned: 2,
      eventsScanned: 7,
      unionFallbacks: 0,
    });
    const c = res.causes[0]!;
    expect(c.rank).toBe(1);
    expect(c.event).toEqual({
      id: 23,
      entity: "me",
      time: 484200,
      type: "meal",
      features: ["coffee", "dairy"],
      weight: 0.8,
      attrs: { portion_size: 2 },
    });
    expect(c.explanation).toMatchObject({
      dt: 5400,
      kernelValue: 0.845,
      linkStrength: 0.67,
      ruleMultiplier: 0.3,
      prospective: false,
      rules: [{ name: "small", multiplier: 0.3 }],
    });
    expect(c.explanation!.jointCauses[0]).toEqual({
      features: ["coffee", "dairy"],
      strength: 0.83,
      joint: { priorAlpha: 1, priorBeta: 4, pos: 4.8, neg: 0, globalPos: 1, globalNeg: 0.2 },
      inseparable: true,
      name: "coffee+dairy",
    });
    expect(res.entity).toBe("me");
  });

  it("decodes the logistic explanation fields (base odds x multipliers)", async () => {
    const logistic = {
      ...explanationWire,
      link_strength: 0.6,
      features: [
        {
          feature: "dairy",
          strength: 0.5,
          alpha: 2,
          beta: 2,
          log_odds: 0.9,
          odds_multiplier: Math.exp(0.9),
        },
      ],
      combine: "logistic",
      base_log_odds: -0.5,
      base_odds: Math.exp(-0.5),
      log_odds: 0.4,
      odds: Math.exp(0.4),
    };
    const { db } = client([
      json({
        causes: [{ rank: 1, event: eventWire, score: 0.4, explanation: logistic }],
        stats: {},
      }),
    ]);
    const ex = (await db.topCauses({ entity: "me", effect: "bloating", at: 54000, explain: true }))
      .causes[0]!.explanation!;
    expect(ex).toMatchObject({
      combine: "logistic",
      baseLogOdds: -0.5,
      logOdds: 0.4,
      linkStrength: 0.6,
    });
    expect(ex.odds).toBeCloseTo(Math.exp(0.4));
    expect(ex.baseOdds).toBeCloseTo(Math.exp(-0.5));
    expect(ex.features[0]).toMatchObject({ feature: "dairy", logOdds: 0.9 });
    expect(ex.features[0]!.oddsMultiplier).toBeCloseTo(Math.exp(0.9));
  });

  it("noisy-OR explanations carry no logistic fields", async () => {
    const { db } = client([
      json({
        causes: [{ rank: 1, event: eventWire, score: 0.4528, explanation: explanationWire }],
        stats: {},
      }),
    ]);
    const ex = (await db.topCauses({ entity: "me", effect: "bloating", at: 54000, explain: true }))
      .causes[0]!.explanation!;
    expect(ex.combine).toBeUndefined();
    expect(ex.baseLogOdds).toBeUndefined();
    expect(ex.features[0]!.logOdds).toBeUndefined();
  });

  it("topCauses (group) sends entities, tolerates a minimal response", async () => {
    const { db, calls } = client([json({ causes: [{ event: { id: 1 }, score: 0.1 }], stats: {} })]);
    const res = await db.topCauses({ entities: ["svc-a", "svc-b"], effect: "incident", at: 10 });
    expect(calls[0]!.body).toEqual({ entities: ["svc-a", "svc-b"], effect: "incident", at: 10 });
    expect(res.causes[0]!.rank).toBe(1);
    expect(res.causes[0]!.explanation).toBeNull();
    expect(res.stats.eventsScanned).toBe(0);
  });

  it("topCauses requires exactly one of entity / entities", async () => {
    const { db } = client([json({})]);
    const both = { entity: "a", entities: ["b"], effect: "e", at: 1 };
    // @ts-expect-error both given
    await expect(db.topCauses(both)).rejects.toThrow(/exactly one/);
    // @ts-expect-error neither given
    await expect(db.topCauses({ effect: "e", at: 1 })).rejects.toThrow(/exactly one/);
    await expect(db.topCauses({ entities: [], effect: "e", at: 1 })).rejects.toThrow(/empty/);
  });

  it("featureRisk -> POST /query/risk", async () => {
    const { db, calls } = client([
      json({
        effect: "incident",
        entity: null,
        features: ["migration"],
        type: "pr",
        score: 0.41,
        explanation: { ...explanationWire, prospective: true },
      }),
    ]);
    const r = await db.featureRisk({
      effect: "incident",
      features: ["migration", "migration"],
      type: "pr",
      weight: null,
      attrs: { lines_changed: 400 },
      explain: true,
    });
    expect(calls[0]!.url).toBe(`${BASE}/query/risk`);
    expect(calls[0]!.body).toEqual({
      effect: "incident",
      features: ["migration"],
      weight: null,
      type: "pr",
      attrs: { lines_changed: 400 },
      explain: true,
    });
    expect(r.score).toBe(0.41);
    expect(r.explanation?.prospective).toBe(true);
    expect(r.entity).toBeNull();
  });

  it("featureRisk tolerates a bare number", async () => {
    const { db } = client([json(0.25)]);
    expect(await db.featureRisk({ effect: "e", features: "f" })).toEqual({
      score: 0.25,
      explanation: null,
    });
  });

  it("rankCandidates -> POST /query/risk with candidates and k", async () => {
    const ranked = [
      {
        rank: 1,
        candidate: {
          id: "pr-7",
          features: ["migration"],
          weight: 2.1,
          type: "pr",
          attrs: null,
          entity: null,
        },
        score: 0.6,
        entity: null,
        explanation: null,
      },
    ];
    const { db, calls } = client([json({ candidates: ranked }), json(ranked)]);
    const res = await db.rankCandidates({
      effect: "incident",
      candidates: [
        { id: "pr-7", features: ["migration"], type: "pr", attrs: { lines_changed: 10 } },
      ],
      k: 5,
    });
    expect(calls[0]!.body).toEqual({
      effect: "incident",
      candidates: [
        { id: "pr-7", features: ["migration"], type: "pr", attrs: { lines_changed: 10 } },
      ],
      k: 5,
    });
    expect(res[0]).toEqual({
      rank: 1,
      candidate: {
        id: "pr-7",
        features: ["migration"],
        weight: 2.1,
        type: "pr",
        attrs: null,
        entity: null,
      },
      score: 0.6,
      entity: null,
      explanation: null,
    });
    // default k = 10; bare-array response also accepted
    const res2 = await db.rankCandidates({ effect: "incident", candidates: [{ id: 1 }] });
    expect((calls[1]!.body as { k: number }).k).toBe(10);
    expect(res2).toHaveLength(1);
  });
});

describe("learning", () => {
  it("feedback -> POST /feedback", async () => {
    const { db, calls } = client([json({ confirmed: 22, rejected: [23], credit: { beans: 0.7 } })]);
    const r = await db.feedback({
      entity: "me",
      effect: "bloating",
      at: 54000,
      confirmedEventId: 22,
      rejectedIds: [23],
      credit: "uniform",
    });
    expect(calls[0]!.url).toBe(`${BASE}/feedback`);
    expect(calls[0]!.body).toEqual({
      entity: "me",
      effect: "bloating",
      at: 54000,
      confirmed_event_id: 22,
      rejected_ids: [23],
      credit: "uniform",
    });
    expect(r).toEqual({ confirmed: 22, rejected: [23], credit: { beans: 0.7 } });
  });

  it("observe -> POST /observe", async () => {
    const { db, calls } = client([
      json({
        entity: "me",
        effect: "bloating",
        at: 54000,
        leak: 0.05,
        leak_responsibility: 0.1,
        credits: [{ event_id: 22, score: 0.3, responsibility: 0.9, features: { beans: 0.6 } }],
        total_credit: 0.9,
      }),
    ]);
    const r = await db.observe({ entity: "me", effect: "bloating", at: 54000, leak: 0.05 });
    expect(calls[0]!.url).toBe(`${BASE}/observe`);
    expect(calls[0]!.body).toEqual({ entity: "me", effect: "bloating", at: 54000, leak: 0.05 });
    expect(r.leakResponsibility).toBe(0.1);
    expect(r.credits[0]).toEqual({
      eventId: 22,
      score: 0.3,
      responsibility: 0.9,
      features: { beans: 0.6 },
    });
    expect(r.totalCredit).toBe(0.9);
  });

  it("settle -> POST /settle; accepts {settled: n} or {settled: report}", async () => {
    const { db, calls } = client([
      json({ settled: 12 }),
      json({
        settled: {
          until: 100,
          events_settled: 5,
          events_rejected: 3,
          rejected_weight: 2.5,
          skipped_credited: 1,
          skipped_observed: 1,
          per_effect: { bloating: 5 },
        },
      }),
    ]);
    expect(await db.settle({ until: 100, effects: "bloating", entity: "me" })).toEqual({
      eventsSettled: 12,
    });
    expect(calls[0]!.url).toBe(`${BASE}/settle`);
    expect(calls[0]!.body).toEqual({ until: 100, effects: ["bloating"], entity: "me" });
    expect(await db.settle({ until: 100 })).toEqual({
      until: 100,
      eventsSettled: 5,
      eventsRejected: 3,
      rejectedWeight: 2.5,
      skippedCredited: 1,
      skippedObserved: 1,
      perEffect: { bloating: 5 },
    });
    expect(calls[1]!.body).toEqual({ until: 100 });
  });
});

describe("reports and admin", () => {
  it("linkReport -> GET /links with query params", async () => {
    const row = {
      feature: "beans",
      strength: 0.7,
      alpha: 4,
      beta: 1,
      n: 3,
      lo: 0.4,
      hi: 0.9,
      lift: 2.1,
    };
    const { db, calls } = client([
      json({ effect: "bloating", entity: "me", baseline: 0.33, level: 0.9, rows: [row] }),
      json([row]),
    ]);
    const r = await db.linkReport({ effect: "bloating", entity: "me", minSupport: 2 });
    expect(calls[0]!.method).toBe("GET");
    expect(calls[0]!.url).toBe(`${BASE}/links?effect=bloating&entity=me&min_support=2`);
    expect(r).toEqual({
      effect: "bloating",
      entity: "me",
      baseline: 0.33,
      level: 0.9,
      rows: [row],
    });
    const { db: db2 } = client([json({ effect: "x", rows: [{ ...row, log_odds: -1.25 }] })]);
    expect((await db2.linkReport({ effect: "x" })).rows[0]).toEqual({ ...row, logOdds: -1.25 });
    const r2 = await db.linkReport({ effect: "a b&c" });
    expect(calls[1]!.url).toBe(`${BASE}/links?effect=a%20b%26c`);
    expect(r2.rows).toEqual([row]);
    expect(r2.effect).toBe("a b&c");
  });

  it("calibrationReport -> GET /calibration", async () => {
    const { db, calls } = client([
      json({
        effect: "bloating",
        n: 4,
        brier: 0.12,
        bins: [{ lo: 0, hi: 0.1, count: 2, mean_predicted: 0.05, observed_rate: null }],
      }),
    ]);
    const r = await db.calibrationReport({ effect: "bloating", bins: 10 });
    expect(calls[0]!.url).toBe(`${BASE}/calibration?effect=bloating&bins=10`);
    expect(r.bins[0]).toEqual({
      lo: 0,
      hi: 0.1,
      count: 2,
      meanPredicted: 0.05,
      observedRate: null,
    });
    expect(r.brier).toBe(0.12);
  });

  it("forgetEntity -> DELETE /entities/{entity}?scrub_global=", async () => {
    const { db, calls } = client([
      json({ events: 12, links: 3, joint_sets: 1 }),
      json({ events: 0 }),
    ]);
    const r = await db.forgetEntity("user/42 ü", { scrubGlobal: true });
    expect(calls[0]!.method).toBe("DELETE");
    expect(calls[0]!.url).toBe(`${BASE}/entities/user%2F42%20%C3%BC?scrub_global=true`);
    expect(r).toEqual({ events: 12, links: 3, jointSets: 1 });
    await db.forgetEntity("x");
    expect(calls[1]!.url).toBe(`${BASE}/entities/x`);
  });

  it("hql -> POST /hql, result passed through", async () => {
    const { db, calls } = client([json({ causes: [] })]);
    const r = await db.hql<{ causes: unknown[] }>("TOP 3 CAUSES OF bloating FOR me AT now");
    expect(calls[0]!.url).toBe(`${BASE}/hql`);
    expect(calls[0]!.body).toEqual({ query: "TOP 3 CAUSES OF bloating FOR me AT now" });
    expect(r).toEqual({ causes: [] });
  });

  it("info -> GET base", async () => {
    const { db, calls } = client([
      json({
        database_id: "db_x",
        plan: "free",
        usage: { requests_this_period: 10, events_stored: 5 },
        limits: { max_requests_per_month: 100000, max_events: null },
      }),
    ]);
    const r = await db.info();
    expect(calls[0]!.method).toBe("GET");
    expect(calls[0]!.url).toBe(BASE);
    expect(r).toEqual({
      databaseId: "db_x",
      plan: "free",
      usage: { requestsThisPeriod: 10, eventsStored: 5 },
      limits: { maxRequestsPerMonth: 100000, maxEvents: null },
    });
  });

  it("empty 2xx bodies are fine", async () => {
    const { db } = client([new Response(null, { status: 204 })]);
    await expect(db.registerEventType("t")).resolves.toBeUndefined();
  });
});

describe("package", () => {
  it("VERSION matches package.json", async () => {
    const { readFile } = await import("node:fs/promises");
    const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
    expect(VERSION).toBe(pkg.version);
  });
});
