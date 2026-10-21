-- HeapDB keyed heaps for PostgreSQL (persist_to = "postgres://...").
--
-- Installs schema `heapdb`: tables heapdb.heaps and heapdb.items (stored
-- generated `key`, index on (heap, key DESC, id)) and the functions
--
--   heapdb.create_heap(name, half_life, landmark, meta, now)  -> heap row
--   heapdb.drop_heap(name)                                    -> boolean
--   heapdb.heap_info(name) / heapdb.list_heaps()              -> heap rows
--   heapdb.push(heap, id, weight, t, attrs)                   -> item row
--   heapdb.set_base(heap, id, base)                           -> item row
--   heapdb.set_attrs(heap, id, attrs, merge)                  -> item row
--   heapdb.remove(heap, id)                                   -> boolean
--   heapdb.get(heap, id, now)                                 -> ranked row (rank 0)
--   heapdb.top_k(heap, k, now, filter)                        -> ranked rows
--   heapdb.pop(heap, now, filter)                             -> ranked row (FOR UPDATE SKIP LOCKED)
--   heapdb.rebase(heap, now, half_life)                       -> heap row
--   heapdb.matches(attrs, filter)                             -> boolean
--
-- Semantics and numerics are those of the Rust engine (crates/heapdb,
-- src/heaps): see docs/heaps.md. The conformance suite
-- (crates/heapdb-postgres/tests/conformance.rs) checks this file against
-- the in-memory engine.
--
-- Every function is SECURITY INVOKER, so row-level security policies on
-- heapdb.items apply to the caller. Callers need USAGE on the schema,
-- SELECT/INSERT/UPDATE/DELETE on heapdb.items and SELECT/UPDATE on
-- heapdb.heaps (plus INSERT/DELETE on heapdb.heaps for create/drop).
--
-- Errors: SQLSTATE HD400 (invalid argument), HD404 (unknown heap),
-- HD409 (create_heap with a different spec).
--
-- Times are Unix seconds (double precision); `now` defaults to the
-- transaction time. The file is idempotent: running it again upgrades the
-- installed version in place (heapdb.schema_version).

BEGIN;

-- Serialise concurrent installs.
SELECT pg_advisory_xact_lock(7268946137285163008);

CREATE SCHEMA IF NOT EXISTS heapdb;

CREATE TABLE IF NOT EXISTS heapdb.schema_version (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    version text NOT NULL,
    installed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS heapdb.heaps (
    name text COLLATE "C" PRIMARY KEY CHECK (octet_length(name) BETWEEN 1 AND 200),
    half_life double precision NOT NULL CHECK (half_life > 0 AND half_life < 'Infinity'),
    landmark double precision NOT NULL CHECK (landmark > '-Infinity' AND landmark < 'Infinity'),
    meta jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(meta) = 'object')
);

CREATE TABLE IF NOT EXISTS heapdb.items (
    heap text COLLATE "C" NOT NULL REFERENCES heapdb.heaps (name) ON DELETE CASCADE,
    id text COLLATE "C" NOT NULL CHECK (octet_length(id) BETWEEN 1 AND 200),
    base double precision NOT NULL DEFAULT 1 CHECK (base > 0 AND base < 'Infinity'),
    -- ln sum_i w_i exp(lambda (t_i - landmark)); NULL until the first push.
    activity double precision CHECK (activity > '-Infinity' AND activity < 'Infinity'),
    -- ln(base) + activity; NULL (not ranked) while activity is NULL. The
    -- + 0.0 turns -0.0 into 0.0, as in Rust (heapdb::heaps::item_key).
    key double precision GENERATED ALWAYS AS (ln(base) + activity + 0.0::double precision) STORED,
    attrs jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(attrs) = 'object'),
    version bigint NOT NULL DEFAULT 1,
    PRIMARY KEY (heap, id)
);

CREATE INDEX IF NOT EXISTS items_rank ON heapdb.items (heap, key DESC, id) WHERE key IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Numerics (mirror heapdb::heaps::{lambda, logaddexp, decay_exp})
-- ---------------------------------------------------------------------------

-- plpgsql, not SQL: an inlined SQL CASE could be constant-folded and raise
-- Postgres' exp() underflow error in a branch that is never taken.

CREATE OR REPLACE FUNCTION heapdb.lambda(half_life double precision)
RETURNS double precision LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE SECURITY INVOKER AS $$
    SELECT 0.6931471805599453::double precision / half_life
$$;

-- ln(exp(a) + exp(b)) = hi + ln(1 + exp(-(hi - lo))); above a difference of
-- 40 the second term is exactly 0 in double precision.
CREATE OR REPLACE FUNCTION heapdb.logaddexp(a double precision, b double precision)
RETURNS double precision LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE SECURITY INVOKER AS $$
DECLARE
    hi double precision := greatest(a, b);
    lo double precision := least(a, b);
    d double precision;
BEGIN
    IF a IS NULL THEN RETURN b; END IF;
    IF b IS NULL THEN RETURN a; END IF;
    d := hi - lo;
    IF d > 40 THEN
        RETURN hi;
    END IF;
    RETURN hi + ln(1::double precision + exp(-d));
END
$$;

-- exp(x), with x < -745 giving 0 and x > ln(DBL_MAX) giving Infinity
-- (Postgres' exp() raises underflow / overflow errors there).
CREATE OR REPLACE FUNCTION heapdb.decay_exp(x double precision)
RETURNS double precision LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE SECURITY INVOKER AS $$
BEGIN
    IF x < -745 THEN RETURN 0; END IF;
    IF x > 709.782712893384 THEN RETURN 'Infinity'; END IF;
    RETURN exp(x);
END
$$;

-- ---------------------------------------------------------------------------
-- Validation helpers
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION heapdb._invalid(msg text)
RETURNS void LANGUAGE plpgsql SECURITY INVOKER AS $$
BEGIN
    RAISE EXCEPTION USING ERRCODE = 'HD400', MESSAGE = msg;
END
$$;

CREATE OR REPLACE FUNCTION heapdb._check_name(what text, s text)
RETURNS void LANGUAGE plpgsql IMMUTABLE SECURITY INVOKER AS $$
BEGIN
    IF s IS NULL OR octet_length(s) NOT BETWEEN 1 AND 200 THEN
        PERFORM heapdb._invalid(format('%s must be 1-200 UTF-8 bytes, got %s bytes', what, coalesce(octet_length(s), 0)));
    END IF;
END
$$;

CREATE OR REPLACE FUNCTION heapdb._check_positive(what text, x double precision)
RETURNS void LANGUAGE plpgsql IMMUTABLE SECURITY INVOKER AS $$
BEGIN
    -- NaN > 0 is true in Postgres, so test it explicitly.
    IF x IS NULL OR x = 'NaN' OR x = 'Infinity' OR x <= 0 THEN
        PERFORM heapdb._invalid(format('%s must be finite and > 0, got %s', what, coalesce(x::text, 'NULL')));
    END IF;
END
$$;

CREATE OR REPLACE FUNCTION heapdb._check_finite(what text, x double precision)
RETURNS void LANGUAGE plpgsql IMMUTABLE SECURITY INVOKER AS $$
BEGIN
    IF x IS NULL OR x = 'NaN' OR x = 'Infinity' OR x = '-Infinity' THEN
        PERFORM heapdb._invalid(format('%s must be finite, got %s', what, coalesce(x::text, 'NULL')));
    END IF;
END
$$;

CREATE OR REPLACE FUNCTION heapdb._check_object(what text, v jsonb)
RETURNS void LANGUAGE plpgsql IMMUTABLE SECURITY INVOKER AS $$
BEGIN
    IF v IS NOT NULL AND jsonb_typeof(v) <> 'object' THEN
        PERFORM heapdb._invalid(format('%s must be a JSON object, got %s', what, jsonb_typeof(v)));
    END IF;
END
$$;

-- The heap row, locked FOR SHARE (FOR UPDATE with exclusive = true), or
-- error HD404.
CREATE OR REPLACE FUNCTION heapdb._heap(p_name text, exclusive boolean DEFAULT false)
RETURNS heapdb.heaps LANGUAGE plpgsql SECURITY INVOKER AS $$
DECLARE
    h heapdb.heaps;
BEGIN
    IF exclusive THEN
        SELECT * INTO h FROM heapdb.heaps WHERE name = p_name FOR UPDATE;
    ELSE
        SELECT * INTO h FROM heapdb.heaps WHERE name = p_name FOR SHARE;
    END IF;
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = 'HD404',
            MESSAGE = format('unknown heap %L; call create_heap first', p_name);
    END IF;
    RETURN h;
END
$$;

-- Validate a filter tree (raises HD400 naming the offending node).
CREATE OR REPLACE FUNCTION heapdb.check_filter(p_filter jsonb, p_path text DEFAULT 'filter', p_depth int DEFAULT 0)
RETURNS void LANGUAGE plpgsql IMMUTABLE SECURITY INVOKER AS $$
DECLARE
    comb text;
    c jsonb;
    i int := 0;
    op text;
    v jsonb;
    k text;
BEGIN
    IF p_filter IS NULL THEN RETURN; END IF;
    IF p_depth > 32 THEN
        PERFORM heapdb._invalid(p_path || ': nested deeper than 32 levels');
    END IF;
    IF jsonb_typeof(p_filter) <> 'object' THEN
        PERFORM heapdb._invalid(format('%s: expected an object, got %s', p_path, jsonb_typeof(p_filter)));
    END IF;
    FOREACH comb IN ARRAY ARRAY['all', 'any', 'not'] LOOP
        IF p_filter ? comb THEN
            IF (SELECT count(*) FROM jsonb_object_keys(p_filter)) <> 1 THEN
                PERFORM heapdb._invalid(format('%s: "%s" must be the only key of its node', p_path, comb));
            END IF;
            IF comb = 'not' THEN
                PERFORM heapdb.check_filter(p_filter -> 'not', p_path || '.not', p_depth + 1);
                RETURN;
            END IF;
            IF jsonb_typeof(p_filter -> comb) <> 'array' THEN
                PERFORM heapdb._invalid(format('%s: "%s" takes an array of filters', p_path, comb));
            END IF;
            FOR c IN SELECT e FROM jsonb_array_elements(p_filter -> comb) e LOOP
                PERFORM heapdb.check_filter(c, format('%s.%s[%s]', p_path, comb, i), p_depth + 1);
                i := i + 1;
            END LOOP;
            RETURN;
        END IF;
    END LOOP;
    FOR k IN SELECT jsonb_object_keys(p_filter) LOOP
        IF k NOT IN ('attr', 'op', 'value') THEN
            PERFORM heapdb._invalid(format('%s: unknown key %L (a leaf has "attr", "op" and "value")', p_path, k));
        END IF;
    END LOOP;
    IF jsonb_typeof(p_filter -> 'attr') IS DISTINCT FROM 'string' OR p_filter ->> 'attr' = '' THEN
        PERFORM heapdb._invalid(p_path || ': "attr" must be a non-empty string');
    END IF;
    IF jsonb_typeof(p_filter -> 'op') IS DISTINCT FROM 'string' THEN
        PERFORM heapdb._invalid(p_path || ': missing "op"');
    END IF;
    op := p_filter ->> 'op';
    v := p_filter -> 'value';
    IF op IN ('exists', 'missing') THEN
        IF v IS NOT NULL THEN
            PERFORM heapdb._invalid(format('%s: op %L takes no "value"', p_path, op));
        END IF;
    ELSIF op IN ('in', 'not_in', 'contains_all', 'subset_of') THEN
        IF jsonb_typeof(v) IS DISTINCT FROM 'array' THEN
            PERFORM heapdb._invalid(format('%s: op %L needs an array value', p_path, op));
        END IF;
    ELSIF op IN ('lt', 'lte', 'gt', 'gte') THEN
        IF jsonb_typeof(v) IS NULL OR jsonb_typeof(v) NOT IN ('number', 'string') THEN
            PERFORM heapdb._invalid(format('%s: op %L needs a number or a string', p_path, op));
        END IF;
    ELSIF op IN ('eq', 'ne', 'contains') THEN
        IF v IS NULL OR jsonb_typeof(v) = 'null' THEN
            PERFORM heapdb._invalid(format('%s: op %L needs a non-null value', p_path, op));
        END IF;
    ELSE
        PERFORM heapdb._invalid(format(
            '%s: unknown op %L (expected one of eq, ne, lt, lte, gt, gte, in, not_in, contains, contains_all, subset_of, exists, missing)',
            p_path, op));
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- Filters (mirror heapdb::heaps::Filter::matches)
-- ---------------------------------------------------------------------------

-- JSON equality with top-level numbers compared as doubles (nested values
-- use jsonb equality, which compares numbers numerically).
CREATE OR REPLACE FUNCTION heapdb._json_eq(a jsonb, b jsonb)
RETURNS boolean LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE SECURITY INVOKER AS $$
BEGIN
    IF jsonb_typeof(a) = 'number' AND jsonb_typeof(b) = 'number' THEN
        RETURN a::double precision = b::double precision;
    END IF;
    RETURN a = b;
END
$$;

CREATE OR REPLACE FUNCTION heapdb._json_in(x jsonb, list jsonb)
RETURNS boolean LANGUAGE sql IMMUTABLE PARALLEL SAFE SECURITY INVOKER AS $$
    SELECT EXISTS (SELECT 1 FROM jsonb_array_elements(list) e WHERE heapdb._json_eq(x, e))
$$;

-- Both numbers (as doubles) or both strings (byte-wise); otherwise NULL.
CREATE OR REPLACE FUNCTION heapdb._json_cmp(a jsonb, b jsonb)
RETURNS int LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE SECURITY INVOKER AS $$
DECLARE
    x double precision;
    y double precision;
    s text COLLATE "C";
    t text COLLATE "C";
BEGIN
    IF jsonb_typeof(a) = 'number' AND jsonb_typeof(b) = 'number' THEN
        x := a::double precision;
        y := b::double precision;
        RETURN CASE WHEN x < y THEN -1 WHEN x > y THEN 1 ELSE 0 END;
    END IF;
    IF jsonb_typeof(a) = 'string' AND jsonb_typeof(b) = 'string' THEN
        s := a #>> '{}';
        t := b #>> '{}';
        RETURN CASE WHEN s < t THEN -1 WHEN s > t THEN 1 ELSE 0 END;
    END IF;
    RETURN NULL;
END
$$;

-- Does `attrs` match `filter` (docs/heaps.md, "Filters")? A NULL filter
-- matches everything. The filter is assumed valid (see check_filter).
CREATE OR REPLACE FUNCTION heapdb.matches(p_attrs jsonb, p_filter jsonb)
RETURNS boolean LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE SECURITY INVOKER AS $$
DECLARE
    c jsonb;
    op text;
    lhs jsonb;
    rhs jsonb;
    cmp int;
BEGIN
    IF p_filter IS NULL THEN RETURN true; END IF;
    IF p_filter ? 'all' THEN
        FOR c IN SELECT e FROM jsonb_array_elements(p_filter -> 'all') e LOOP
            IF NOT heapdb.matches(p_attrs, c) THEN RETURN false; END IF;
        END LOOP;
        RETURN true;
    END IF;
    IF p_filter ? 'any' THEN
        FOR c IN SELECT e FROM jsonb_array_elements(p_filter -> 'any') e LOOP
            IF heapdb.matches(p_attrs, c) THEN RETURN true; END IF;
        END LOOP;
        RETURN false;
    END IF;
    IF p_filter ? 'not' THEN
        RETURN NOT heapdb.matches(p_attrs, p_filter -> 'not');
    END IF;

    op := p_filter ->> 'op';
    lhs := p_attrs -> (p_filter ->> 'attr');
    IF jsonb_typeof(lhs) = 'null' THEN
        lhs := NULL;  -- null counts as missing
    END IF;
    rhs := p_filter -> 'value';
    IF lhs IS NULL THEN
        RETURN op IN ('ne', 'not_in', 'missing', 'subset_of');
    END IF;
    CASE op
        WHEN 'exists' THEN RETURN true;
        WHEN 'missing' THEN RETURN false;
        WHEN 'eq' THEN RETURN heapdb._json_eq(lhs, rhs);
        WHEN 'ne' THEN RETURN NOT heapdb._json_eq(lhs, rhs);
        WHEN 'lt', 'lte', 'gt', 'gte' THEN
            cmp := heapdb._json_cmp(lhs, rhs);
            IF cmp IS NULL THEN RETURN false; END IF;  -- mixed types never match
            RETURN CASE op
                WHEN 'lt' THEN cmp < 0
                WHEN 'lte' THEN cmp <= 0
                WHEN 'gt' THEN cmp > 0
                ELSE cmp >= 0
            END;
        WHEN 'in' THEN RETURN heapdb._json_in(lhs, rhs);
        WHEN 'not_in' THEN RETURN NOT heapdb._json_in(lhs, rhs);
        WHEN 'contains' THEN
            IF jsonb_typeof(lhs) = 'array' THEN
                RETURN heapdb._json_in(rhs, lhs);
            ELSIF jsonb_typeof(lhs) = 'string' AND jsonb_typeof(rhs) = 'string' THEN
                RETURN strpos(lhs #>> '{}', rhs #>> '{}') > 0;
            END IF;
            RETURN false;
        WHEN 'contains_all' THEN
            IF jsonb_typeof(lhs) <> 'array' THEN RETURN false; END IF;
            RETURN NOT EXISTS (SELECT 1 FROM jsonb_array_elements(rhs) v WHERE NOT heapdb._json_in(v, lhs));
        WHEN 'subset_of' THEN
            IF jsonb_typeof(lhs) <> 'array' THEN RETURN false; END IF;
            RETURN NOT EXISTS (SELECT 1 FROM jsonb_array_elements(lhs) v WHERE NOT heapdb._json_in(v, rhs));
        ELSE
            PERFORM heapdb.check_filter(p_filter);
            RETURN false;
    END CASE;
END
$$;

-- ---------------------------------------------------------------------------
-- Heaps
-- ---------------------------------------------------------------------------

-- Signatures changed between versions are dropped here before being
-- recreated (CREATE OR REPLACE cannot change a return type).

CREATE OR REPLACE FUNCTION heapdb._info(h heapdb.heaps)
RETURNS TABLE (name text, half_life double precision, landmark double precision, meta jsonb, size bigint)
LANGUAGE sql STABLE SECURITY INVOKER AS $$
    SELECT h.name, h.half_life, h.landmark, h.meta,
           (SELECT count(*) FROM heapdb.items i WHERE i.heap = h.name)
$$;

CREATE OR REPLACE FUNCTION heapdb.create_heap(
    p_name text,
    p_half_life double precision,
    p_landmark double precision DEFAULT NULL,
    p_meta jsonb DEFAULT NULL,
    p_now double precision DEFAULT extract(epoch FROM now())::double precision
)
RETURNS TABLE (name text, half_life double precision, landmark double precision, meta jsonb, size bigint)
LANGUAGE plpgsql SECURITY INVOKER AS $$
DECLARE
    h heapdb.heaps;
    diffs text[] := '{}';
BEGIN
    PERFORM heapdb._check_name('heap name', p_name);
    PERFORM heapdb._check_positive('half_life', p_half_life);
    IF p_landmark IS NOT NULL THEN
        PERFORM heapdb._check_finite('landmark', p_landmark);
    END IF;
    PERFORM heapdb._check_object('meta', p_meta);
    IF p_landmark IS NULL THEN
        PERFORM heapdb._check_finite('now', p_now);
    END IF;
    INSERT INTO heapdb.heaps AS t (name, half_life, landmark, meta)
    VALUES (p_name, p_half_life, coalesce(p_landmark, p_now), coalesce(p_meta, '{}'))
    ON CONFLICT ON CONSTRAINT heaps_pkey DO NOTHING;
    SELECT * INTO STRICT h FROM heapdb.heaps t WHERE t.name = p_name;
    IF h.half_life <> p_half_life THEN
        diffs := diffs || format('half_life %s (requested %s)', h.half_life, p_half_life);
    END IF;
    IF p_landmark IS NOT NULL AND h.landmark <> p_landmark THEN
        diffs := diffs || format('landmark %s (requested %s)', h.landmark, p_landmark);
    END IF;
    IF p_meta IS NOT NULL AND h.meta <> p_meta THEN
        diffs := diffs || 'different meta'::text;
    END IF;
    IF cardinality(diffs) > 0 THEN
        RAISE EXCEPTION USING ERRCODE = 'HD409',
            MESSAGE = format('heap %L already exists with %s', p_name, array_to_string(diffs, ', '));
    END IF;
    RETURN QUERY SELECT * FROM heapdb._info(h);
END
$$;

CREATE OR REPLACE FUNCTION heapdb.drop_heap(p_name text)
RETURNS boolean LANGUAGE plpgsql SECURITY INVOKER AS $$
BEGIN
    DELETE FROM heapdb.heaps t WHERE t.name = p_name;
    RETURN FOUND;
END
$$;

CREATE OR REPLACE FUNCTION heapdb.heap_info(p_name text)
RETURNS TABLE (name text, half_life double precision, landmark double precision, meta jsonb, size bigint)
LANGUAGE sql STABLE SECURITY INVOKER AS $$
    SELECT i.* FROM heapdb.heaps h, heapdb._info(h) i WHERE h.name = p_name
$$;

CREATE OR REPLACE FUNCTION heapdb.list_heaps()
RETURNS TABLE (name text, half_life double precision, landmark double precision, meta jsonb, size bigint)
LANGUAGE sql STABLE SECURITY INVOKER AS $$
    SELECT h.name, h.half_life, h.landmark, h.meta, count(i.id)
    FROM heapdb.heaps h LEFT JOIN heapdb.items i ON i.heap = h.name
    GROUP BY h.name, h.half_life, h.landmark, h.meta
    ORDER BY h.name
$$;

-- ---------------------------------------------------------------------------
-- Items
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION heapdb.push(
    p_heap text,
    p_id text,
    p_weight double precision DEFAULT 1,
    p_t double precision DEFAULT extract(epoch FROM now())::double precision,
    p_attrs jsonb DEFAULT NULL
)
RETURNS TABLE (id text, base double precision, activity double precision, key double precision, attrs jsonb, version bigint)
LANGUAGE plpgsql SECURITY INVOKER AS $$
DECLARE
    h heapdb.heaps;
    x double precision;
BEGIN
    PERFORM heapdb._check_name('item id', p_id);
    PERFORM heapdb._check_positive('weight', p_weight);
    PERFORM heapdb._check_finite('time', p_t);
    PERFORM heapdb._check_object('attrs', p_attrs);
    h := heapdb._heap(p_heap);
    BEGIN
        x := ln(p_weight) + heapdb.lambda(h.half_life) * (p_t - h.landmark);
    EXCEPTION WHEN numeric_value_out_of_range THEN
        x := 'Infinity';
    END;
    IF x = 'Infinity' OR x = '-Infinity' THEN
        PERFORM heapdb._invalid('the push time is too far from the heap''s landmark (the key is not finite); rebase the heap');
    END IF;
    BEGIN
        RETURN QUERY
        INSERT INTO heapdb.items AS t (heap, id, base, activity, attrs, version)
        VALUES (p_heap, p_id, 1, x, coalesce(p_attrs, '{}'), 1)
        ON CONFLICT ON CONSTRAINT items_pkey DO UPDATE
            SET activity = heapdb.logaddexp(t.activity, excluded.activity),
                attrs = coalesce(p_attrs, t.attrs),
                version = t.version + 1
        RETURNING t.id, t.base, t.activity, t.key, t.attrs, t.version;
    EXCEPTION WHEN numeric_value_out_of_range OR check_violation THEN
        PERFORM heapdb._invalid('the push time is too far from the heap''s landmark (the key is not finite); rebase the heap');
    END;
END
$$;

CREATE OR REPLACE FUNCTION heapdb.set_base(p_heap text, p_id text, p_base double precision)
RETURNS TABLE (id text, base double precision, activity double precision, key double precision, attrs jsonb, version bigint)
LANGUAGE plpgsql SECURITY INVOKER AS $$
BEGIN
    PERFORM heapdb._check_name('item id', p_id);
    PERFORM heapdb._check_positive('base', p_base);
    PERFORM heapdb._heap(p_heap);
    BEGIN
        RETURN QUERY
        INSERT INTO heapdb.items AS t (heap, id, base, activity, attrs, version)
        VALUES (p_heap, p_id, p_base, NULL, '{}', 1)
        ON CONFLICT ON CONSTRAINT items_pkey DO UPDATE
            SET base = excluded.base, version = t.version + 1
        RETURNING t.id, t.base, t.activity, t.key, t.attrs, t.version;
    EXCEPTION WHEN numeric_value_out_of_range THEN
        PERFORM heapdb._invalid('the push time is too far from the heap''s landmark (the key is not finite); rebase the heap');
    END;
END
$$;

CREATE OR REPLACE FUNCTION heapdb.set_attrs(p_heap text, p_id text, p_attrs jsonb, p_merge boolean DEFAULT false)
RETURNS TABLE (id text, base double precision, activity double precision, key double precision, attrs jsonb, version bigint)
LANGUAGE plpgsql SECURITY INVOKER AS $$
BEGIN
    PERFORM heapdb._check_name('item id', p_id);
    IF p_attrs IS NULL THEN
        PERFORM heapdb._invalid('attrs must be a JSON object, got NULL');
    END IF;
    PERFORM heapdb._check_object('attrs', p_attrs);
    PERFORM heapdb._heap(p_heap);
    RETURN QUERY
    INSERT INTO heapdb.items AS t (heap, id, base, activity, attrs, version)
    VALUES (p_heap, p_id, 1, NULL, p_attrs, 1)
    ON CONFLICT ON CONSTRAINT items_pkey DO UPDATE
        SET attrs = CASE WHEN p_merge THEN t.attrs || excluded.attrs ELSE excluded.attrs END,
            version = t.version + 1
    RETURNING t.id, t.base, t.activity, t.key, t.attrs, t.version;
END
$$;

CREATE OR REPLACE FUNCTION heapdb.remove(p_heap text, p_id text)
RETURNS boolean LANGUAGE plpgsql SECURITY INVOKER AS $$
BEGIN
    PERFORM heapdb._heap(p_heap);
    DELETE FROM heapdb.items t WHERE t.heap = p_heap AND t.id = p_id;
    RETURN FOUND;
END
$$;

CREATE OR REPLACE FUNCTION heapdb.get(
    p_heap text,
    p_id text,
    p_now double precision DEFAULT extract(epoch FROM now())::double precision
)
RETURNS TABLE (rank bigint, id text, key double precision, score double precision, heat double precision,
               base double precision, attrs jsonb, version bigint)
LANGUAGE plpgsql STABLE SECURITY INVOKER AS $$
BEGIN
    PERFORM heapdb._check_finite('now', p_now);
    IF NOT EXISTS (SELECT 1 FROM heapdb.heaps h WHERE h.name = p_heap) THEN
        RAISE EXCEPTION USING ERRCODE = 'HD404',
            MESSAGE = format('unknown heap %L; call create_heap first', p_heap);
    END IF;
    RETURN QUERY
    SELECT 0::bigint, i.id, i.key,
           coalesce(heapdb.decay_exp(i.key - s.shift), 0),
           coalesce(heapdb.decay_exp(i.activity - s.shift), 0),
           i.base, i.attrs, i.version
    FROM heapdb.items i
    JOIN heapdb.heaps h ON h.name = i.heap
    CROSS JOIN LATERAL (SELECT heapdb.lambda(h.half_life) * (p_now - h.landmark) AS shift) s
    WHERE i.heap = p_heap AND i.id = p_id;
END
$$;

-- The k best ranked items matching p_filter (NULL: all), highest key first,
-- ties by id. One query, so the heap row and the items come from the same
-- snapshot.
CREATE OR REPLACE FUNCTION heapdb.top_k(
    p_heap text,
    p_k bigint DEFAULT 10,
    p_now double precision DEFAULT extract(epoch FROM now())::double precision,
    p_filter jsonb DEFAULT NULL
)
RETURNS TABLE (rank bigint, id text, key double precision, score double precision, heat double precision,
               base double precision, attrs jsonb, version bigint)
LANGUAGE plpgsql STABLE SECURITY INVOKER AS $$
BEGIN
    PERFORM heapdb._check_finite('now', p_now);
    IF p_k IS NULL OR p_k < 0 THEN
        PERFORM heapdb._invalid('k must be >= 0');
    END IF;
    PERFORM heapdb.check_filter(p_filter);
    IF NOT EXISTS (SELECT 1 FROM heapdb.heaps h WHERE h.name = p_heap) THEN
        RAISE EXCEPTION USING ERRCODE = 'HD404',
            MESSAGE = format('unknown heap %L; call create_heap first', p_heap);
    END IF;
    RETURN QUERY
    SELECT row_number() OVER (ORDER BY top.key DESC, top.id),
           top.id, top.key, heapdb.decay_exp(top.key - top.shift), heapdb.decay_exp(top.activity - top.shift),
           top.base, top.attrs, top.version
    FROM (
        SELECT i.id, i.key, i.activity, i.base, i.attrs, i.version, s.shift
        FROM heapdb.heaps h
        CROSS JOIN LATERAL (SELECT heapdb.lambda(h.half_life) * (p_now - h.landmark) AS shift) s
        JOIN heapdb.items i ON i.heap = h.name
        WHERE h.name = p_heap AND i.key IS NOT NULL
          AND (p_filter IS NULL OR heapdb.matches(i.attrs, p_filter))
        ORDER BY i.key DESC, i.id
        LIMIT p_k
    ) top
    ORDER BY top.key DESC, top.id;
END
$$;

-- top_k(.., 1, ..) and remove, atomically. FOR UPDATE SKIP LOCKED: an item
-- locked by a concurrent pop is skipped, so concurrent poppers never get
-- the same item.
CREATE OR REPLACE FUNCTION heapdb.pop(
    p_heap text,
    p_now double precision DEFAULT extract(epoch FROM now())::double precision,
    p_filter jsonb DEFAULT NULL
)
RETURNS TABLE (rank bigint, id text, key double precision, score double precision, heat double precision,
               base double precision, attrs jsonb, version bigint)
LANGUAGE plpgsql SECURITY INVOKER AS $$
DECLARE
    h heapdb.heaps;
    r heapdb.items;
    shift double precision;
BEGIN
    PERFORM heapdb._check_finite('now', p_now);
    PERFORM heapdb.check_filter(p_filter);
    h := heapdb._heap(p_heap);
    SELECT * INTO r
    FROM heapdb.items i
    WHERE i.heap = p_heap AND i.key IS NOT NULL
      AND (p_filter IS NULL OR heapdb.matches(i.attrs, p_filter))
    ORDER BY i.key DESC, i.id
    LIMIT 1
    FOR UPDATE SKIP LOCKED;
    IF NOT FOUND THEN
        RETURN;
    END IF;
    DELETE FROM heapdb.items i WHERE i.heap = p_heap AND i.id = r.id;
    shift := heapdb.lambda(h.half_life) * (p_now - h.landmark);
    RETURN QUERY SELECT 1::bigint, r.id, r.key, heapdb.decay_exp(r.key - shift),
                        heapdb.decay_exp(r.activity - shift), r.base, r.attrs, r.version;
END
$$;

-- Move the landmark to p_now (optionally changing the half-life), keeping
-- every item's heat: activity <- activity - lambda_old (now - L_old).
-- Versions do not change.
CREATE OR REPLACE FUNCTION heapdb.rebase(
    p_heap text,
    p_now double precision DEFAULT extract(epoch FROM now())::double precision,
    p_half_life double precision DEFAULT NULL
)
RETURNS TABLE (name text, half_life double precision, landmark double precision, meta jsonb, size bigint)
LANGUAGE plpgsql SECURITY INVOKER AS $$
DECLARE
    h heapdb.heaps;
    shift double precision;
BEGIN
    PERFORM heapdb._check_finite('now', p_now);
    IF p_half_life IS NOT NULL THEN
        PERFORM heapdb._check_positive('half_life', p_half_life);
    END IF;
    h := heapdb._heap(p_heap, true);
    BEGIN
        shift := heapdb.lambda(h.half_life) * (p_now - h.landmark);
        UPDATE heapdb.items i SET activity = i.activity - shift
        WHERE i.heap = p_heap AND i.activity IS NOT NULL;
    EXCEPTION WHEN numeric_value_out_of_range OR check_violation THEN
        PERFORM heapdb._invalid('now is too far from the heap''s landmark (the key is not finite); rebase the heap');
    END;
    UPDATE heapdb.heaps t SET landmark = p_now, half_life = coalesce(p_half_life, t.half_life)
    WHERE t.name = p_heap
    RETURNING * INTO h;
    RETURN QUERY SELECT * FROM heapdb._info(h);
END
$$;

-- ---------------------------------------------------------------------------
-- Version
-- ---------------------------------------------------------------------------

INSERT INTO heapdb.schema_version (singleton, version) VALUES (true, '0.7.0')
ON CONFLICT (singleton) DO UPDATE SET version = excluded.version, installed_at = now();

COMMIT;
