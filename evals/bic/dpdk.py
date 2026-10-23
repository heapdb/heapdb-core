"""``Fixes: <sha>`` trailer dataset from DPDK, built at run time.

DPDK's contribution guide asks every fix to name the commit it fixes in a
``Fixes: <12-hex> ("subject")`` trailer. That gives developer-provided
fix -> bug-inducing-commit labels from git metadata alone.

Task (a "suggest the Fixes: tag" helper for a fix under review): given a
fix commit (its message with every trailer and hex id removed, its author
and the files it touches), rank the earlier commits that may have
introduced the bug.

* Candidates (the window): commits strictly earlier (commit time) than the
  fix that touched at least one file the fix touches, newest
  ``MAX_SCOPE`` of them. This is the SZZ-style scope; a BIC outside it (no
  shared file, or older than the newest ``MAX_SCOPE``) counts as a miss for
  every method and is reported.
* Labels: every ``Fixes:`` id that resolves to exactly one earlier commit.

Labels are developer-written but not perfect (a wrong or partial ``Fixes:``
tag is possible). Source pinned to ``sources.DPDK_COMMIT`` (BSD-3-Clause
code; we use commit metadata, file paths and line counts only).
"""

from __future__ import annotations

import bisect
import gzip
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from . import sources
from .text import token_counts

BUILD_VERSION = 2
MAX_SCOPE = 300
#: sha256 of the raw ``git log`` text read at the pinned commit.
EXPECTED_LOG = "d3efc60239f8810409c6c7e11688a056adbaabf02f9b3b6985b41c44e63e9810"

_FIXES = re.compile(r"^\s*Fixes:\s*([0-9a-fA-F]{7,40})", re.M)
_TRAILER = re.compile(
    r"^\s*(fixes|cc|signed-off-by|acked-by|reviewed-by|tested-by|reported-by|suggested-by|"
    r"co-authored-by|co-developed-by|bugzilla id|coverity issue|link|depends-on|reviewed-on|"
    r"change-id|cve|closes|see-also|ack|nacked-by)\s*:", re.I)
_HEX = re.compile(r"\b[0-9a-fA-F]{7,40}\b")


def query_text(body: str) -> str:
    """Fix message without trailers and without hex commit ids."""
    lines = [ln for ln in body.split("\n") if not _TRAILER.match(ln)]
    return _HEX.sub(" ", "\n".join(lines))


def component(subject: str) -> str:
    """DPDK subject prefix, e.g. ``net/mlx5`` from ``net/mlx5: fix ...``."""
    head = subject.split(":", 1)[0].strip().lower()
    return head if ":" in subject and len(head) <= 40 else "?"


def build(out: Path, verbose: bool = True) -> dict:
    t0 = time.time()
    repo = sources.dpdk_repo()
    raw = sources._git("-C", str(repo), "log", "--no-renames", "--numstat",
                       "--format=%x1e%H%x1f%P%x1f%ct%x1f%ae%x1f%B%x1d", sources.DPDK_COMMIT)
    log_sha = __import__("hashlib").sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    commits = []
    for rec in text.split("\x1e")[1:]:
        head, _, stat = rec.partition("\x1d")
        sha, parents, ct, ae, body = head.split("\x1f", 4)
        files = []
        for line in stat.strip().split("\n"):
            if not line.strip():
                continue
            a, d, p = line.split("\t", 2)
            files.append([p, int(a) if a != "-" else 0, int(d) if d != "-" else 0])
        commits.append({"sha": sha, "merge": len(parents.split()) > 1, "t": int(ct),
                        "a": ae.lower(), "body": body.strip(), "f": files})
    commits.sort(key=lambda c: (c["t"], c["sha"]))       # oldest first; index = id
    idx = {c["sha"]: i for i, c in enumerate(commits)}
    by_prefix = defaultdict(list)
    for c in commits:
        by_prefix[c["sha"][:7]].append(c["sha"])
    vocab: dict[str, int] = {}

    def enc(cnt) -> list[list[int]]:
        return [[vocab.setdefault(t, len(vocab)), n] for t, n in sorted(cnt.items())]

    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)   # path -> [(t, idx)]
    out_commits = []
    for i, c in enumerate(commits):
        subject = c["body"].split("\n", 1)[0]
        q = query_text(c["body"])
        out_commits.append({
            "sha": c["sha"], "t": c["t"], "a": c["a"], "comp": component(subject),
            "f": c["f"], "fix": bool(_FIXES.search(c["body"])), "merge": c["merge"],
            "tm": enc(token_counts(q)), "subj": subject,
        })
        if not c["merge"]:
            for p, _, _ in c["f"]:
                by_file[p].append((c["t"], i))
    bugs = []
    stats = defaultdict(int)
    for i, c in enumerate(commits):
        refs = _FIXES.findall(c["body"])
        if not refs:
            continue
        stats["fix_commits"] += 1
        if c["merge"] or not c["f"]:
            stats["skipped_merge_or_empty"] += 1
            continue
        bics = []
        for r in refs:
            r = r.lower()
            m = [s for s in by_prefix.get(r[:7], []) if s.startswith(r)]
            if len(m) != 1:
                stats["unresolved_refs"] += 1
                continue
            j = idx[m[0]]
            if commits[j]["t"] >= c["t"] or j == i:
                stats["refs_not_earlier"] += 1
                continue
            bics.append(j)
        if not bics:
            stats["skipped_no_resolvable_bic"] += 1
            continue
        scope = set()
        for p, _, _ in c["f"]:
            lst = by_file.get(p, [])
            k = bisect.bisect_left(lst, (c["t"], -1))
            scope.update(j for _, j in lst[:k] if j != i)
        ordered = sorted(scope, key=lambda j: (-commits[j]["t"], commits[j]["sha"]))
        in_files = any(j in scope for j in bics)
        cands = ordered[:MAX_SCOPE]
        in_cap = any(j in cands for j in bics)
        stats["bugs"] += 1
        stats["bic_no_shared_file"] += not in_files
        stats["bic_beyond_cap"] += in_files and not in_cap
        bic_subjects = [out_commits[j]["subj"] for j in bics]
        q = query_text(c["body"]).lower()
        stats["query_contains_bic_subject"] += any(len(s) >= 15 and s.lower() in q for s in bic_subjects)
        bugs.append({"id": c["sha"][:12], "fix": i, "bics": bics, "cands": cands})
    data = {
        "meta": {"builder": BUILD_VERSION, "dpdk_url": sources.DPDK_URL,
                 "dpdk_commit": sources.DPDK_COMMIT, "log_sha256": log_sha,
                 "max_scope": MAX_SCOPE, "commits": len(commits), "stats": dict(stats),
                 "build_seconds": round(time.time() - t0, 1)},
        "vocab": [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])],
        "commits": out_commits,
        "bugs": bugs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    if verbose:
        print(f"[dpdk] built {out} in {time.time() - t0:.0f}s; {dict(stats)}", flush=True)
    return data


def load(allow_mismatch: bool = False, rebuild: bool = False) -> dict:
    path = sources.cache_dir() / f"dpdk-v{BUILD_VERSION}.json.gz"
    if rebuild or not path.exists():
        data = build(path)
    else:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    sources.check_pin("DPDK git log", data["meta"]["log_sha256"], EXPECTED_LOG, allow_mismatch)
    return data
