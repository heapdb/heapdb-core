"""Pinned, run-time downloads for the bug-inducing-commit evaluation.

Nothing from the datasets is committed to this repository. Every source is a
git repository on GitHub, fetched into a cache directory and pinned to an
exact commit. The bytes the evaluation consumes are then hashed (sha256) and
compared with the values recorded here, so a moved branch, a rewritten
history or a changed extraction is detected instead of silently producing
different numbers.

Cache: ``$HEAPDB_BIC_CACHE`` or ``~/.cache/heapdb-bic``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

#: FONTE artifact (Kim et al., ICSE 2023 + the journal extension), MIT licence.
#: Branch ``extension`` holds the 206-bug Defects4J BIC dataset.
FONTE_URL = "https://github.com/coinse/fonte"
FONTE_COMMIT = "f8739873615164a69d9274bc0b12334f8bd1a41a"

#: DPDK (BSD-3-Clause), used for the ``Fixes: <sha>`` trailer dataset.
DPDK_URL = "https://github.com/DPDK/dpdk"
DPDK_COMMIT = "0e03411639496a7f9705494db31cde313f0e80b5"

#: Upstream repositories of the Defects4J projects, used only for commit
#: authors (FONTE's commit files have no author field). Treeless clones:
#: commit objects only. Pinned to the HEAD seen when the evaluation was built.
UPSTREAM = {
    "Cli": ("apache/commons-cli", "03ff1171f35a45721131c4c7ef65973a796f8d6a"),
    "Closure": ("google/closure-compiler", "bf3df97a98240da9b3b36218e2a251a33e2d2091"),
    "Codec": ("apache/commons-codec", "bbcb88433ebf69310774c07f26bbb8b2762795f1"),
    "Collections": ("apache/commons-collections", "892bdc8fa734b57f8f7ede3d41e23f8283a95b71"),
    "Compress": ("apache/commons-compress", "be13953fc467dad1e9bc83c727b79e070e9ad339"),
    "Csv": ("apache/commons-csv", "b217f58fac4bdf26507cde121ffb9cd50fd335e7"),
    "Gson": ("google/gson", "e2292f9c9f2e07a4791827e8eea34aa7d80545d6"),
    "JacksonCore": ("FasterXML/jackson-core", "8155a6fb0832523909ea8fd8dd617334eb5d617a"),
    "JacksonDatabind": ("FasterXML/jackson-databind", "b72438baf8755836913f9d4d9f0863c24c99d42d"),
    "JacksonXml": ("FasterXML/jackson-dataformat-xml", "c92315af6655ad6a4f94413231bf67770db8ba07"),
    "Jsoup": ("jhy/jsoup", "bcf979840fdc986893cce9eb54b88cabe74b18ff"),
    "JxPath": ("apache/commons-jxpath", "c212f8553aa2bff1214dad3d3010d6da8fa8cc5a"),
    "Lang": ("apache/commons-lang", "567bcda3de4d8768a2dc1e3ffc567b5879a9a74f"),
    "Math": ("apache/commons-math", "a0df91d88eef1a124d0a93ea13c230de39e7b6cf"),
    "Mockito": ("mockito/mockito", "220c594adceebf92a0d06e2fb1e4398b25344d46"),
    "Time": ("JodaOrg/joda-time", "5ffcf3c2bce34c95207ef2746229a40a932afeee"),
}


def cache_dir() -> Path:
    base = os.environ.get("HEAPDB_BIC_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "heapdb-bic")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _git(*args: str, cwd: Path | None = None, input: bytes | None = None) -> bytes:
    env = dict(os.environ)
    # Never let a lookup of a missing object in a partial clone hit the network.
    env["GIT_NO_LAZY_FETCH"] = "1"
    r = subprocess.run(["git", *args], cwd=cwd, input=input, capture_output=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace')[-2000:]}")
    return r.stdout


def has_commit(repo: Path, sha: str) -> bool:
    try:
        return _git("-C", str(repo), "cat-file", "-t", sha).strip() == b"commit"
    except RuntimeError:
        return False


def ensure_repo(url: str, dest: Path, commit: str, *, depth1_branch: str | None = None,
                treeless: bool = False) -> Path:
    """Bare clone of ``url`` into ``dest`` that contains ``commit``.

    ``depth1_branch``: clone only that branch at depth 1 (the FONTE artifact
    is a large tree with a short history). ``treeless``: commit objects only.
    """
    if dest.exists() and has_commit(dest, commit):
        return dest
    if not dest.exists():
        args = ["clone", "--quiet", "--bare"]
        if depth1_branch:
            args += ["--single-branch", "--branch", depth1_branch, "--depth", "1"]
        if treeless:
            args += ["--filter=tree:0"]
        _git(*args, url, str(dest))
    if not has_commit(dest, commit):
        # The branch moved on: fetch the pinned commit itself (GitHub serves
        # reachable commits by id).
        extra = ["--depth", "1"] if depth1_branch else []
        _git("-C", str(dest), "fetch", "--quiet", *extra, "origin", commit)
    if not has_commit(dest, commit):
        raise RuntimeError(f"{url}: pinned commit {commit} not available")
    return dest


def fonte_repo() -> Path:
    return ensure_repo(FONTE_URL, cache_dir() / "fonte.git", FONTE_COMMIT, depth1_branch="extension")


def dpdk_repo() -> Path:
    return ensure_repo(DPDK_URL, cache_dir() / "dpdk.git", DPDK_COMMIT)


def upstream_repo(project: str) -> Path:
    slug, commit = UPSTREAM[project]
    return ensure_repo(f"https://github.com/{slug}", cache_dir() / f"upstream-{project}.git",
                       commit, treeless=True)


# -- reading blobs -------------------------------------------------------------

def ls_tree(repo: Path, commit: str, path: str) -> list[tuple[str, str]]:
    """Recursive ``(path, blob id)`` under ``path`` at ``commit``."""
    out = _git("-C", str(repo), "ls-tree", "-r", "-z", commit, "--", path).decode()
    rows = []
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, p = rec.split("\t", 1)
        _mode, kind, oid = meta.split()
        if kind == "blob":
            rows.append((p, oid))
    return rows


def read_blobs(repo: Path, oids: list[str]) -> dict[str, bytes]:
    """Read many blobs with one ``git cat-file --batch``."""
    uniq = list(dict.fromkeys(oids))
    if not uniq:
        return {}
    data = _git("-C", str(repo), "cat-file", "--batch", input=("\n".join(uniq) + "\n").encode())
    out: dict[str, bytes] = {}
    i = 0
    for oid in uniq:
        j = data.index(b"\n", i)
        hdr = data[i:j].split()
        if len(hdr) < 3 or hdr[1] == b"missing":
            raise RuntimeError(f"blob {oid} missing")
        n = int(hdr[2])
        out[oid] = data[j + 1:j + 1 + n]
        i = j + 1 + n + 1
    return out


class Manifest:
    """sha256 of every input file consumed, combined into one digest.

    ``digest`` = sha256 over sorted ``"<path>\\t<sha256>\\n"`` lines.
    """

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def add(self, path: str, data: bytes) -> bytes:
        self.files[path] = hashlib.sha256(data).hexdigest()
        return data

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.files):
            h.update(f"{p}\t{self.files[p]}\n".encode())
        return h.hexdigest()


def sha256_text(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8", errors="surrogateescape"))
        h.update(b"\n")
    return h.hexdigest()


def check_pin(name: str, got: str, expected: str | None, allow_mismatch: bool) -> None:
    if expected is None:
        print(f"[pin] {name}: sha256 {got} (no expected value recorded)")
        return
    if got != expected:
        msg = f"[pin] {name}: sha256 {got} != expected {expected}"
        if not allow_mismatch:
            raise SystemExit(msg + " (inputs changed; rerun with --allow-mismatch to proceed anyway)")
        print(msg + " (continuing: --allow-mismatch)")
