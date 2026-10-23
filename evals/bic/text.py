"""Tokenisation and failure-text parsing (pure Python)."""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

STOP = frozenset("""
a an and are as at be been but by can could did do does for from had has have he her his
how i if in into is it its it's me my no not of on or our she so than that the their them
then there these they this those to too up us was we were what when where which while who
why will with would you your also just only should may might must shall any all some each
other such more most very own same here out over under again further once both few nor
abstract boolean break byte case catch char class const continue default double else enum
extends final finally float goto implements import instanceof int interface long native new
null package private protected public return short static strictfp super switch synchronized
throw throws transient try void volatile true false var
""".split())


def tokens(text: str) -> list[str]:
    """Lower-cased identifier tokens; camelCase/snake_case parts are added
    next to the whole identifier. Stop words, Java keywords and 1-char
    tokens are dropped."""
    out: list[str] = []
    for w in _WORD.findall(text):
        lw = w.lower()
        parts = [p for chunk in w.split("_") for p in _CAMEL.findall(chunk)]
        if len(lw) > 1 and lw not in STOP:
            out.append(lw)
        if len(parts) > 1:
            for p in parts:
                lp = p.lower()
                if len(lp) > 1 and lp not in STOP and not lp.isdigit():
                    out.append(lp)
    return out


def token_counts(text: str) -> Counter:
    return Counter(tokens(text))


# -- unified diffs ---------------------------------------------------------------

def parse_diff(diff: str) -> dict:
    """Files touched with added/deleted line counts, from a unified diff
    with ``--- a/x`` / ``+++ b/x`` headers (FONTE/FBL-BERT format)."""
    files: dict[str, list[int]] = {}
    cur = None
    old = None
    for line in diff.split("\n"):
        if line.startswith("--- "):
            old = line[4:].strip()
            old = old[2:] if old.startswith("a/") else old
            continue
        if line.startswith("+++ "):
            new = line[4:].strip()
            new = new[2:] if new.startswith("b/") else new
            cur = new if new != "/dev/null" else old
            files.setdefault(cur, [0, 0])
            continue
        if cur is None:
            continue
        if line.startswith("+"):
            files[cur][0] += 1
        elif line.startswith("-"):
            files[cur][1] += 1
    return {"files": [[p, a, d] for p, (a, d) in files.items()]}


# -- stack traces --------------------------------------------------------------------

_FRAME = re.compile(r"^\s*at\s+([\w$.]+)\.([\w$<>]+)\(([^)]*)\)", re.M)
FRAMEWORK = ("java.", "javax.", "sun.", "jdk.", "junit.", "org.junit.", "org.apache.tools.ant.",
             "org.hamcrest.", "org.easymock.", "org.mockito.internal.runners.", "com.sun.")


def parse_failures(text: str) -> dict:
    """Failing test ids, exception lines and non-framework stack frames from
    FONTE's ``failing_tests`` file."""
    tests, exc, frames = [], [], []
    expect_exc = False
    for line in text.split("\n"):
        if line.startswith("--- "):
            tests.append(line[4:].strip())
            expect_exc = True
            continue
        m = _FRAME.match(line)
        if m:
            cls, meth = m.group(1), m.group(2)
            if not cls.startswith(FRAMEWORK):
                frames.append((cls.split("$")[0], meth))
            continue
        if expect_exc and line.strip():
            exc.append(line.strip())
            expect_exc = False
    return {"tests": tests, "exceptions": exc, "frames": frames}


def class_to_path_suffix(cls: str) -> str:
    return cls.replace(".", "/") + ".java"


def failure_query_text(f: dict) -> str:
    """The failure as text: test names, exception messages, frame
    class/method names (framework frames dropped)."""
    parts = list(f["tests"]) + list(f["exceptions"])
    parts += [f"{c} {m}" for c, m in f["frames"]]
    return "\n".join(parts)
