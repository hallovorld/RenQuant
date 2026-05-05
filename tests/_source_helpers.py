"""Shared utilities for source-level pinning tests.

Many tests in this suite assert that a specific class/function body
contains a specific string (e.g. "must use atomic rename, not direct
save"). The natural pattern is::

    src = path.read_text()
    idx = src.find("class Foo")
    block = src[idx:idx + 6000]      # ← BRITTLE
    assert "thing" in block

The hardcoded char-window above is a time-bomb: as soon as the class
body grows past the window, the test silently misses the asserted
string and fails. We hit this twice in 2026-05-04 (once with NGBoost
fingerprint stamping, once with G7 lot mutation).

Fix: use ``class_body()`` instead. It bounds the slice by the next
class / top-level def, so growth inside the body never breaks the
assertion.

    from tests._source_helpers import class_body
    block = class_body(src, "class Foo")
    assert "thing" in block
"""
from __future__ import annotations

import re


def class_body(src: str, class_decl: str) -> str:
    """Extract the body of a class/function from a source file.

    Slice from ``class_decl`` to the next top-level ``class `` or
    ``def `` (no leading whitespace). Returns empty string if the
    declaration isn't found.

    Why ``\\nclass `` / ``\\ndef `` (with a leading newline): the
    next sibling at top level always begins on a new line at column
    zero. Methods inside the class are ``    def `` (indented) and
    won't terminate the slice. Tested against pp_panel_training.py's
    NGBoostSaveTask which has 12+ inner methods.
    """
    idx = src.find(class_decl)
    if idx < 0:
        return ""
    # Match either next top-level class or top-level def.
    rest = src[idx + len(class_decl):]
    end_class = rest.find("\nclass ")
    end_def   = rest.find("\ndef ")
    candidates = [e for e in (end_class, end_def) if e >= 0]
    if not candidates:
        return src[idx:]
    end = idx + len(class_decl) + min(candidates)
    return src[idx:end]


def function_body(src: str, def_decl: str) -> str:
    """Extract a function body bounded by the next top-level class or def.

    Same algorithm as ``class_body``; the name distinguishes intent.
    """
    return class_body(src, def_decl)


_FUNCDEF_RE = re.compile(r"^(    )?def (\w+)\(", re.MULTILINE)


def method_names(src: str, class_decl: str) -> list[str]:
    """Return the list of method names defined inside a class body.

    Useful for tests that pin "class X must define methods Y and Z".
    """
    body = class_body(src, class_decl)
    return [m.group(2) for m in _FUNCDEF_RE.finditer(body)
            if m.group(1) == "    "]
