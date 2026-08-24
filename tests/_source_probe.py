"""Locate code by STRUCTURE, and fail loudly when the target is gone.

Four audit assertions rotted silently (orch#1022) and each rotted differently:

  * a substring searched inside a hand-tuned BYTE WINDOW (`src[idx:idx+10_000]`),
    bumped from 4k to 8k to 10k across three previous repairs, until the
    function grew past it again;
  * `src.find("class ApplyNGBoostTask")` returning **-1** after the class moved
    to another module, so `src[idx:]` became `src[-1:]` — the assertion then ran
    against a haystack of ONE CHARACTER and could only ever fail, never explain;
  * a literal moved into an extracted helper, so the caller no longer contains
    it while the behaviour is completely intact;
  * an assertion made at the wrong LAYER — the task now latches a flag that the
    job boundary applies, so testing the task alone reports a block that the
    real pipeline still performs.

The common defect is not "source-text assertions are bad". It is that a
*missing target* was indistinguishable from a *failing property*. Everything
here raises `TargetNotFound` when the thing it was asked to inspect does not
exist, so a refactor produces "the code moved" and never a silent -1.
"""

from __future__ import annotations

import ast
from pathlib import Path


class TargetNotFound(AssertionError):
    """The code under inspection is gone or moved — NOT a property failure.

    A distinct type on purpose: "this guarantee is broken" and "I cannot find
    the code that carries it" need different responses, and the old assertions
    could not tell you which one you had.
    """


def parse(path: Path) -> ast.Module:
    if not path.is_file():
        raise TargetNotFound(f"{path} does not exist")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def find_def(tree: ast.AST, name: str, *, where: str) -> ast.AST:
    """The function/class node named `name`, anywhere in `tree`."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == name:
            return node
    raise TargetNotFound(
        f"{name!r} not found in {where} — it was moved, renamed or deleted. "
        f"This is a stale locator, not a failed guarantee: re-point the test "
        f"before concluding anything about the behaviour.")


def find_def_in_package(root: Path, name: str, *, glob: str = "**/*.py") -> tuple[ast.AST, Path]:
    """Same, but searches a package so a file-level move does not rot the test."""
    for path in sorted(root.glob(glob)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and node.name == name:
                return node, path
    raise TargetNotFound(f"{name!r} not found anywhere under {root}")


def assigns_to(node: ast.AST, target_src: str) -> list[ast.Assign]:
    """Every assignment inside `node` whose target renders as `target_src`."""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            for t in sub.targets:
                if ast.unparse(t) == target_src:
                    out.append(sub)
    return out


def guarded_by(node: ast.AST, *, assignment: str, condition: str) -> bool:
    """True if EVERY assignment to `assignment` inside `node` sits under an
    `if` whose test renders as `condition`.

    "Every", not "some": one unguarded write is the bug, and a version that
    accepted the guarded one and ignored the rest would pass on exactly the
    code it exists to reject.
    """
    found = False
    for sub in ast.walk(node):
        if not isinstance(sub, ast.If) or ast.unparse(sub.test) != condition:
            continue
        if assigns_to_body(sub.body, assignment):
            found = True
    if not found:
        return False
    # No assignment may live outside such a branch.
    total = len(assigns_to(node, assignment))
    inside = sum(len(assigns_to_body(sub.body, assignment))
                 for sub in ast.walk(node)
                 if isinstance(sub, ast.If) and ast.unparse(sub.test) == condition)
    return total == inside


def assigns_to_body(body: list, target_src: str) -> list[ast.Assign]:
    out = []
    for stmt in body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if ast.unparse(t) == target_src:
                        out.append(sub)
    return out


def calls(node: ast.AST, name: str) -> bool:
    """Does `node` call `name` (bare or as an attribute)?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id == name:
                return True
            if isinstance(f, ast.Attribute) and f.attr == name:
                return True
    return False
