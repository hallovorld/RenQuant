"""Tests for tests/_source_helpers.py — pin the class-body extraction
behavior so future refactors don't silently break dozens of source-
level pinning tests."""
from __future__ import annotations

from tests._source_helpers import class_body, function_body, method_names


SRC_TWO_CLASSES = '''
import os

class Alpha:
    """First class."""

    def __init__(self):
        self.x = 1

    def do_stuff(self):
        return "alpha-stuff"


class Beta:
    """Second class — separator."""

    def helper(self):
        return "beta"


def free_function():
    return "loose"
'''


SRC_NESTED_DEFS = '''
class Outer:
    def method_a(self):
        return 1

    def method_b(self, x):
        # has nested helper
        def inner():
            return 2
        return inner()


class Sibling:
    pass
'''


class TestClassBody:
    def test_extracts_only_the_named_class(self):
        body = class_body(SRC_TWO_CLASSES, "class Alpha")
        assert '"""First class."""' in body
        assert "do_stuff" in body
        # Must NOT include the next class
        assert "class Beta" not in body
        assert "free_function" not in body

    def test_finds_class_and_stops_at_next_class(self):
        body = class_body(SRC_TWO_CLASSES, "class Beta")
        assert "helper" in body
        # Must stop at top-level free function
        assert "free_function" not in body

    def test_method_with_inner_def_kept_inside_outer_class(self):
        """A nested def inside a method is NOT a top-level def, so
        it must NOT terminate the slice early."""
        body = class_body(SRC_NESTED_DEFS, "class Outer")
        assert "method_a" in body
        assert "method_b" in body
        # The nested `def inner()` is indented by 8 spaces — must stay
        assert "def inner()" in body
        # Sibling class must not bleed in
        assert "class Sibling" not in body

    def test_returns_empty_string_when_not_found(self):
        assert class_body(SRC_TWO_CLASSES, "class NonExistent") == ""

    def test_handles_class_at_end_of_file(self):
        src_no_trailing = "class Lone:\n    def x(self):\n        return 1"
        body = class_body(src_no_trailing, "class Lone")
        assert "def x" in body


class TestFunctionBody:
    def test_extracts_top_level_function(self):
        body = function_body(SRC_TWO_CLASSES, "def free_function")
        assert "loose" in body


class TestMethodNames:
    def test_lists_only_class_methods(self):
        names = method_names(SRC_TWO_CLASSES, "class Alpha")
        assert names == ["__init__", "do_stuff"]

    def test_does_not_include_inner_defs(self):
        names = method_names(SRC_NESTED_DEFS, "class Outer")
        # `inner` is nested inside method_b — NOT a method
        assert names == ["method_a", "method_b"]
