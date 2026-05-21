"""kernel — self-contained strategy logic shared by LEAN, notebook, and live runner.

No common/ imports.  Only stdlib + numpy + pandas.

The repo also has a small top-level ``kernel/`` package for research
utilities and pytest-only helpers.  Keep this strategy-local package as a
namespace participant so whichever ``kernel`` package is imported first can
still resolve modules from the other location during local tests.
"""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
