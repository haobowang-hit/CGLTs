#!/usr/bin/env python3
"""
Backward-compatible entrypoint for Bayesian multi-objective optimization.

This module is intentionally a thin wrapper over `optimization.hybrid_moo`
to avoid duplicated optimization code paths.
"""

from __future__ import annotations

import sys

try:
    from optimization.hybrid_moo import main as unified_main
except ModuleNotFoundError:
    # Allow direct execution: python src/optimization/moo_bayesian.py ...
    from hybrid_moo import main as unified_main  # type: ignore


def _inject_bayesian_mode(argv: list[str]) -> list[str]:
    # Keep explicit mode if user already provided it.
    if "--mode" in argv:
        return argv
    return [argv[0], "--mode", "bayesian", *argv[1:]]


def main() -> None:
    # Reuse unified parser/implementation in hybrid_moo.py
    sys.argv = _inject_bayesian_mode(sys.argv)
    unified_main()


if __name__ == "__main__":
    main()
