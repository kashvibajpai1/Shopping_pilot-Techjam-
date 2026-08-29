#!/usr/bin/env python3
"""Convenience wrapper around the official local evaluator.

Equivalent to `python3 -m evaluator.local_evaluator` with the same default
paths — kept as a separate script only so `README.md` has one obvious
command to point at. Does not modify evaluator/local_evaluator.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import main  # noqa: E402

if __name__ == "__main__":
    main()
