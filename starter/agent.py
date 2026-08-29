"""Evaluator entry point.

`evaluator/local_evaluator.py` (an unmodified copy of the official kit's
evaluator — see docs/submission_rules.md: "do not edit the evaluator") does
`from starter.agent import Agent`. The actual implementation lives in
`src/agent.py`; this module only re-exports it so the official evaluator's
import path keeps working unchanged.
"""
from src.agent import Agent  # noqa: F401
