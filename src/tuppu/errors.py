"""Shared compilation error base + non-fatal warnings."""
from __future__ import annotations

from dataclasses import dataclass


class CompileError(Exception):
    """Base class for all user-facing compilation errors (lex, parse, check,
    codegen). Tests that don't care which stage caught an error can match
    on CompileError directly."""


@dataclass
class CompileWarning:
    """Non-fatal diagnostic collected during compilation and printed to
    stderr. Format mirrors CompileError: `line:col: message`."""
    message: str
    line: int = 0
    col: int = 0

    def format(self) -> str:
        if self.line:
            return f"tuppu: warning: {self.line}:{self.col}: {self.message}"
        return f"tuppu: warning: {self.message}"
