"""Shared compilation error base + non-fatal warnings."""
from __future__ import annotations

from dataclasses import dataclass


class CompileError(Exception):
    """Base class for all user-facing compilation errors (lex, parse, check,
    codegen). Tests that don't care which stage caught an error can match
    on CompileError directly.

    Optional `path` attribute set post-construction by the driver when
    the source file the error came from is known. Used by
    `format_error` to render source context."""


def format_error(
    e: CompileError, source_text: str | None = None,
) -> str:
    """Render a CompileError with optional source context.

    Compact gcc/clang-style: `path:line:col: message` followed by
    the offending source line + a caret pointer when source_text and
    line are available. Falls back to bare message-only form when
    neither is known."""
    path = getattr(e, "path", None)
    line = getattr(e, "line", 0)
    col = getattr(e, "col", 0)
    msg = getattr(e, "message", None) or str(e)

    head = ""
    if path:
        head += f"{path}:"
    if line:
        head += f"{line}:{col}: "
    head += msg

    if source_text and line:
        lines = source_text.splitlines()
        if 1 <= line <= len(lines):
            raw = lines[line - 1]
            # Tabs throw off the caret column; expand to the same
            # tab stops the user's editor probably uses (4 cols).
            stripped = raw.replace("\t", "    ")
            tab_shift = sum(
                3 for ch in raw[: max(col - 1, 0)] if ch == "\t"
            )
            caret_col = max(0, col - 1) + tab_shift
            head += f"\n    {stripped}\n    {' ' * caret_col}^"
    return head


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
