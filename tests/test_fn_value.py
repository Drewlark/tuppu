"""First-class function values without environment capture. A bare fn
name is a value of type `fn(params) -> ret`; it can be passed, stored
in a binding, returned, held in a struct field, or called through.
Closures (capturing local state) remain a future pass."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_to_ir
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- pass / store / call --------------------------------------------------

def test_fn_passed_and_called_through_param(tmp_path):
    src = (
        'fn double(x: i64) -> i64 { x * 2 }\n'
        'fn triple(x: i64) -> i64 { x * 3 }\n'
        'fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }\n'
        'fn main() -> i32 {\n'
        '  println(apply(double, 7))\n'
        '  println(apply(triple, 7))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"14\n21\n"


def test_fn_stored_in_step_binding(tmp_path):
    src = (
        'fn add_one(x: i64) -> i64 { x + 1 }\n'
        'fn main() -> i32 {\n'
        '  step f = add_one\n'
        '  println(f(41))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"42\n"


def test_fn_reassigned_through_mut(tmp_path):
    src = (
        'fn neg(x: i64) -> i64 { 0 - x }\n'
        'fn abs(x: i64) -> i64 { if x < 0 { 0 - x } else { x } }\n'
        'fn main() -> i32 {\n'
        '  mut f: fn(i64) -> i64 = neg\n'
        '  println(f(5))\n'
        '  f = abs\n'
        '  println(f(0 - 7))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"-5\n7\n"


def test_fn_returned_from_fn(tmp_path):
    # Ret-type is a fn pointer. Caller binds, calls through. No
    # capture, no lifetime issue — the returned pointer refers to a
    # statically-compiled fn.
    src = (
        'fn succ(x: i64) -> i64 { x + 1 }\n'
        'fn pred(x: i64) -> i64 { x - 1 }\n'
        'fn pick(up: bool) -> fn(i64) -> i64 {\n'
        '  if up { succ } else { pred }\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  step f = pick(true)\n'
        '  step g = pick(false)\n'
        '  println(f(10))\n'
        '  println(g(10))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"11\n9\n"


def test_fn_value_with_str_param(tmp_path):
    # Marshaling parity with direct calls: a fn-value call on a fn
    # that takes `str` must apply the same cap=0 borrow shape so the
    # callee's cleanup frame no-ops.
    src = (
        'fn first_byte(s: str) -> i64 { s[0] as i64 }\n'
        'fn main() -> i32 {\n'
        '  step f = first_byte\n'
        '  println(f("A"))\n'       # 65
        '  println(f("z"))\n'       # 122
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"65\n122\n"


def test_fn_value_held_in_struct_field(tmp_path):
    src = (
        'tablet Unary { name: str, run: fn(i64) -> i64 }\n'
        'fn sq(x: i64) -> i64 { x * x }\n'
        'fn cb(x: i64) -> i64 { x * x * x }\n'
        'fn main() -> i32 {\n'
        '  step a: Unary = Unary { name: "sq", run: sq }\n'
        '  step b: Unary = Unary { name: "cb", run: cb }\n'
        '  println(a.run(5))\n'
        '  println(b.run(5))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"25\n125\n"


# --- errors / rejections --------------------------------------------------

def test_colophon_cannot_be_taken_as_value():
    with pytest.raises(CompileError, match="taking its address"):
        compile_to_ir(
            'colophon fn exit(code: i32)\n'
            'fn main() -> i32 {\n'
            '  step f = exit\n'
            '  0\n'
            '}\n'
        )


def test_generic_fn_cannot_be_taken_as_value():
    with pytest.raises(CompileError, match="generic"):
        compile_to_ir(
            'fn id<T>(x: T) -> T { x }\n'
            'fn main() -> i32 {\n'
            '  step f = id\n'
            '  0\n'
            '}\n'
        )


def test_non_fn_binding_not_callable():
    with pytest.raises(CompileError, match="not callable"):
        compile_to_ir(
            'fn main() -> i32 {\n'
            '  step x: i64 = 5\n'
            '  x(3)\n'
            '  0\n'
            '}\n'
        )


def test_fn_value_arg_type_mismatch():
    # str doesn't coerce to i64, so passing one where the fn-value
    # wants an int should error — via the fn-value call typecheck,
    # not the direct-call path.
    with pytest.raises(CompileError, match="fn-value call"):
        compile_to_ir(
            'fn need_i64(x: i64) -> i64 { x }\n'
            'fn main() -> i32 {\n'
            '  step f = need_i64\n'
            '  f("nope")\n'
            '  0\n'
            '}\n'
        )


def test_fn_value_arity_mismatch():
    with pytest.raises(CompileError, match="fn-value call"):
        compile_to_ir(
            'fn add(a: i64, b: i64) -> i64 { a + b }\n'
            'fn main() -> i32 {\n'
            '  step f = add\n'
            '  f(1)\n'
            '  0\n'
            '}\n'
        )
