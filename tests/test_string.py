"""Variable-length strings as the built-in `str` seal.

Covers: string literal → str tablet value, print dispatch via %.*s, field
access `s.len`, byte indexing `s[i]`, passing strings as function args,
and the stdlib helpers in `stdlib/str.tpu`."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import (
    compile_files_to_binary,
    compile_to_binary,
    compile_to_ir,
    stdlib_files,
)
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


def run_with_stdlib(src: str, tmp_path: Path) -> tuple[int, bytes, bytes]:
    user_file = tmp_path / "main.tpu"
    user_file.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user_file], tmp_path, name="prog",
    )
    r = subprocess.run([str(binary)], capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- literal, print, len, index -------------------------------------------

def test_string_literal_print(tmp_path):
    _, out, _ = run('fn main() -> i32 { println("hello")\n 0 }', tmp_path)
    assert out == b"hello\n"


def test_print_vs_println(tmp_path):
    _, out, _ = run(
        'fn main() -> i32 { print("one ")\n print("two ")\n println("three")\n 0 }',
        tmp_path,
    )
    assert out == b"one two three\n"


def test_s_dot_len(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hello"\n'
        '  println(s.len)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"5\n"


def test_empty_string_len_and_print(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  step s = ""\n'
        '  println(s.len)\n'
        '  print(s)\n'
        '  println("done")\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\ndone\n"


def test_s_at_index_returns_u8(tmp_path):
    # "hello"[1] == 'e' (0x65 = 101)
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hello"\n'
        '  println(s[1] as i64)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"101\n"


def test_string_index_bounds_trap(tmp_path):
    # Reading past the end should trap (SIGILL via llvm.trap on macOS).
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hi"\n'
        '  println(s[2] as i64)\n'   # out of bounds
        '  0\n'
        '}\n'
    )
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.returncode != 0, "expected trap on OOB index"


# --- escape sequences survive the redesign --------------------------------

def test_escapes_newline_and_tab(tmp_path):
    src = 'fn main() -> i32 { println("a\\tb\\nc")\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"a\tb\nc\n"


def test_escapes_with_internal_nul(tmp_path):
    # %.*s respects the length field and must print past an embedded NUL.
    src = 'fn main() -> i32 { print("ab\\0cd")\n println("!")\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"ab\x00cd!\n"


# --- strings as function parameters and return values ---------------------

def test_string_as_parameter(tmp_path):
    src = (
        'fn greet(name: str) {\n'
        '  print("hi, ")\n'
        '  println(name)\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  greet("Drew")\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hi, Drew\n"


def test_string_as_return_value(tmp_path):
    src = (
        'fn label(tag: i64) -> str {\n'
        '  if tag == 0 { "zero" } else { "nonzero" }\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  println(label(0))\n'
        '  println(label(7))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"zero\nnonzero\n"


# --- ownership across fn boundaries ---------------------------------------
#
# The cap=0 borrow sentinel: call-site forces cap=0 on every str arg so
# the callee registers the param uniformly — release becomes a no-op for
# borrows, caller retains sole ownership of the heap bytes.

def test_heap_str_passed_to_fn_usable_after(tmp_path):
    # If the callee double-freed an owned str, the second println would
    # hit freed memory. Shipping the borrow means the caller still owns.
    src = (
        'fn show(s: str) { println(s) }\n'
        'fn main() -> i32 {\n'
        '  step s = str_concat("hi, ", "Drew")\n'
        '  show(s)\n'
        '  println(s)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hi, Drew\nhi, Drew\n"


def test_mut_str_param_reassign_releases(tmp_path):
    # Reassigning a mut str param to a heap-owned str must free the new
    # storage at scope exit — otherwise we leak every call.
    src = (
        'fn rebuild(mut s: str) {\n'
        '  s = str_concat(s, "!")\n'
        '  println(s)\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  rebuild("hi")\n'
        '  rebuild("bye")\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hi!\nbye!\n"


def test_unbound_concat_does_not_crash(tmp_path):
    # `println(str_concat(...))` used to leak; the anonymous-temp auto-
    # release machinery registers the rvalue in the caller's frame so it
    # frees at scope exit. Correctness proxy: program runs + right output.
    src = (
        'fn main() -> i32 {\n'
        '  println(str_concat("foo", "bar"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"foobar\n"


def test_nested_concat_inside_call_no_leak(tmp_path):
    # Each inner str_concat produces a heap str consumed by the outer
    # call; every layer must register a cleanup so intermediates get
    # freed. If this hits the allocator repeatedly without freeing,
    # long loops would blow out RSS — here we just verify the program
    # terminates with the right output.
    src = (
        'fn main() -> i32 {\n'
        '  mut i: i64 = 0\n'
        '  while i < 1000 {\n'
        '    println(str_concat(str_concat("a", "b"), int_to_str(i)))\n'
        '    i = i + 1\n'
        '  }\n'
        '  0\n'
        '}\n'
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    lines = out.split(b"\n")
    assert lines[0] == b"ab0"
    assert lines[999] == b"ab999"


def test_mut_str_param_release_is_wired(tmp_path):
    # Compile-only check: a mut str param needs a release call so a
    # reassignment to a heap value doesn't leak at scope exit. Non-mut
    # params stay SSA with cap=0 (nothing to free), so this test uses
    # `mut` deliberately.
    src = (
        'fn maybe_grow(mut s: str) { println(s) }\n'
        'fn main() -> i32 { maybe_grow("hi")\n 0 }\n'
    )
    ir = compile_to_ir(src)
    assert "__tuppu_str_release" in ir


def test_str_rvalue_temp_release_is_wired(tmp_path):
    # `println(str_concat(a, b))` must register an anonymous cleanup for
    # the concat result — the IR should show at least one release call
    # in main after the concat.
    src = (
        'fn main() -> i32 {\n'
        '  println(str_concat("foo", "bar"))\n'
        '  0\n'
        '}\n'
    )
    ir = compile_to_ir(src)
    assert "__tuppu_str_release" in ir


# --- stdlib helpers --------------------------------------------------------

def test_stdlib_str_eq(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_eq("foo", "foo"))\n'
        '  println(str_eq("foo", "bar"))\n'
        '  println(str_eq("foo", "foobar"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\nfalse\n"


def test_stdlib_starts_with(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_starts_with("hello, world", "hello"))\n'
        '  println(str_starts_with("hello", "hi"))\n'
        '  println(str_starts_with("hi", "hello"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\nfalse\n"


def test_stdlib_ends_with(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_ends_with("hello.tpu", ".tpu"))\n'
        '  println(str_ends_with("hello.tpu", ".py"))\n'
        '  println(str_ends_with("hi", "hello"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\nfalse\n"


def test_stdlib_index_of(tmp_path):
    # 'l' = 108
    src = (
        'fn main() -> i32 {\n'
        '  println(str_index_of("hello", 108 as u8))\n'   # first 'l' at 2
        '  println(str_index_of("hello", 122 as u8))\n'   # 'z' absent
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"2\n-1\n"


def test_stdlib_is_empty(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_is_empty(""))\n'
        '  println(str_is_empty("x"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\n"


# --- error cases -----------------------------------------------------------

def test_bracket_array_type_rejected():
    # The old `[N]u8` shorthand is retired; users should write `str`.
    with pytest.raises(CompileError, match="array types are not supported"):
        compile_to_ir(
            "fn f(s: [5]u8) -> i32 { 0 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_str_index_needs_integer():
    with pytest.raises(CompileError, match="str index must be integer"):
        compile_to_ir(
            'fn main() -> i32 {\n'
            '  step s = "hi"\n'
            '  println(s[true] as i64)\n'
            '  0\n'
            '}\n'
        )
