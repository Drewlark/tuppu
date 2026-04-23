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
    # String tests implicitly use the bundled stdlib — str_concat,
    # int_to_str, str_repeat, etc. are defined there. Matches the
    # default `./tuppu run` shape; individual tests wanting a bare
    # compilation call compile_to_binary directly.
    user_file = tmp_path / "main.tpu"
    user_file.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user_file], tmp_path, name="prog",
    )
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


# Back-compat alias: keep the old name working for tests that spell it.
run_with_stdlib = run


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


# --- typed-to-str conversions --------------------------------------------

def test_int_to_str(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(int_to_str(0))\n'
        '  println(int_to_str(42))\n'
        '  println(int_to_str(-9223372036854775807))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n42\n-9223372036854775807\n"


def test_rat_to_str(tmp_path):
    # rat_to_str lives in stdlib/str.tpu — built on top of the native
    # int_to_str + str_concat.
    src = (
        'fn main() -> i32 {\n'
        '  println(rat_to_str(rat(3, 4)))\n'
        '  println(rat_to_str(rat(-7, 2)))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"3/4\n-7/2\n"


def test_sex_to_str_mirrors_print(tmp_path):
    # sex_to_str must match the Babylonian form emitted by println(s)
    # so string-built logs read identically to direct prints.
    src = (
        'fn main() -> i32 {\n'
        '  step a = 3600 as sex\n'
        '  println(a)\n'
        '  println(sex_to_str(a))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1 0 0\n1 0 0\n"


def test_bool_to_str(tmp_path):
    # bool_to_str lives in stdlib/str.tpu — returns a borrow of a
    # literal so there's no allocation.
    src = (
        'fn main() -> i32 {\n'
        '  println(bool_to_str(true))\n'
        '  println(bool_to_str(false))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\n"


def test_str_concat_variadic(tmp_path):
    # `str_concat(a, b, c, ...)` takes any arity >= 2 and emits one
    # linear-time join — single malloc, memcpy each part in order.
    # Mixed literals and Call rvalues compose cleanly; any heap
    # intermediates get freed at scope exit.
    src = (
        'fn main() -> i32 {\n'
        '  println(str_concat("a", "b", "c"))\n'
        '  println(str_concat("x=", int_to_str(7), " y=", int_to_str(9)))\n'
        '  println(str_concat("", "u", "", "v", ""))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"abc\nx=7 y=9\nuv\n"


def test_str_plus_operator(tmp_path):
    # `a + b` on strs lowers to the same single-malloc concat as
    # str_concat — syntax sugar with identical semantics.
    src = (
        'fn main() -> i32 {\n'
        '  step a = "hello"\n'
        '  step b = ", "\n'
        '  step c = "world"\n'
        '  println(a + b + c)\n'
        '  println("count=" + int_to_str(42))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hello, world\ncount=42\n"


def test_str_plus_equals(tmp_path):
    # `s += t` desugars to `s = s + t`; the mut-reassign machinery
    # releases the old s before storing the new heap concat result,
    # so the accumulator doesn't leak across iterations.
    src = (
        'fn main() -> i32 {\n'
        '  mut acc: str = ""\n'
        '  acc += "alpha "\n'
        '  acc += "beta "\n'
        '  acc += "gamma"\n'
        '  println(acc)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"alpha beta gamma\n"


def test_str_plus_equals_loop_no_leak(tmp_path):
    # Classic O(n²) accumulator shape. It IS quadratic in work but
    # the ownership machinery still collects every intermediate —
    # RSS stays bounded, no UAFs, just some wasted copies.
    src = (
        'fn main() -> i32 {\n'
        '  mut acc: str = ""\n'
        '  mut i: i64 = 0\n'
        '  while i < 100 {\n'
        '    acc += "x"\n'
        '    i = i + 1\n'
        '  }\n'
        '  println(acc.len)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"100\n"


def test_concat_typed_parts_builds_log_line(tmp_path):
    # Realistic dynamic-string workflow: assemble a line from mixed
    # typed pieces via concat + to_str. All intermediates must free
    # at scope exit.
    src = (
        'fn main() -> i32 {\n'
        '  step name = "answer"\n'
        '  step piece = str_concat(name, " = ")\n'
        '  step line = str_concat(piece, int_to_str(42))\n'
        '  println(line)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"answer = 42\n"


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


def test_mut_str_returned_as_tail_keeps_ownership(tmp_path):
    # If a block's tail is an Ident naming a local str binding, the
    # scope-exit cleanup must NOT free it — the caller receives the
    # value and owns it. Without the ownership-transfer rule, the heap
    # bytes get freed inside the callee and the caller sees garbage.
    src = (
        'fn build() -> str {\n'
        '  mut s: str = ""\n'
        '  s = str_concat(s, "hello")\n'
        '  s = str_concat(s, ", world")\n'
        '  s\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  step msg = build()\n'
        '  println(msg)\n'
        '  println(msg.len)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hello, world\n12\n"


def test_step_str_returned_as_tail_keeps_ownership(tmp_path):
    # Same transfer rule applies to step bindings that hold a heap str.
    src = (
        'fn build() -> str {\n'
        '  step s = str_concat("foo", "bar")\n'
        '  s\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  println(build())\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"foobar\n"


def test_step_borrow_does_not_double_free(tmp_path):
    # `step x = y` where y owns a heap str shares y's metadata. Both
    # x and y used to register cleanup, producing a double-free at
    # scope exit (SIGABRT on macOS). The borrow rule skips x's
    # registration; only y's owner releases.
    src = (
        'fn main() -> i32 {\n'
        '  mut y: str = str_concat("hello", ", world")\n'
        '  step x = y\n'
        '  println(x)\n'
        '  println(y)\n'
        '  0\n'
        '}\n'
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello, world\nhello, world\n"


def test_step_borrow_chain_transfers_through(tmp_path):
    # Chained borrows: `step w = y; step v = w`. v's transfer_on_tail
    # must thread back through w to y (the real owner) so returning v
    # hands heap ownership to the caller.
    src = (
        'fn build() -> str {\n'
        '  mut y: str = str_concat("Ur", "uk")\n'
        '  step w = y\n'
        '  step v = w\n'
        '  v\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  step msg = build()\n'
        '  println(msg)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"Uruk\n"


def test_step_borrow_returned_as_tail_transfers_source(tmp_path):
    # `step x = y; x` as the function tail: transfer-on-tail follows
    # the borrow chain to the real owner (y) and deregisters that so
    # the returned heap ptr is live in the caller's scope.
    src = (
        'fn build() -> str {\n'
        '  mut y: str = str_concat("hi", "!")\n'
        '  step x = y\n'
        '  x\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  step msg = build()\n'
        '  println(msg)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hi!\n"


def test_tail_transfer_inside_if_arm(tmp_path):
    # The transfer happens at every block boundary: if-arm block
    # returns a locally-bound str, outer block returns the if result.
    src = (
        'fn build(flag: bool) -> str {\n'
        '  if flag {\n'
        '    step s = str_concat("flag ", "on")\n'
        '    s\n'
        '  } else {\n'
        '    "default"\n'
        '  }\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  println(build(true))\n'
        '  println(build(false))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"flag on\ndefault\n"


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


def test_slice_syntax_basic(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hello, world"\n'
        '  println(s[0:5])\n'
        '  println(s[7:12])\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hello\nworld\n"


def test_slice_syntax_elided_bounds(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hello"\n'
        '  println(s[:3])\n'
        '  println(s[2:])\n'
        '  println(s[:])\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hel\nllo\nhello\n"


def test_slice_of_concat_no_leak(tmp_path):
    # Slicing an rvalue: the heap bytes from str_concat must be
    # registered for cleanup alongside the slice result so neither
    # leaks across the iteration.
    src = (
        'fn main() -> i32 {\n'
        '  mut i: i64 = 0\n'
        '  while i < 500 {\n'
        '    println(str_concat("foo", "bar")[1:5])\n'
        '    i = i + 1\n'
        '  }\n'
        '  0\n'
        '}\n'
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out.count(b"ooba\n") == 500


def test_slice_bounds_trap(tmp_path):
    # Out-of-range hi should trap at runtime via __tuppu_str_slice's check.
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hi"\n'
        '  println(s[0:99])\n'
        '  0\n'
        '}\n'
    )
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.returncode != 0


def test_slice_non_str_rejected():
    with pytest.raises(CompileError, match="slice syntax is only supported on str"):
        compile_to_ir(
            'fn main() -> i32 {\n'
            '  mut t: tablets[4]i64\n'
            '  t[0:2]\n'
            '  0\n'
            '}\n'
        )


def test_str_rvalue_temp_release_is_wired(tmp_path):
    # `println("foo" + "bar")` must register an anonymous cleanup for
    # the concat result — the IR should show at least one release call
    # in main after the concat. Uses `+` since it's the native path
    # that doesn't require stdlib.
    src = (
        'fn main() -> i32 {\n'
        '  println("foo" + "bar")\n'
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


def test_stdlib_find(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_find("hello, world", "world"))\n'   # 7
        '  println(str_find("hello", "hello"))\n'           # 0
        '  println(str_find("hello", ""))\n'                # 0 (empty matches at start)
        '  println(str_find("hello", "xyz"))\n'             # -1
        '  println(str_find("ab", "abc"))\n'                # -1 (too short)
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"7\n0\n0\n-1\n-1\n"


def test_stdlib_contains(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_contains("hello, world", "world"))\n'   # true
        '  println(str_contains("hello", "xyz"))\n'            # false
        '  println(str_contains("hello", ""))\n'               # true
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\ntrue\n"


def test_bytes_to_str_single_chunk(tmp_path):
    # Build a short byte buffer, flatten it — fits in one chunk.
    src = (
        'fn main() -> i32 {\n'
        '  mut buf: tablets[64]u8\n'
        '  buf.push(72 as u8)\n'   # H
        '  buf.push(105 as u8)\n'  # i
        '  println(bytes_to_str(buf))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"Hi\n"


def test_bytes_to_str_empty(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  mut buf: tablets[64]u8\n'
        '  step s = bytes_to_str(buf)\n'
        '  println(s.len)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n"


def test_bytes_to_str_spans_chunks(tmp_path):
    # Chunk size is 4 here, push 10 bytes — forces the intrinsic to
    # walk at least three nodes and memcpy each chunk's used bytes.
    src = (
        'fn main() -> i32 {\n'
        '  mut buf: tablets[4]u8\n'
        '  mut i: i64 = 0\n'
        '  while i < 10 {\n'
        '    buf.push((65 + i) as u8)\n'   # A..J
        '    i = i + 1\n'
        '  }\n'
        '  println(bytes_to_str(buf))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"ABCDEFGHIJ\n"


def test_bytes_to_str_rejects_non_u8():
    with pytest.raises(CompileError, match="tablets\\[N\\]u8"):
        compile_to_ir(
            'fn main() -> i32 {\n'
            '  mut t: tablets[4]i64\n'
            '  t.push(1)\n'
            '  println(bytes_to_str(t))\n'
            '  0\n'
            '}\n'
        )


def test_str_repeat_linear_on_large_n(tmp_path):
    # Repeat a 10-byte string 5000 times — ~50KB of output. A
    # quadratic implementation would take multiple seconds and
    # allocate gigabytes; the linear tablets-backed version should
    # finish under a second. We only assert correctness here; the
    # stress is a bounded smoke check.
    src = (
        'fn main() -> i32 {\n'
        '  step s = str_repeat("0123456789", 5000)\n'
        '  println(s.len)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"50000\n"


def test_stdlib_repeat(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_repeat("ab", 3))\n'
        '  println(str_repeat("x", 0))\n'        # empty
        '  println(str_repeat("-", 5))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"ababab\n\n-----\n"


def test_stdlib_repeat_no_leak(tmp_path):
    # str_repeat does O(n) allocations per call. Call it many times;
    # every intermediate must free at scope exit.
    src = (
        'fn main() -> i32 {\n'
        '  mut i: i64 = 0\n'
        '  while i < 200 {\n'
        '    println(str_repeat("ab", 5))\n'
        '    i = i + 1\n'
        '  }\n'
        '  0\n'
        '}\n'
    )
    rc, out, _ = run_with_stdlib(src, tmp_path)
    assert rc == 0
    assert out.count(b"ababababab\n") == 200


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
