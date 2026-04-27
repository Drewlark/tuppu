"""GC torture tests — the cursed compositions.

Each test stresses a specific shape that historically tripped the
collector or the rooting machinery: deeply nested seals, tablets of
structs of seals, mutually recursive cycles via wedge indirection,
loops with heavy intra-iteration allocation, etc. The lua interpreter
example exercises most of these in combination; this file isolates
each shape so a future regression points at exactly one cursed
composition instead of at 1500 lines of interpreter.

Each runs in both normal mode and `TUPPU_GC_STRESS=1` mode (via the
parametrized `stress` fixture below). Stress mode forces a collect
on every allocation so any missed root shows up immediately.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_files_to_binary, stdlib_files


@pytest.fixture(params=[False, True], ids=["normal", "stress"])
def stress(request):
    return request.param


def run(src: str, tmp_path: Path, stress: bool) -> tuple[int, bytes]:
    user = tmp_path / "main.tpu"
    user.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user], tmp_path, name="prog",
    )
    env = dict(os.environ)
    if stress:
        env["TUPPU_GC_STRESS"] = "1"
    r = subprocess.run([str(binary)], capture_output=True, env=env)
    return r.returncode, r.stdout


# --- recursive structures via wedge indirection --------------------------


def test_binary_tree_via_wedges(tmp_path, stress):
    # Tree-of-strs: each node has two wedge children. Build a small
    # tree, in-order traverse. Recursive str + wedge composition that
    # used to trip the chunk trace_fn before the fix landed.
    src = """
tablet Tree { label: str, left: wedge Tree, right: wedge Tree }

fn inorder(t: wedge Tree, mut out: tablets[64]str) {
  if t == lost { yield }
  inorder(t.left, out)
  out.push(copy t.label)
  inorder(t.right, out)
}

fn main() -> i32 {
  mut nodes: tablets[16]Tree
  step l1 = nodes.push(Tree { label: "1", left: lost, right: lost })
  step l3 = nodes.push(Tree { label: "3", left: lost, right: lost })
  step n2 = nodes.push(Tree { label: "2", left: l1, right: l3 })
  step l5 = nodes.push(Tree { label: "5", left: lost, right: lost })
  step n4 = nodes.push(Tree { label: "4", left: n2, right: l5 })

  mut out: tablets[64]str
  inorder(n4, out)
  for s in out { println(s) }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"1\n2\n3\n4\n5\n"


def test_seal_tree_via_tablet_indirection(tmp_path, stress):
    # `seal Expr { Lit(i64), Add(wedge ExprNode, wedge ExprNode) }` —
    # the canonical AST shape, with tablets-resident nodes so the
    # seal's wedge fields never recur into themselves directly.
    src = """
seal Expr { Lit(i64), Add(wedge ExprNode, wedge ExprNode) }
tablet ExprNode { e: Expr }

fn eval(n: wedge ExprNode) -> i64 {
  match n.e {
    Lit(v) => v,
    Add(l, r) => eval(l) + eval(r),
  }
}

fn main() -> i32 {
  mut nodes: tablets[16]ExprNode
  // (1 + 2) + (3 + 4) = 10
  step a = nodes.push(ExprNode { e: Lit(1) })
  step b = nodes.push(ExprNode { e: Lit(2) })
  step c = nodes.push(ExprNode { e: Lit(3) })
  step d = nodes.push(ExprNode { e: Lit(4) })
  step ab = nodes.push(ExprNode { e: Add(a, b) })
  step cd = nodes.push(ExprNode { e: Add(c, d) })
  step root = nodes.push(ExprNode { e: Add(ab, cd) })
  println(eval(root))
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"10\n"


# --- nested containers ---------------------------------------------------


def test_tablets_of_tablets_of_str(tmp_path, stress):
    # Three levels deep — exercises chunk-of-tablets-of-tablets
    # tracing. Each push at any level allocates; under stress every
    # alloc is a collect, so any missed root surfaces.
    src = """
fn main() -> i32 {
  mut groups: tablets[4]tablets[4]tablets[4]str
  mut g1: tablets[4]tablets[4]str
  mut row1: tablets[4]str
  row1.push("alpha")
  row1.push("beta")
  g1.push(row1)
  mut row2: tablets[4]str
  row2.push("gamma")
  g1.push(row2)
  groups.push(g1)
  println(groups[0][0][0])
  println(groups[0][0][1])
  println(groups[0][1][0])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"alpha\nbeta\ngamma\n"


def test_struct_with_tablets_of_seals_with_str(tmp_path, stress):
    # struct -> tablets -> seal -> str. The seal payload is the bit
    # the chunk trace_fn fix unlocked; we exercise it inside a
    # struct field for good measure.
    src = """
seal Msg { Text(str), Empty }
tablet Inbox { name: str, msgs: tablets[8]Msg }

fn main() -> i32 {
  mut inboxes: tablets[4]Inbox
  mut box1: Inbox = Inbox { name: "alice", msgs: tablets[8]Msg { } }
  box1.msgs.push(Text("hi"))
  box1.msgs.push(Empty)
  box1.msgs.push(Text("bye"))
  inboxes.push(box1)
  step who = inboxes[0].name
  println(who)
  for m in inboxes[0].msgs {
    match m {
      Text(t) => println(t),
      Empty => println("(empty)"),
    }
  }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"alice\nhi\n(empty)\nbye\n"


# --- nested seals --------------------------------------------------------


def test_nested_seal_str_payload(tmp_path, stress):
    # `seal Outer { Wrap(Inner), Stop }` where Inner has a str.
    # Trace_fn must recurse into the inner seal's trace.
    src = """
seal Inner { Hello(str), Bye(str) }
seal Outer { Wrap(Inner), Stop }

fn show(o: Outer) {
  match o {
    Wrap(i) => match i {
      Hello(s) => println("hello, ", s),
      Bye(s) => println("bye, ", s),
    },
    Stop => println("stop"),
  }
}

fn main() -> i32 {
  step a: Outer = Wrap(Hello("world"))
  step b: Outer = Wrap(Bye("then"))
  step c: Outer = Stop
  show(a)
  show(b)
  show(c)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"hello, world\nbye, then\nstop\n"


def test_seal_inside_struct_inside_seal(tmp_path, stress):
    # Outer seal carries a struct payload whose field is a nested
    # seal carrying a heap str. Three layers of indirection.
    src = """
seal Status { Ok(str), Err(str) }
tablet Frame { name: str, status: Status }
seal Result { Done(Frame), Pending }

fn show(r: Result) {
  match r {
    Done(f) => {
      println("frame: ", f.name)
      match f.status {
        Ok(s) => println("  ok: ", s),
        Err(s) => println("  err: ", s),
      }
    },
    Pending => println("pending"),
  }
}

fn main() -> i32 {
  step a: Result = Done(Frame { name: "build", status: Ok("hi" + "!") })
  step b: Result = Done(Frame { name: "test",  status: Err("bo" + "om") })
  step c: Result = Pending
  show(a)
  show(b)
  show(c)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"frame: build\n  ok: hi!\nframe: test\n  err: boom\npending\n"


# --- repeated allocation under stress ------------------------------------


def test_concat_in_long_loop(tmp_path, stress):
    # 200 iterations of str_concat — under stress every concat
    # triggers a collect with the loop carry alive on the shadow
    # stack. If acc's bytes ever go unrooted across the alloc, the
    # output corrupts.
    src = """
fn main() -> i32 {
  mut acc: str = ""
  mut i: i64 = 0
  while i < 200 {
    acc = acc + "."
    i = i + 1
  }
  println(acc.len)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"200\n"


def test_push_then_get_then_clone_in_loop(tmp_path, stress):
    # Long-running mutation with clones-on-return interleaved.
    src = """
tablet Row { label: str, n: i64 }

fn copy_row(r: Row) -> Row { r }

fn main() -> i32 {
  mut store: tablets[64]Row
  mut i: i64 = 0
  while i < 50 {
    store.push(Row { label: "r" + int_to_str(i), n: i })
    i = i + 1
  }
  // Pull row 7 out, clone via the by-value pass through `copy_row`,
  // verify both fields survive the clone path.
  step r = copy_row(store[7])
  println(r.label)
  println(r.n)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"r7\n7\n"


def test_match_binder_used_after_mutation(tmp_path, stress):
    # Match binder on a heap-bearing payload, parser-style. Binder
    # must survive a mutation of the scrutinee's source via implicit
    # copy.
    src = """
seal Tok { Ident(str), EOF }
tablet Lex { cur: Tok }

fn bump(mut l: Lex) {
  l.cur = EOF
}

fn main() -> i32 {
  mut l: Lex = Lex { cur: Ident("hel" + "lo") }
  match l.cur {
    Ident(name) => {
      bump(l)
      println(name)
    },
    EOF => println("eof"),
  }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"hello\n"


# --- the cursed composition ---------------------------------------------


def test_kitchen_sink(tmp_path, stress):
    # tablets of structs with str + tablets-of-seals fields, mutated
    # in a loop, returned by value through a fn boundary, deep-cloned
    # implicitly. Roughly the lua interpreter in miniature.
    src = """
seal Op { OAdd(i64), OSub(i64), OMul(i64) }
tablet Tape { name: str, ops: tablets[16]Op }

fn run_tape(t: Tape) -> i64 {
  mut acc: i64 = 0
  for op in t.ops {
    match op {
      OAdd(n) => { acc = acc + n },
      OSub(n) => { acc = acc - n },
      OMul(n) => { acc = acc * n },
    }
  }
  acc
}

fn main() -> i32 {
  mut tapes: tablets[8]Tape
  mut t1: Tape = Tape { name: "t" + "1", ops: tablets[16]Op { } }
  t1.ops.push(OAdd(10))
  t1.ops.push(OMul(3))
  t1.ops.push(OSub(5))
  tapes.push(t1)
  mut t2: Tape = Tape { name: "t" + "2", ops: tablets[16]Op { } }
  t2.ops.push(OAdd(100))
  t2.ops.push(OSub(1))
  tapes.push(t2)
  for tape in tapes {
    println(tape.name, " = ", run_tape(tape))
  }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"t1 = 25\nt2 = 99\n"

# --- regression tracking -------------------------------------------------

def test_struct_alignment_gc_offsets(tmp_path, stress):
    # Exercises the padding bug in `_size_of_ty`. An i8 followed by an i64
    # and a str pointer forces LLVM to insert alignment padding. If the
    # GC computes offsets strictly by summing element sizes, it marks
    # garbage bytes.
    src = """
tablet Padded { a: i8, b: i64, c: str }
fn main() -> i32 {
  mut store: tablets[4]Padded
  store.push(Padded { a: 1, b: 2, c: "aligned" + "!" })
  // Force a collect while the Padded struct sits in the chunk
  mut acc: str = ""
  mut i: i64 = 0
  while i < 10 {
    acc = acc + "."
    i = i + 1
  }

  println(store[0].c)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"aligned!\n"


def test_shadowed_variable_cleanup_eviction(tmp_path, stress):
    # Stresses `_transfer_cleanup_into_container` traversal direction.
    # If it iterates outermost-first, the outer `s` loses its cleanup
    # registration, gets swept by the allocation loop, and causes a UAF.
    src = """
fn main() -> i32 {
  mut store: tablets[4]str
  step s = "outer" + "_str"
  {
    step s = "inner" + "_str"
    step _ = store.push(s)
  }

  mut acc: str = ""
  mut i: i64 = 0
  while i < 10 {
    acc = acc + "."
    i = i + 1
  }

  println(s)
  println(store[0])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"outer_str\ninner_str\n"


def test_field_return_survives_caller_gc_pressure(tmp_path, stress):
    # Returning a Field expression from a local struct: the caller
    # forces multiple GC cycles before reading the result. The GC
    # must keep the returned str's bytes alive via the str type
    # descriptor (str.ptr → heap byte buffer), even though the local
    # struct that originally owned the field is gone.
    src = """
tablet Row { label: str }

fn build() -> str {
  step r: Row = Row { label: "abc" + "def" }
  r.label
}

fn main() -> i32 {
  step s = build()
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "."
    i = i + 1
  }
  println(s)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"abcdef\n"


def test_index_return_from_local_tablets_survives_gc(tmp_path, stress):
    # Returning a tablets-Index value: the local tablets is unreachable
    # after the function returns, but the str at slot 0 still has a
    # live byte-buffer that the caller's binding keeps rooted.
    src = """
fn build() -> str {
  mut local: tablets[4]str
  local.push("first" + "_str")
  local.push("second" + "_str")
  local[0]
}

fn main() -> i32 {
  step s = build()
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "."
    i = i + 1
  }
  println(s)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"first_str\n"


def test_field_return_from_match_arm_survives_gc(tmp_path, stress):
    # The Field return is inside a match arm; the arm's binding goes
    # out of scope along with the local scrutinee at fn return, but
    # the returned str must stay alive in the caller through GC.
    src = """
seal Pick { First, Second }
tablet Row { label: str }

fn pick(p: Pick) -> str {
  step r: Row = Row { label: "winner" + "!" }
  match p {
    First => r.label,
    Second => "other",
  }
}

fn main() -> i32 {
  step s = pick(First)
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "."
    i = i + 1
  }
  println(s)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"winner!\n"


def test_nested_field_return_survives_gc(tmp_path, stress):
    # Returning a Field-of-Field — outer.inner.label. Both the outer
    # and inner structs are local; only the str's byte buffer must
    # survive in the caller.
    src = """
tablet Inner { label: str }
tablet Outer { inner: Inner }

fn build() -> str {
  step o: Outer = Outer { inner: Inner { label: "deep" + "_str" } }
  o.inner.label
}

fn main() -> i32 {
  step s = build()
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "."
    i = i + 1
  }
  println(s)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"deep_str\n"


def test_anonymous_rvalue_block_tail_uaf(tmp_path, stress):
    # Exercises the `.rvalue.root` double-free/UAF.
    # `make_box().s` registers an anonymous cleanup for the Box.
    # If the block tail doesn't explicitly consume the rvalue cleanup,
    # the frame pop frees the source bytes immediately before assignment.
    src = """
tablet Box { s: str }
fn make_box() -> Box {
  Box { s: "heap" + "_string" }
}
fn main() -> i32 {
  step s = {
     make_box().s
  }

  mut acc: str = ""
  mut i: i64 = 0
  while i < 10 {
    acc = acc + "."
    i = i + 1
  }

  println(s)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"heap_string\n"

def test_yield_mid_expression_unwind(tmp_path, stress):
    # Registers a heap-str step, then yields with another heap-str value.
    # The yield unwinder must sweep the named `leaked` cleanup frame entry
    # and hand off the anonymous `"escaped" + "!"` temp as the return value
    # without double-freeing either.
    src = """
fn bailout() -> str {
  step leaked = "leak" + "1"
  if true { yield "escaped" + "!" }
  "never"
}

fn main() -> i32 {
  println(bailout())
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"escaped!\n"


def test_match_implicit_clone_shadow_stack(tmp_path, stress):
    # Pattern binders on cleanup-bearing payloads implicitly deep-clone
    # so the scrutinee can be safely mutated inside the arm. This test
    # ensures the implicit clone is properly registered as a GC root
    # in the arm's cleanup frame and survives a forced collection.
    src = """
seal Msg { S(str), Empty }

fn main() -> i32 {
  mut m: Msg = S("hello" + "_world")
  match m {
    S(text) => {
      // Force a GC collect inside the arm
      mut acc: str = ""
      mut i: i64 = 0
      while i < 10 {
        acc = acc + "."
        i = i + 1
      }
      println(text)
    },
    Empty => { }
  }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"hello_world\n"


def test_tablets_circular_wedge_assignment(tmp_path, stress):
    # Creating a cycle by assigning a wedge into the tablets it belongs to.
    # While tuppu forbids by-value cycles, wedge cycles are valid. The GC
    # must be able to trace this without infinite recursion or stack overflows.
    src = """
tablet Cell { next: wedge Cell, val: str }

fn main() -> i32 {
  mut store: tablets[4]Cell
  step w1 = store.push(Cell { next: lost, val: "a" + "1" })
  step w2 = store.push(Cell { next: w1, val: "b" + "2" })

  // Close the loop
  w1.next = w2

  // Force GC
  mut acc: str = ""
  mut i: i64 = 0
  while i < 10 {
    acc = acc + "."
    i = i + 1
  }

  println(w1.next.val)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"b2\n"


# --- smart wedges: wedge escapes keep the source arena alive --------------
#
# The v0.4.1 soundness bug: a wedge that escaped via a wrapping struct or
# seal payload kept no GC trace back to its source arena. Under stress the
# arena got swept while the caller still held the wedge, producing silent
# zero-reads. The fix is interior-pointer marking via __tuppu_gc_mark_wedge,
# emitted by trace fns for any field declared as `wedge T`. Each test below
# escapes a wedge through a different wrapper shape and reads the wedge
# after heavy collection pressure; under the bug, all of them read 0.


def test_wedge_in_struct_returned_survives_gc(tmp_path, stress):
    src = """
tablet Tree { value: i64 }
tablet Box { handle: wedge Tree }

fn make() -> Box {
  mut nodes: tablets[16]Tree
  step h = nodes.push(Tree { value: 42 })
  Box { handle: h }
}

fn main() -> i32 {
  step b = make()
  // Force collection pressure unrelated to the boxed wedge.
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "x"
    i = i + 1
  }
  println(b.handle.value)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"42\n"


def test_wedge_in_seal_payload_returned_survives_gc(tmp_path, stress):
    src = """
tablet Cell { value: i64 }
seal Holder { Empty, Some(wedge Cell) }

fn make() -> Holder {
  mut cells: tablets[16]Cell
  step h = cells.push(Cell { value: 7 })
  Some(h)
}

fn main() -> i32 {
  step holder = make()
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "y"
    i = i + 1
  }
  match holder {
    Empty => println(0),
    Some(h) => println(h.value),
  }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"7\n"


def test_tablets_of_wedge_returned_survives_gc(tmp_path, stress):
    # tablets[N]wedge T — each chunk slot is a wedge into another arena.
    # The chunk descriptor must dispatch each slot through mark_wedge.
    src = """
tablet Item { v: i64 }

fn build() -> tablets[8]wedge Item {
  mut store: tablets[16]Item
  mut handles: tablets[8]wedge Item
  handles.push(store.push(Item { v: 11 }))
  handles.push(store.push(Item { v: 22 }))
  handles.push(store.push(Item { v: 33 }))
  handles
}

fn main() -> i32 {
  mut hs = build()
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "z"
    i = i + 1
  }
  for h in hs { println(h.v) }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"11\n22\n33\n"


def test_wedge_in_nested_struct_returned_survives_gc(tmp_path, stress):
    # Wedge two levels deep: outer struct holds an inner struct that holds
    # the wedge. Trace fn must recurse through the inner struct's fields.
    src = """
tablet Leaf { tag: i64 }
tablet Inner { ref: wedge Leaf }
tablet Outer { inner: Inner }

fn make() -> Outer {
  mut leaves: tablets[8]Leaf
  step h = leaves.push(Leaf { tag: 99 })
  Outer { inner: Inner { ref: h } }
}

fn main() -> i32 {
  step o = make()
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "q"
    i = i + 1
  }
  println(o.inner.ref.tag)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"99\n"