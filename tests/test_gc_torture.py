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
  step _push = out.push(copy t.label)
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
  step _a = row1.push("alpha")
  step _b = row1.push("beta")
  step _r1 = g1.push(row1)
  mut row2: tablets[4]str
  step _c = row2.push("gamma")
  step _r2 = g1.push(row2)
  step _g = groups.push(g1)
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
  step _m1 = box1.msgs.push(Text("hi"))
  step _m2 = box1.msgs.push(Empty)
  step _m3 = box1.msgs.push(Text("bye"))
  step _i = inboxes.push(box1)

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
    step _p = store.push(Row { label: "r" + int_to_str(i), n: i })
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
  step _o1 = t1.ops.push(OAdd(10))
  step _o2 = t1.ops.push(OMul(3))
  step _o3 = t1.ops.push(OSub(5))
  step _t1 = tapes.push(t1)

  mut t2: Tape = Tape { name: "t" + "2", ops: tablets[16]Op { } }
  step _o4 = t2.ops.push(OAdd(100))
  step _o5 = t2.ops.push(OSub(1))
  step _t2 = tapes.push(t2)

  for tape in tapes {
    println(tape.name, " = ", run_tape(tape))
  }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"t1 = 25\nt2 = 99\n"
