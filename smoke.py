"""Stage-1 pipeline smoke test.

Emits LLVM IR for the smallest meaningful program:

    define i32 @main() { ret i32 42 }

Compiles it to a native aarch64 object file via llvmlite, links it to a
real Mach-O executable via clang, runs it, and checks the exit code.
If this passes, the llvmlite -> object -> clang -> binary pipeline
works end to end on this machine, and the rest of the compiler can be
built on top of it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llvmlite import binding as llvm
from llvmlite import ir


def build_ir() -> ir.Module:
    module = ir.Module(name="tuppu_smoke")
    module.triple = llvm.get_default_triple()
    main_fn = ir.Function(module, ir.FunctionType(ir.IntType(32), []), name="main")
    builder = ir.IRBuilder(main_fn.append_basic_block(name="entry"))
    builder.ret(ir.Constant(ir.IntType(32), 42))
    return module


def emit_object(module: ir.Module, out_path: Path) -> None:
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    ref = llvm.parse_assembly(str(module))
    ref.verify()
    tm = llvm.Target.from_default_triple().create_target_machine(reloc="pic")
    out_path.write_bytes(tm.emit_object(ref))


def link(obj: Path, binary: Path) -> None:
    subprocess.run(["clang", str(obj), "-o", str(binary)], check=True)


def main() -> int:
    build = Path("build")
    build.mkdir(exist_ok=True)
    ir_path, obj_path, bin_path = build / "smoke.ll", build / "smoke.o", build / "smoke"

    module = build_ir()
    ir_path.write_text(str(module))
    emit_object(module, obj_path)
    link(obj_path, bin_path)

    result = subprocess.run([str(bin_path)])
    print(f"ir:     {ir_path}")
    print(f"object: {obj_path}")
    print(f"binary: {bin_path}")
    print(f"exit:   {result.returncode}")

    if result.returncode != 42:
        print(f"FAIL: expected 42, got {result.returncode}")
        return 1
    print("OK: pipeline works end to end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
