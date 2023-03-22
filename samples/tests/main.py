#!/bin/python3
import subprocess
import os
import sys

# maps an input file to a desired exit-code
tests = {
    './basic/complete.rush': 50,
    './basic/loops.rush': 45,
    './basic/float_casts.rush': 2,
    './basic/pow.rush': 26,
    './basic/fib.rush': 110,
    './basic/globals.rush': 44,
    './basic/if_else.rush': 20,
    './basic/nan.rush': 11,
    './basic/char.rush': 100,
    './basic/wrapping.rush': 42,
    './basic/exit_0.rush': 0,
    './basic/blocks.rush': 20,
    './basic/nested_calls.rush': 21,
    './basic/approx_pi.rush': 0,
    './basic/approx_e.rush': 0,
    './basic/approx_apery.rush': 0,
    './basic/wasm_test.rush': 37,
    './basic/x64_test.rush': 170,
    # 'evil exits' test if the `!` type can occur everywhere
    './exits/infix.rush': 5,
    './exits/if_else.rush': 6,
    './exits/nested_exit.rush': 7,
    './exits/while.rush': 8,
    './exits/for_1.rush': 9,
    './exits/for_2.rush': 10,
    './exits/for_3.rush': 11,
    './exits/logical_or.rush': 12,
    './exits/logical_and.rush': 13,
    './exits/let.rush': 14,
    './exits/final_fn_expr.rush': 15,
    './exits/in_call.rush': 16,
    # pointers
    './pointers/basic.rush': 42,
    './pointers/assignments.rush': 43,
    './pointers/depth.rush': 121,
    './pointers/globals.rush': 44,
    './pointers/in_params.rush': 45,
    './pointers/types.rush': 46,
    './pointers/swap.rush': 47,
    './pointers/for_loop.rush': 15,
    './pointers/assignment_edge_cases.rush': 104,
    './pointers/as_return_type.rush': 69,
    './pointers/shadow_ref.rush': 42,
}

# saves the backend and any additional commands to be executed after `cargo r`
backends = {
    'rush-interpreter-tree': '',
    'rush-interpreter-vm': '',
    'rush-compiler-wasm': 'exit $(wasmer output.wasm 2>&1 >/dev/null | cut -d ":" -f 4 )',
    'rush-compiler-llvm': 'gcc output.o -o test && ./test',
    'rush-compiler-x86-64': """
        gcc output.s -L corelib/ -lcore -nostdlib -o output \
        && ./output
    """,
    'rush-compiler-risc-v': """
        riscv64-linux-gnu-as output.s -o output.o \
        && riscv64-linux-gnu-ld output.o -L corelib -lcore-rush-riscv-lp64d -static -nostdlib -no-relax -o test \
        && qemu-riscv64 ./test
    """,
    'rush-transpiler-c': 'gcc output.c -o out && ./out',
}


def run():
    failed = []
    tests_ran = 0

    backends_ran = set()
    failed_backends = set()

    for name, cmd in backends.items():
        if not name.endswith(sys.argv[2] if len(sys.argv) == 3 else ''):
            continue
        backends_ran.add(name)
        os.chdir(f'../../crates/{name}')
        for file, code in tests.items():
            tests_ran += 1
            if not run_test(file, code, name, cmd):
                failed_backends.add(name)
                failed += [[name, file]]

    print('=== Summary ===')
    if tests_ran == 0:
        print('    => WARNING: no tests executed')
    else:
        print(f'    => {len(failed)} of {tests_ran} test(s) failed')
    for name, _ in backends.items():
        if name in backends_ran:
            if name in failed_backends:
                print(f'    \x1b[1;31mFAIL\x1b[1;0m: {name}')
            else:
                print(f'    \x1b[1;32mPASS\x1b[1;0m: {name}')
        else:
            print(f'    \x1b[2mSKIP\x1b[1;0m: {name}')

    if failed:
        sys.exit(1)


def run_test(file: str, code: int, name: str, cmd: str):
    print(f'\x1b[2m\x1b[1;39mRUNNING\x1b[1;0m: {file.ljust(15)} {name}')
    command = 'cargo r ../../samples/tests/' + file
    if cmd:
        command += '&&' + cmd

    p = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print('\x1b[1A', end='')

    if p.returncode != code:
        print(
            f'\x1b[1;31mFAIL\x1b[1;0m: {file}/{name} => expected {code}, got {p.returncode}'
        )
        return False

    print(f'\x1b[1;32mPASS\x1b[1;0m: {file.ljust(15)} {name} {" " * 10}')
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            f'Expected at least one, at most two arguments, got {len(sys.argv) - 1}'
        )
        exit(1)

    if sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'count-tests':
        print(len(tests))
    else:
        print(f'Invalid command-line argument: {sys.argv[1]}')
        exit(1)
