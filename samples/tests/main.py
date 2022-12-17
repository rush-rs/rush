import subprocess
import os
import sys


# maps an input file to a desired exit-code
tests = {
    './complete.rush': 50,
    './loops.rush': 45,
    './float_casts.rush': 2,
    './pow.rush': 26,
    './fib.rush': 110,
    './globals.rush': 44,
    './if_else.rush': 20,
    './nan.rush': 11,
    './char.rush': 100,
    './wrapping.rush': 42,
    './exit_0.rush': 0,
    # 'evil exits' test if the `!` type can occur everywhere
    './exits/infix.rush': 5,
    './exits/if_else.rush': 6,
    './exits/calls.rush': 7,
    './exits/while.rush': 8,
    './exits/for_1.rush': 9,
    './exits/for_2.rush': 10,
    './exits/for_3.rush': 11,
    './exits/logical_or.rush': 12,
    './exits/logical_and.rush': 13,
    './exits/let.rush': 14,
    './exits/final_fn_expr.rush': 15,
}

# saves the backend an any additional commands to be executed after `cargo r`
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
}


def run():
    failed = []
    for name, cmd in backends.items():
        if not name.endswith(sys.argv[1] if len(sys.argv) >= 2 else ''):
            continue
        os.chdir(f'../../crates/{name}')
        for file, code in tests.items():
            if not run_test(file, code, name, cmd):
                failed += [[name, file]]

    if failed:
        print(f'=== {len(failed)} test(s) failed ===')
        for test in failed:
            print(f'    {test[1].ljust(15)} {test[0]}')
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
    run()
