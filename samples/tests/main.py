import subprocess
import os
import time


# maps an input file to a desired exit-code
tests = {'complete.rush': 50, 'loops.rush': 80}

# saves the backend an any additional commands to be executed after `cargo r`
backends = {
    'rush-interpreter-tree': '',
    'rush-interpreter-vm': '',
    #'rush-compiler-wasm': 'wasmer output.wasm',
    'rush-compiler-llvm': 'gcc output.o -o test && ./test',
    'rush-compiler-x86-64': """
        gcc output.s -L corelib/ -lcore -nostdlib -o test \
        && ./test
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
        os.chdir(f'../../crates/{name}')
        for file, code in tests.items():
            if not run_test(file, code, name, cmd):
                failed += [[name, file]]

    if failed:
        print(f'=== {len(failed)} test(s) failed ===')
        for test in failed:
            print(f'    {test[1].ljust(15)} {test[0]}')


def run_test(file: str, code: int, name: str, cmd: str):
    command = 'cargo r ../../samples/tests/' + file
    if cmd:
        command += '&&' + cmd

    p = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if p.returncode != code:
        print(
            f'\x1b[1;31mFAIL\x1b[1;0m: {file}/{name} => expected {code}, got {p.returncode}'
        )
        return False

    print(f'\x1b[1;32mPASS\x1b[1;0m: {file.ljust(15)} {name}')
    return True


if __name__ == '__main__':
    run()
