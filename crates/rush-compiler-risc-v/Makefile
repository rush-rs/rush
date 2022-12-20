src := output.s
out := test

build: src
	riscv64-linux-gnu-as -g $(src) -o output.o
	riscv64-linux-gnu-ld output.o -L corelib -lcore-rush-riscv-lp64d -static -nostdlib -no-relax -o $(out)

run: build
	qemu-riscv64 ./$(out)
