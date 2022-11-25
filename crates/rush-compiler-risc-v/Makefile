src := samples/test.rush
out := test

run: $(src) src
	cargo r $(src)
	riscv64-linux-gnu-as output.s -o output.o
	riscv64-linux-gnu-ld output.o -L corelib -lcore-rush-riscv-lp64d -static -nostdlib -no-relax -o $(out)
	qemu-riscv64 ./$(out)
