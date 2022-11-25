src := test.rush
out := test

test: $(src)
	cargo r $(src)
	riscv64-linux-gnu-as output.s -o output.o
	riscv64-linux-gnu-ld output.o -L corelib -lcore-rush-riscv-lp64d -static -nostdlib -o $(out)

run: test
	qemu-riscv64 ./$(out)
