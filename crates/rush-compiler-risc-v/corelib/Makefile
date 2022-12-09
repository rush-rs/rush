src := pow exit cast
obj := pow.o exit.o cast.o
dest := libcore-rush-riscv

# default RISC-V ABI
abis := lp64 lp64d

# creates an `.a` file of the entire corelib
libcore.a: src/pow.s src/exit.s src/cast.s
	for abi in $(abis); do \
		for file in $(src); do \
			riscv64-linux-gnu-as -mabi=$$abi src/$$file.s -o $$file.o; \
		done; \
		riscv64-linux-gnu-ar -cq $(dest)-$$abi.a $(obj); \
		rm $(obj); \
	done

all: clean libcore.a

clean:
	rm -f *.a

# runs `main.c` alongside the corelib in order to test it
test: libcore.a
	riscv64-linux-gnu-gcc -nostdlib -static main.c -L . -lcore-rush-riscv-lp64d -o main
	qemu-riscv64 ./main
