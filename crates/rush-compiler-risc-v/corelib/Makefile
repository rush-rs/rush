src := pow exit cast
obj := pow.o exit.o cast.o
dest := libcore-rush-riscv

# default RISC-V ABI
abis := lp64 lp64d

assembler := riscv64-linux-gnu-as
archiver := riscv64-linux-gnu-ar
gcc := riscv64-linux-gnu-gcc

# creates an `.a` file of the entire corelib
libcore.a: src/pow.s src/exit.s src/cast.s
	for abi in $(abis); do \
		for file in $(src); do \
			$(assembler) -mabi=$$abi src/$$file.s -o $$file.o; \
		done; \
		$(archiver) -cq $(dest)-$$abi.a $(obj); \
		rm $(obj); \
	done

all: clean libcore.a

clean:
	rm -f *.a

# runs `main.c` alongside the corelib in order to test it
test: libcore.a
	$(gcc) -nostdlib -static main.c -L . -lcore-rush-riscv-lp64d -o main
	qemu-riscv64 ./main
