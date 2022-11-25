src := pow exit cast
obj := pow.o exit.o cast.o
dest := libcore-rush-riscv

# default RISC-V ABI
abi := lp64d

# creates an `.a` file of the entire corelib
libcore.a: src/pow.s src/exit.s src/cast.s
	for file in $(src); do \
		riscv64-linux-gnu-as -mabi=$(abi) src/$$file.s -o $$file.o; \
	done

	riscv64-linux-gnu-ar -cq $(dest)-$(abi).a $(obj)
	rm $(obj)

all:
	rm -f *.a
	# TODO: implement this differently
	make abi=lp64
	make abi=lp64d

# runs `main.c` alongside the corelib in order to test it
test: libcore.a
	riscv64-linux-gnu-gcc -nostdlib -static main.c -L . -lcore -o main
	qemu-riscv64 ./main
