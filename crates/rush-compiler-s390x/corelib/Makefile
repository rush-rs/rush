src := pow exit cast
obj := pow.o exit.o cast.o
dest_name := core-rush-s390x
dest := lib$(dest_name)

assembler := s390x-unknown-linux-gnu-as
archiver := s390x-unknown-linux-gnu-ar
gcc := s390x-unknown-linux-gnu-gcc

# creates an `.a` file of the entire corelib
libcore.a: src/pow.s src/exit.s src/cast.s
	for file in $(src); do \
		$(assembler) src/$$file.s -o $$file.o; \
	done; \
	$(archiver) -cq $(dest).a $(obj); \
	rm $(obj);

all: clean libcore.a

clean:
	rm -f *.a

# runs `main.c` alongside the corelib in order to test it
test: libcore.a
	$(gcc) -g -Wl,-z -Wl,execstack -nostdlib -static main.c -L . -l$(dest_name) -o main
	qemu-s390x ./main; echo "exit: $$?"
