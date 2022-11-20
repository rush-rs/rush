libcore.a: src/corelib.s
	as --64 src/corelib.s -o corelib.o
	! [ -f libcore.a ] || rm libcore.a
	ar -vq libcore.a corelib.o
