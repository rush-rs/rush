src := output.s
out := test

build: src
	s390x-unknown-linux-gnu-as  -g $(src) -o output.o
	s390x-unknown-linux-gnu-ld output.o -L corelib -lcore-rush-s390x -static -nostdlib -no-relax -o $(out)

run: build
	qemu-s390x ./$(out); echo "exit: $$?"

load: ./output.s
	echo "UPLOADING..."
	scp -v ./output.s lnx-one:~/test0/a.s
	echo "UPLOAD COMPLETED"
