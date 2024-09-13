# This file is part of the rush corelib for IBM System/390x
# - Authors: MikMuellerDev
# This file, alongside others is linked to form an IBM System/390x targeted rush program.

# Converts a i64 into a u8 char.
# If the source is < 0, the result will be 0.
# Furthermore, if the source is > 127, the result will be 127.
# fn __rush_internal_cast_int_to_char(from: int) -> char
.global __rush_internal_cast_int_to_char

__rush_internal_cast_int_to_char:
	# if from < 0; return 0
	chi %r2, 0
	jl int_return_0
	# if from > 127; return 127
	chi %r2, 127
	jh int_return_127
	br %r14

	int_return_0:
		lghi %r2, 0
		br %r14

	int_return_127:
		lghi %r2, 127
		br %r14


# Converts a f64 into a u8 char.
# If the source is < 0.0, the result will be 0.
# Furthermore, if the source is > 127.0, the result will be 127.
# fn __rush_internal_cast_float_to_char(from: float) -> char
.global __rush_internal_cast_float_to_char

__rush_internal_cast_float_to_char:
	# Save %r5, %r14 here
	ahi %r15, -24
	stg %r5, 0(%r15)
	stg %r14, 8(%r15)

	# load 127 from mem
	larl	%r5, float_127
	lde	%f1,0(%r5)

	ste	 %f1,	16(%r15)
	ldeb %f1,	16(%r15)

	# if from > 127.0; return 127
	kebr %f0, %f1
	jh float_return_127

	# if from < 0.0; return 0
	larl	%r5, float_0		# load 0 here
	keb	%f0, 0(%r5)				# compare %f0 and the memory location where 0 is
	jl float_return_0

	# round / truncate and return
	cfdbr	%r2,5,%f0

	float_return:
		lg %r5, 0(%r15)
		lg %r14, 8(%r15)
		ahi %r15, 24
		br %r14

	float_return_0:
		lghi %r2, 0
		j float_return

	float_return_127:
		lghi %r2, 127
		j float_return



### UTILS ###


.globl main

test_float:
	ahi %r15, -16
	stg %r14, 0(%r15)

	larl	%r13, float_test
	lde	%f0,0(%r13)

	ste	 %f0,	8(%r15)
	ldeb %f0,8(%r15)

	brasl %r14, __rush_internal_cast_float_to_char

	lg %r14, 0(%r15)

	ahi %r15, 16
	br	%r14

test_int:
	ahi %r15, -8
	stg %r14, 0(%r15)


	lghi %r2, -2
	brasl %r14, __rush_internal_cast_int_to_char

	lg %r14, 0(%r15)

	ahi %r15, 8
	br	%r14

main:
	ahi %r15, -8
	stg %r14, 0(%r15)

	# Float.
	# brasl %r14, test_float

	# Int.
	# brasl %r14, test_int

	lg %r14, 0(%r15)

	ahi %r15, 8
	br	%r14


### CONSTANTS ###
.section	.rodata
.align	8
float_0:
	.long 0x0000000
	.align 2

float_127:
	.long 0x0000000042fe0000
	.align 2

float_test:
	.long 0x00000000c3000000 # -128
	.align 2
