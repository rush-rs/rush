# This file is part of the rush corelib for IBM System/390x
# - Authors: MikMuellerDev
# This file, alongside others is linked to form an IBM System/390x targeted rush program.

# Calculates the nth power of the specified base.
# If the exponent is 0, the result is 1.
# Furthermore, if the exponent is < 0, the result is 0 (simulates truncated float).
# fn __rush_internal_pow_int(base: int, exp: int) -> int
.globl __rush_internal_pow_int

__rush_internal_pow_int:
	# base in %r2, exp in %r3
	# NOTE: r4 is used as the acc register.
	# NOTE: r5 is used as a temp register.

	# Prologue.
	aghi %r15, -40
	stg	%r4, 0(%r15)
	stg	%r5, 8(%r15)
	stg	%r14, 16(%r15)

	# More scrap registers (division)
	stg	%r6, 24(%r15)
	stg	%r8, 32(%r15)

	# if exp == 0; return 1
	chi	%r3, 0
	je return_1

	# if exp < 0; return 0
	chi %r3, 0
	jl return_0

	lghi %r4, 1			# acc = 1
	lgr %r5, %r3		# iterations remaining

	pow_loop_head:
		chi %r5, 0
		je pow_loop_end

	pow_loop_body:
		aghi %r5, -1    # decr iter
		msgr %r4, %r2   # incr acc

		j pow_loop_head

	pow_loop_end:
		lgr %r2, %r4
		j pow_ret

	return_0:
		lghi %r2, 0
		j pow_ret

	return_1:
		lghi %r2, 1
		j pow_ret

	pow_ret:
		# Epilogue.
		lg %r4,	0(%r15)
		lg %r5,	8(%r15)
		lg %r14,16(%r15)
		lg %r6,24(%r15)
		lg %r8,32(%r15)
		aghi %r15,	40
		br %r14
