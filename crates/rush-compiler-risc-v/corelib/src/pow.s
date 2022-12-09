# This file is part of the rush corelib for RISC-V
# - Authors: MikMuellerDev
# This file, alongside others is linked to form a RISC-V targeted rush program.

# Calculates the nth power of the specified base.
# If the exponent is 0, the result is 1.
# Furthermore, if the exponent is < 0, the result is 0 (simulates truncated float).
# fn __rush_internal_pow_int(base: int, exp: int) -> int
.global __rush_internal_pow_int

__rush_internal_pow_int:
	# if exp == 0; return 1
	beq a1, zero, return_1
	# if exp < 0; return 0
	blt a1, zero, return_0

	li t0, 1			# acc = 1

	loop_head:
		# if exp < 2; { break }
		li t1, 2
		blt a1, t1, after_loop

		# if (exp & 1) == 1
		andi t1, a1, 1
		li t2, 1
		beq t1, t2, inc_acc

		loop_body:
			srli a1, a1, 1		# exp /= 2
			mul a0, a0, a0
			j loop_head			# continue

	inc_acc:
		mul t0, t0, a0
		j loop_body

	after_loop:
		mul a0, t0, a0			# acc * base
		ret

	return_0:
		li a0, 0
		ret

	return_1:
		li a0, 1
		ret
