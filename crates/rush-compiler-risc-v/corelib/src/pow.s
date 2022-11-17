# This file is part of the rush corelib for RISC-V
# - Authors: MikMuellerDev
# This file, alongside others is linked to form a RISC-V targeted rush program.

# Calculates the nth power of the specified base.
# If the exponent is 0, the result is 1.
# Furthermore, if the exponent is < 0, the result is 0 (simulates truncated float).
# fn __rush_internal_pow_int(base: int, exp: int) -> int
.global __rush_internal_pow_int

__rush_internal_pow_int:
	addi sp, sp, -8		# allocate 8 bytes (for ra)
	sd ra, 0(sp)		# push ra
	addi s0, sp, 8		# increase frame ptr

	# if exp < 0; return 0
	blt a1, zero, return
	j body

	return:
		li t0, 0			# return 0
		j after_loop		# transfer control to `after_loop`

	body:
		li t0, 1			# accumulator = 1

	loop_head:
		# if exp == 0 { break }
		beq a1, zero, after_loop

		mul t0, t0, a0		# accumulator *= base
		addi a1, a1, -1		# exp -= 1
		j loop_head			# continue

	after_loop:
		add a0, zero, t0	# return accumulator
		addi sp, sp, 8		# free stack memory again
		ret
