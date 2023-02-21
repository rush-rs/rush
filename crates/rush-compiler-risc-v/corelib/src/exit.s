# This file is part of the rush corelib for RISC-V
# - Authors: MikMuellerDev
# This file, alongside others is linked to form a RISC-V targeted rush program.

# Calls the Linux kernel to exit using the specified argument.
# Any code after this function call is unreachable, therefore its return type is !.
# fn exit(code: int) -> !
.global exit

exit:
	li a7, 93 # syscall type is `exit`
	ecall     # exit code is already in `a0`
