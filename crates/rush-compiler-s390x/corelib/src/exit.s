# This file is part of the rush corelib for IBM System/390x
# - Authors: MikMuellerDev
# This file, alongside others is linked to form an IBM System/390x targeted rush program.

# Calls the Linux kernel to exit using the specified argument.
# Any code after this function call is unreachable, therefore its return type is !.
# fn exit(code: int) -> !
.global exit

exit:
	# exit code is already in `%r2`
	svc 1
