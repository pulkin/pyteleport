"""
Extends opcode collections.
"""
import sys
from dis import opmap

locals().update(opmap)  # unpack opcodes here
python_version = sys.version_info.major * 0x100 + sys.version_info.minor

# These unconditionally interrupt the normal bytecode flow
interrupting = tuple(
    opmap[i]
    for i in (
        "JUMP_ABSOLUTE",
        "JUMP_FORWARD",
        "RETURN_VALUE",
        "RAISE_VARARGS",
        "RERAISE",  # 3.9+
    )
    if i in opmap
)
resuming = tuple(
    opmap[i]
    for i in (
        "GEN_START",  # 3.10+
    )
    if i in opmap
)
del opmap  # cleanup


def guess_entering_stack_size(opcode: int) -> int:
    """
    Figure out the starting stack size given the starting opcode.
    This usually returns zero, except the special GEN_START case when it is one.

    Parameters
    ----------
    opcode
        The starting opcode.

    Returns
    -------
    The stack size.
    """
    return int(opcode in resuming)
