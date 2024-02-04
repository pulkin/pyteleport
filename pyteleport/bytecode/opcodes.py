"""
Extends opcode collections.
"""
import sys
from dis import opmap

locals().update(opmap)  # unpack opcodes here
_python_version = sys.version_info.major * 0x100 + sys.version_info.minor

"""
Prior to python 3.11 the bytecode representation has actual instructions for exception handling.
When the code enters another "try" clause it executes the SETUP_FINALLY instruction that
pushes exception handling information onto the "block stack". This feature is replaced by a
static representation of exception handling information in 3.11 and above. 
"""
python_feature_block_stack = _python_version <= 0x030A
"""
Since python 3.10 all jump arguments are divided by two as instruction opcodes occupy only
even bytecode offsets. This saves some EXTENDED_ARGs.
"""
python_feature_jump_2x = _python_version >= 0x030A
"""
Python 3.10 introduces a GEN_START no-op instruction. Python 3.11 and above re-works this further.
"""
python_feature_gen_start_opcode = _python_version == 0x030A
"""
Python 3.11 and above introduce bytecode speedup through collecting statistical information ("cache")
about how exactly some bytecode instructions are executed. This has major bytecode implications:
first of all, cache is stored right in the bytecode following some of the instructions. This means
that some instruction occupy more space (as opposed to two bytes per instruction before).
Second, all function calls are now processed through the CALL instruction, (CALL_FUNCTION_EX still avail).
Third, LOAD_GLOBAL falls victim to loading (non-)class methods which mess with NULLs on the value stack
(it was LOAD_METHOD's job prior to this version).
"""
python_feature_pre_call = _python_version >= 0x030B
python_feature_cache = _python_version >= 0x030B
python_feature_load_global_null = _python_version >= 0x030B

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
    for i in ("GEN_START",)
    if i in opmap
)
call_function = tuple(
    i
    for name, i in opmap.items()
    if "CALL_FUNCTION" in name
)
call_method = tuple(
    opmap[i]
    for i in ("CALL", "CALL_METHOD")
    if i in opmap
)
del opmap


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
