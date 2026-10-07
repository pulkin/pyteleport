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
Before python 3.13, f_lasti was pointing at the end of currently executed instruction. 
"""
python_feature_f_lasti_is_offset = _python_version >= 0x030D
"""
Python 3.10 introduces a GEN_START no-op instruction. Python 3.11 and above re-works this further towards RESUME.
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
python_feature_load_attr_method = _python_version >= 0x030C
python_feature_put_null = _python_version >= 0x030B
"""
Python up to 3.13 has LOAD_METHOD replaced by LOAD_GLOBAL, LOAD_ATTR, etc with a flag bit to push NULL/self.
"""
python_feature_load_method = "LOAD_METHOD" in opmap
python_feature_resume_opcode = _python_version >= 0x030B
python_feature_return_generator_opcode = _python_version >= 0x030B
"""
Python 3.10-3.12 generators start with one item in the value stack and RETURN_GENERATOR has zero stack effect.
Python 3.13 and above effectively start from scratch.
"""
python_feature_generator_value_stack_pre_filled = _python_version >= 0x030A and _python_version <= 0x030C
"""
Prior to Python 3.11 qualname is required for MAKE_FUNCTION
"""
python_feature_make_function_qualname = _python_version < 0x030B
"""
Python 3.13 MAKE_FUNCTION does not accept any arguments.
"""
python_feature_simple_make_function = _python_version >= 0x030D
"""
Python 3.11 introduces exception tables to replace block stack and opcodes such as SETUP_FINALLY, etc.
"""
python_feature_exceptiontable = _python_version >= 0x030B
"""
Python 3.11 introduces a simple CALL.
"""
python_feature_simple_call = _python_version >= 0x030B
"""
Python 3.11-3.12 require NULL to be put BEFORE the callable when doing simple calls while subsequent python
versions require it to be put AFTER (i.e. as if it is a regular argument).
"""
python_feature_call_null_swapped = _python_version >= 0x030B and _python_version <= 0x030C
"""
Python up to 3.13 have BINARY_SUBSCR replaced by BINARY_OP since python 3.14.
BINARY_OP itself exists since python 3.11.
"""
python_feature_binary_subscr = "BINARY_SUBSCR" in opmap
python_feature_binary_op = "BINARY_OP" in opmap

# These unconditionally interrupt the normal bytecode flow
interrupting = tuple(
    opmap[i]
    for i in (
        "JUMP_ABSOLUTE",
        "JUMP_FORWARD",
        "JUMP_BACKWARD",
        "RETURN_VALUE",
        "RAISE_VARARGS",
        "RERAISE",  # 3.9+
        "JUMP_BACKWARD_NO_INTERRUPT",  # 3.11+
        "RETURN_CONST",  # 3.12+
    )
    if i in opmap
)
gen_start = tuple(
    opmap[i]
    for i in ("GEN_START", "RETURN_GENERATOR")
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
double_packed = {
    opmap[i]: tuple(opmap[_j] for _j in j)
    for i, j in (
        ("LOAD_FAST_LOAD_FAST", ("LOAD_FAST", "LOAD_FAST")),
        ("LOAD_FAST_BORROW_LOAD_FAST_BORROW", ("LOAD_FAST_BORROW", "LOAD_FAST_BORROW")),
        ("STORE_FAST_STORE_FAST", ("STORE_FAST", "STORE_FAST")),
        ("STORE_FAST_LOAD_FAST", ("STORE_FAST", "LOAD_FAST")),
    )
    if i in opmap
}
binary_op_arg = {}
if python_feature_binary_op:
    from dis import _nb_ops
    for val, (name, _) in enumerate(_nb_ops):
        binary_op_arg[name] = val
    del _nb_ops
"""
Python 3.11 and above introduce a contiguous memory chunk for locals plus cells.
"""
locals_plus = ()
if _python_version >= 0x030B:
    locals_plus += tuple(
        opmap[i]
        for i in ("LOAD_DEREF", "STORE_DEREF", "DELETE_DEREF", "MAKE_CELL", "COPY_FREE_VARS", "LOAD_CLOSURE")
        if i in opmap
    )
if _python_version >= 0x030D:
    locals_plus += tuple(
        opmap[i]
        for i in("LOAD_FAST", "STORE_FAST", "LOAD_FAST_LOAD_FAST", "STORE_FAST_STORE_FAST", "LOAD_FAST_CHECK",
                 "LOAD_FAST_AND_CLEAR", "STORE_FAST_LOAD_FAST", "LOAD_FAST_BORROW",
                 "LOAD_FAST_BORROW_LOAD_FAST_BORROW", "STORE_FAST_MAYBE_NULL")
        if i in opmap
    )
del opmap


def guess_entering_stack_size(opcode: int) -> int:
    """
    Figure out the starting stack size given the starting opcode.

    Parameters
    ----------
    opcode
        The starting opcode.

    Returns
    -------
    The size of the value stack at entry.
    """
    return int(opcode in gen_start) if python_feature_generator_value_stack_pre_filled else 0
