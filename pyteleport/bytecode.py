from dis import opmap

locals().update(opmap)  # unpack opcodes here

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
