import dis
import logging
from types import CodeType
import sys
import marshal

from .minias import Bytecode, jump_multiplier
from .primitives import NULL
from .bytecode import (
    POP_TOP, UNPACK_SEQUENCE,
    LOAD_CONST, LOAD_FAST, LOAD_ATTR, LOAD_METHOD, LOAD_GLOBAL,
    STORE_FAST,
    JUMP_ABSOLUTE,
    CALL_FUNCTION, CALL_METHOD,
    IMPORT_NAME, IMPORT_FROM, MAKE_FUNCTION,
    RAISE_VARARGS, SETUP_FINALLY,
)
from .util import log_bytecode

EXCEPT_HANDLER = 257
python_version = sys.version_info.major * 0x100 + sys.version_info.minor

if python_version > 0x0309:  # 3.10 and above
    from .bytecode import GEN_START


def _iter_stack(value_stack, block_stack):
    """
    Iterates value and block stacks in the order they have to be placed.

    Parameters
    ----------
    value_stack : Iterable
        Value stack items.
    block_stack : Iterable
        Block stack items.

    Yields
    ------
    stack_item
        The next item to place.
    is_value : bool
        Indicates if the item belongs to the value stack.
    """
    v_stack_iter = enumerate(value_stack, start=1)
    cur_stack_level = 0
    for block_stack_item in block_stack:

        # check if items are coming in accending order
        if block_stack_item.level < cur_stack_level:
            raise ValueError(f"Illegal block_stack.level={block_stack_item.level}")

        # output stack items up to the block stack level
        while cur_stack_level < block_stack_item.level:
            try:
                cur_stack_level, stack_item = next(v_stack_iter)
            except StopIteration:
                raise StopIteration(f"Depleted value stack items ({cur_stack_level}) for block_stack.level={block_stack_item.level}")
            yield stack_item, True

        # output block stack item
        yield block_stack_item, False

    # output the rest of the value stack
    for _, stack_item in v_stack_iter:
        yield stack_item, True


def _put_except_handler(code):
    """
    Puts except handler and 3 items (NULL, NULL, None) on the stack.

    Parameters
    ----------
    code : Bytecode
        Code to process.
    """
    setup_finally = code.I(SETUP_FINALLY, None)
    code.i(RAISE_VARARGS, 0)
    for i in range(3):
        pop_top = code.i(POP_TOP, 0)
        if i == 0:
            setup_finally.jump_to = pop_top


def _put_null(code):
    """
    Puts a single NULL on the stack.

    Parameters
    ----------
    code : Bytecode
        Code to process.
    """
    # any unbound method will work here
    code.I(LOAD_GLOBAL, "property")
    code.I(LOAD_METHOD, "fget")
    code.i(POP_TOP, 0)


def morph_execpoint(p, nxt, pack=None, unpack=None, module_globals=None, fake_return=True, flags=0):
    """
    Prepares a code object which morphs into the desired state
    and continues the execution afterwards.

    Parameters
    ----------
    p : execpoint
        The execution point to morph into.
    nxt : (CodeType, module)
        A 2-tuple with the code object which develops the stack
        further and the scope it belongs to.
    pack : Callable, None
        A method turning objects into bytes (serializer)
        locally.
    unpack : Callable, None
        Another method turning bytes into objects (deserializer).
        The function has to be self-consistent (i.e. only rely on locals).
    module_globals : list
        An optional list of execpoints to initialize module globals.
    fake_return : bool
        If set, fakes returning None by putting None on top
        of the stack. This will be ignored if nxt is not
        None.
    flags : int
        Code object flags.

    Returns
    -------
    result : CodeType
        The resulting morph.
    """
    assert pack is None and unpack is None or pack is not None and unpack is not None,\
        "Either both or none pack and unpack arguments have be specified"
    logging.debug("Assembling morph ...")
    for i in str(p).split("\n"):
        logging.debug(i)
    logging.debug(f"  pack={pack} unpack={unpack}")
    code = Bytecode.disassemble(p.code)
    if python_version >= 0x030A and next(code.iter_opcodes()).opcode == GEN_START:
        # Leave the generator header on top
        code.pos = 1
    else:
        code.pos = 0
    f_code = p.code

    def _IMPORT(_from, _what):
        code.c(f"from {_from} import {_what}")
        code.I(LOAD_CONST, 0)
        code.I(LOAD_CONST, (_what,))
        code.I(IMPORT_NAME, _from)
        code.I(IMPORT_FROM, _what)
        _rtn_value = code.I(STORE_FAST, _what, create_new=True).arg
        code.i(POP_TOP, 0)
        return _rtn_value

    if pack:
        code.c(f"def upack(...)")
        code.I(LOAD_CONST, unpack.__code__)
        code.I(LOAD_CONST, "unpack")
        code.i(MAKE_FUNCTION, 0)
        unpack = code.I(STORE_FAST, "unpack", create_new=True).arg
        def _LOAD(_what):
            try:
                marshal.dumps(_what)
            except ValueError:
                code.i(LOAD_FAST, unpack)
                code.I(LOAD_CONST, pack(_what))
                code.i(CALL_FUNCTION, 1)
            else:
                code.I(LOAD_CONST, _what)
    else:
        def _LOAD(_what):
            code.I(LOAD_CONST, _what)

    # globals: unpack them into ALL modules
    if module_globals is not None:
        for p in module_globals:
            code.c(f"{p.module.__name__}.__dict__.update(...)")
            _LOAD(p.module)
            code.I(LOAD_ATTR, "__dict__")
            code.I(LOAD_METHOD, "update")
            _LOAD(p.v_globals)
            code.i(CALL_METHOD, 1)
            code.i(POP_TOP, 0)

    # locals
    if len(p.v_locals) > 0:
        code.c(f"Locals ...")
        klist, vlist = zip(*p.v_locals.items())
        _LOAD(vlist)
        code.i(UNPACK_SEQUENCE, len(vlist))
        for k in klist:
            # k = v
            code.I(STORE_FAST, k)

    # load block and value stacks
    code.c("*stack")
    stack_items = _iter_stack(p.v_stack, p.block_stack)
    for item, is_value in stack_items:
        if is_value:
            if item is NULL:
                _put_null(code)
            else:
                _LOAD(item)
        else:
            if item.type == SETUP_FINALLY:
                code.i(SETUP_FINALLY, 0, jump_to=code.by_pos(item.handler * jump_multiplier))
            elif item.type == EXCEPT_HANDLER:
                assert next(stack_items) == (NULL, True)  # traceback
                assert next(stack_items) == (NULL, True)  # value
                assert next(stack_items) == (None, True)  # type
                _put_except_handler(code)
            else:
                raise NotImplementedError(f"Unknown block type={type} ({dis.opname.get(type, 'unknown opcode')})")

    if nxt is not None:
        # call nxt which is a code object
        nxt, nxt_scope = nxt
        code.c(f"nxt()")
        ftype = _IMPORT("types", "FunctionType")
        code.i(LOAD_FAST, ftype)  # FunctionType(
        code.I(LOAD_CONST, nxt)  # nxt,
        _LOAD(nxt_scope)  # module
        code.I(LOAD_ATTR, "__dict__")  # .__dict__
        code.i(CALL_FUNCTION, 2)  # )
        code.i(CALL_FUNCTION, 0)  # ()

    elif fake_return:
        code.c(f"fake return None")
        code.I(LOAD_CONST, None)  # fake nxt returning None

    # now jump to the previously saved position
    code.c(f"goto saved pos")
    last_opcode = code.i(JUMP_ABSOLUTE, 0, jump_to=code.by_pos(p.pos + 2))

    code.c(f"---------------------")
    code.c(f"The original bytecode")
    code.c(f"---------------------")

    # add signature
    code.pos = len(code)
    code.c("Signature")
    code.nop(b'mrph')  # signature

    # finalize
    bytecode_data = code.get_bytecode()
    
    # determine desired stack size
    s = 0
    preamble_stacksize = 0
    for i in code.iter_opcodes():
        s += i.get_stack_effect(jump=True)
        preamble_stacksize = max(preamble_stacksize, s)
        if i is last_opcode:
            break

    result = CodeType(
        0,
        0,
        0,
        len(code.co_varnames),
        max(f_code.co_stacksize, preamble_stacksize),
        flags,
        bytecode_data,
        tuple(code.co_consts),
        tuple(code.co_names),
        tuple(code.co_varnames),
        f_code.co_filename,  # TODO: something smarter should be here
        f_code.co_name,
        f_code.co_firstlineno,  # TODO: this has to be fixed
        f_code.co_lnotab,
        )
    for i in str(code).split("\n"):
        log_bytecode(i)
    return result


def morph_stack(frame_data, root=True, **kwargs):
    """
    Morphs the stack.

    Parameters
    ----------
    frame_data : list
        States of all individual frames.
    root : bool
        Indicates if the stack contains a root
        frame where globals instead of locals
        need to be unpacked.
    kwargs
        Arguments to morph_execpoint.

    Returns
    -------
    result : CodeType, module
        The resulting morph for the root frame and
        the scope it belongs to.
    """
    prev = None
    for frame in frame_data:
        prev = morph_execpoint(frame, prev,
            module_globals=frame_data if root and frame is frame_data[-1] else None,
            **kwargs), frame.module
    return prev
