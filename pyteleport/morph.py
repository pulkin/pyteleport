import dis
import logging
from types import CodeType, FunctionType
import sys
import marshal

from .minias import Bytecode, jump_multiplier
from .primitives import NULL
from .bytecode import (
    POP_TOP, UNPACK_SEQUENCE,
    LOAD_CONST, LOAD_FAST, LOAD_ATTR, LOAD_METHOD, LOAD_GLOBAL,
    STORE_FAST, STORE_NAME,
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


def morph_execpoint(p, nxt, call_nxt=False, pack=None, unpack=None, module_globals=None, flags=0):
    """
    Prepares a code object which morphs into the desired state
    and continues the execution afterwards.

    Parameters
    ----------
    p : execpoint
        The execution point to morph into.
    nxt : object
        An item to put on top of the stack.
    call_nxt : bool
        If True, calls `nxt` without arguments.
        Used to develop the stack.
    pack : Callable, None
        A method turning objects into bytes (serializer)
        locally.
    unpack : Callable, None
        Another method turning bytes into objects (deserializer).
        The function has to be self-consistent (i.e. only rely on locals).
    module_globals : list
        An optional list of execpoints to initialize module globals.
    flags : int
        Code object flags.

    Returns
    -------
    result : FunctionType
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

    # locals
    for unpack_data, unpack_name, unpack_opcode in [
        (p.v_locals, "locals", STORE_FAST),
        (module_globals, "globals", STORE_NAME),
    ]:
        if unpack_data is not None and len(unpack_data) > 0:
            code.c(f"{unpack_name} ...")
            klist, vlist = zip(*unpack_data.items())
            _LOAD(vlist)
            code.i(UNPACK_SEQUENCE, len(vlist))
            for k in klist:
                # k = v
                code.I(unpack_opcode, k)

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

    if nxt is not NULL:
        code.c("stack top")
        _LOAD(nxt)
        if call_nxt:
            code.c("... call")
            code.i(CALL_FUNCTION, 0)

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
    return FunctionType(result, p.v_globals)


def morph_stack(frame_data, tos=None, root=True, **kwargs):
    """
    Morphs the stack.

    Parameters
    ----------
    frame_data : list
        States of all individual frames.
    tos : object
        Top-of-stack object for the executing frame.
    root : bool
        Indicates if the stack contains a root
        frame where globals need to be unpacked.
    kwargs
        Arguments to morph_execpoint.

    Returns
    -------
    function : FunctionType
        The resulting morph for the root frame.
    """
    for i_frame, frame in enumerate(frame_data):
        tos = morph_execpoint(
            frame, tos,
            call_nxt=i_frame != 0,
            module_globals=frame.v_globals if root and frame is frame_data[-1] else None,
            **kwargs)
    return tos
