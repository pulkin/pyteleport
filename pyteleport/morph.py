import dis
import logging
from types import CodeType

from .py import (
    JX,
    put_NULL,
    put_EXCEPT_HANDLER,
    disassemble,
)
from .primitives import NULL


locals().update(dis.opmap)
EXCEPT_HANDLER = 257


def is_marshalable(o):
    """
    Determines if the object is marshalable.

    Parameters
    ----------
    o
        Object to test.

    Returns
    -------
    result : bool
        True if marshalable. False otherwise.
    """
    return isinstance(o, (str, bytes, int, float, complex, CodeType))  # TODO: add lists, tuples and dicts


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


def morph_execpoint(p, nxt, pack=None, unpack=None, globals=False, locals=True, fake_return=True, flags=0):
    """
    Prepares a code object which morphs into the desired state
    and continues the execution afterwards.

    Parameters
    ----------
    p : execpoint
        The execution point to morph into.
    nxt : CodeType
        The code object which develops the stack further.
    pack : Callable, None
        A method turning objects into bytes (serializer)
        locally.
    unpack : tuple, None
        A 2-tuple `(module_name, method_name)` specifying
        the method that morph uses to unpack the data.
    globals : bool
        If True, unpacks globals.
    locals : bool
        If True, unpacks locals.
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
    logging.info(f"Preparing a morph into execpoint {p} pack={pack is not None} ...")
    code = disassemble(p.code)
    code.pos = 0
    code.c("Header")
    code.nop(b'mrph')  # signature
    f_code = p.code
    new_stacksize = f_code.co_stacksize

    def _IMPORT(_from, _what):
        for i in range(len(code.co_varnames) + 1):
            _candidate_name = f"{_from}_{_what}{i:d}"
            if _candidate_name not in code.co_varnames:
                break
        code.c(f"from {_from} import {_what} as {_candidate_name}")
        _rtn_value = code.co_varnames(_candidate_name)
        code.I(LOAD_CONST, 0)
        code.I(LOAD_CONST, (_what,))
        code.I(IMPORT_NAME, _from)
        code.I(IMPORT_FROM, _what)
        code.i(STORE_FAST, _rtn_value)
        code.i(POP_TOP, 0)
        return _rtn_value

    if pack:
        unpack = _IMPORT(*unpack)
        def _LOAD(_what):
            if is_marshalable(_what):
                code.I(LOAD_CONST, _what)
            else:
                code.i(LOAD_FAST, unpack)
                code.I(LOAD_CONST, pack(_what))
                code.i(CALL_FUNCTION, 1)
    else:
        def _LOAD(_what):
            code.I(LOAD_CONST, _what)

    scopes = []
    if locals:
        scopes.append((p.v_locals, STORE_FAST, "locals"))
    if globals:
        scopes.append((p.v_globals, STORE_GLOBAL, "globals"))
    for _dict, _STORE, log_name in scopes:
        logging.info(f"  {log_name} ...")
        if len(_dict) > 0:
            code.c(f"{log_name} = ...")
            klist, vlist = zip(*_dict.items())
            _LOAD(vlist)
            code.i(UNPACK_SEQUENCE, len(vlist))
            for k in klist:
                # k = v
                code.I(_STORE, k)
            new_stacksize = max(new_stacksize, len(vlist))

    # load block and value stacks
    code.c("*stack")
    stack_items = _iter_stack(p.v_stack, p.block_stack)
    for item, is_value in stack_items:
        if is_value:
            if item is NULL:
                put_NULL(code)
            else:
                _LOAD(item)
        else:
            if item.type == SETUP_FINALLY:
                code.i(SETUP_FINALLY, 0, jump_to=code.by_pos(item.handler * JX))
            elif item.type == EXCEPT_HANDLER:
                assert next(stack_items) == (NULL, True)  # traceback
                assert next(stack_items) == (NULL, True)  # value
                assert next(stack_items) == (None, True)  # type
                put_EXCEPT_HANDLER(code)
            else:
                raise NotImplementedError(f"Unknown block type={type} ({dis.opname.get(type, 'unknown opcode')})")

    if nxt is not None:
        # call nxt which is a code object
        code.c(f"nxt()")

        # load code object
        code.I(LOAD_CONST, nxt)
        code.I(LOAD_CONST, None)  # function name
        code.i(MAKE_FUNCTION, 0)  # turn code object into a function
        code.i(CALL_FUNCTION, 0)  # call it
    elif fake_return:
        code.c(f"fake return None")
        code.I(LOAD_CONST, None)  # fake nxt returning None

    # now jump to the previously saved position
    code.c(f"goto saved pos")
    code.i(JUMP_ABSOLUTE, 0, jump_to=code.by_pos(p.pos + 2))

    code.c(f"---------------------")
    code.c(f"The original bytecode")
    code.c(f"---------------------")
    result = CodeType(
        0,
        0,
        0,
        len(code.co_varnames),
        new_stacksize + 1,
        flags,
        code.get_bytecode(),
        tuple(code.co_consts),
        tuple(code.co_names),
        tuple(code.co_varnames),
        f_code.co_filename,  # TODO: something smarter should be here
        f_code.co_name,
        f_code.co_firstlineno,  # TODO: this has to be fixed
        f_code.co_lnotab,
        )
    logging.info(f"resulting morph:\n{str(code)}")
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
    result : CodeType
        The resulting morph for the root frame.
    """
    prev = None
    for i, frame in enumerate(frame_data):
        logging.info(f"Preparing morph #{i:d}")
        prev = morph_execpoint(frame, prev,
            globals=root and frame is frame_data[-1],
            locals=frame is not frame_data[-1] or not root,
            **kwargs)
    return prev

