"""
Preparing morph bytecode.

- `morph_execpoint()`: prepares a bytecode in form of a function for a single stack item;
- `morph_stack()`: prepares a series of functions representing the state of the entire stack;
"""
import dis
import logging
from types import CodeType, FunctionType
from functools import partial
import sys

from .minias import Bytecode, jump_multiplier
from .primitives import NULL
from .opcodes import (
    POP_TOP, UNPACK_SEQUENCE,
    LOAD_CONST, LOAD_FAST, LOAD_ATTR, LOAD_METHOD, LOAD_GLOBAL,
    STORE_FAST, STORE_NAME, STORE_GLOBAL,
    JUMP_ABSOLUTE,
    CALL_FUNCTION, CALL_METHOD,
    IMPORT_NAME, IMPORT_FROM, MAKE_FUNCTION,
    RAISE_VARARGS, SETUP_FINALLY,
)
from .util import log_bytecode

EXCEPT_HANDLER = 257
python_version = sys.version_info.major * 0x100 + sys.version_info.minor

if python_version > 0x0309:  # 3.10 and above
    from .opcodes import GEN_START


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


class MorphCode(Bytecode):
    def put_except_handler(self) -> None:
        """
        Puts except handler and 3 items (NULL, NULL, None) on the stack.
        """
        # try:
        #     raise
        # except:
        #     POP, POP, POP
        #     ...
        setup_finally = self.I(SETUP_FINALLY, None)
        self.i(RAISE_VARARGS, 0)
        for i in range(3):
            pop_top = self.i(POP_TOP, 0)
            if i == 0:
                setup_finally.jump_to = pop_top

    def put_null(self) -> None:
        """
        Puts a single NULL on the stack.
        """
        # any unbound method will work here
        # property.fget
        # POP
        self.I(LOAD_GLOBAL, "property")
        self.I(LOAD_METHOD, "fget")
        self.i(POP_TOP, 0)

    def sign(self, signature=b'mrph') -> None:
        """
        Marks the code with a static signature.
        """
        self.pos = len(self)
        self.c("!signature")
        self.nop(signature)

    def put_unpack(self, storage_name: str, storage, tos) -> None:
        """
        Unpack an object from the storage to TOS.
        Assembles a bytecode to unpack an object from the
        storage accessed via its name. At the same time,
        registers the object in the storage.

        Parameters
        ----------
        storage_name
            The name of the storage in globals.
        storage
            The storage itself.
        tos
            The object to put.
        """
        # storage_name(id(tos))
        self.I(LOAD_GLOBAL, storage_name)
        self.I(LOAD_CONST, storage.store(tos))
        self.i(CALL_FUNCTION, 1)

    def unpack_storage(self, storage_name: str, storage) -> int:
        """
        Unpack the storage.

        Parameters
        ----------
            The name of the storage in globals.
        storage
            The storage itself.

        Returns
        -------
        handle
            Position in `code.co_consts` where the packed
            storage resides.
        """
        # storage.loads(data) (kinda)
        self.I(LOAD_CONST, storage.loads.__code__)
        self.I(LOAD_CONST, "unpack")
        self.i(MAKE_FUNCTION, 0)
        handle = self.I(LOAD_CONST, "<storage_data>", create_new=True).arg
        self.i(CALL_FUNCTION, 1)
        self.I(STORE_GLOBAL, storage_name)
        return handle


def morph_execpoint(p, nxt, call_nxt=False, storage=None, storage_name=None,
                    pin_storage=False, module_globals=None, flags=0):
    """
    Prepares a code object which morphs into the desired state
    and continues the execution afterwards.

    Parameters
    ----------
    p : execpoint
        The execution point to morph into.
    nxt : object
        An item to put on top of the stack.
        Typically, appears as if this item was returned.
    call_nxt : bool
        If True, calls `nxt` without arguments,
        assuming `nxt` is a code object without
        arguments. Use it to develop the call
        stack.
    storage : LocalStorage, None
        Storage for python objects.
    storage_name : str
        Storage name in globals.
    pin_storage : bool
        If True, pins the storage into this frame's globals.
    module_globals : list
        An optional list of execpoints to initialize module globals.
    flags : int
        Code object flags.

    Returns
    -------
    result : FunctionType
        The resulting morph.
    """
    if storage is not None:
        if storage_name is None:
            storage_name = "pyteleport_morph_global_storage"
        if pin_storage and module_globals is None:
            raise ValueError("Module globals required to pin the storage")
    logging.debug("Assembling morph ...")
    for i in str(p).split("\n"):
        logging.debug(i)
    logging.debug(f"  storage={storage}")
    logging.debug(f"  storage_name='{storage_name}'")
    logging.debug(f"  pin_storage={pin_storage}")
    code = Bytecode.disassemble(p.code).copy(MorphCode)
    if python_version >= 0x030A and next(code.iter_opcodes()).opcode == GEN_START:
        # Leave the generator header on top
        code.pos = 1
    else:
        code.pos = 0
    f_code = p.code

    if storage is not None:
        if pin_storage:
            logging.debug(f"Storage will be pinned in this frame's globals as '{storage_name}'")
            code.c("!unpack global storage")
            storage_future_data_handle = code.unpack_storage(storage_name, storage)

        put = partial(code.put_unpack, storage_name, storage)
    else:
        put = partial(code.I, LOAD_CONST)

    # locals
    code.c(f"!unpack locals")
    for i_obj_in_locals, obj_in_locals in enumerate(p.v_locals):
        if obj_in_locals is not NULL:
            put(obj_in_locals)
            code.i(STORE_FAST, i_obj_in_locals)

    # globals
    for unpack_data, unpack_name, store_opcode in [
        # (p.v_locals, "locals", STORE_FAST),
        (module_globals, "globals", STORE_NAME),
    ]:
        if unpack_data is not None and len(unpack_data) > 0:
            code.c(f"!unpack {unpack_name}")
            klist, vlist = zip(*unpack_data.items())
            put(vlist)
            code.i(UNPACK_SEQUENCE, len(vlist))
            for k in klist:
                # k = v
                code.I(store_opcode, k)

    # load block and value stacks
    code.c("!unpack stack")
    stack_items = _iter_stack(p.v_stack, p.block_stack)
    for item, is_value in stack_items:
        if is_value:
            if item is NULL:
                code.put_null()
            else:
                put(item)
        else:
            if item.type == SETUP_FINALLY:
                code.i(SETUP_FINALLY, 0, jump_to=code.by_pos(item.handler * jump_multiplier))
            elif item.type == EXCEPT_HANDLER:
                assert next(stack_items) == (NULL, True)  # traceback
                assert next(stack_items) == (NULL, True)  # value
                assert next(stack_items) == (None, True)  # type
                code.put_except_handler()
            else:
                raise NotImplementedError(f"Unknown block type={type} ({dis.opname.get(type, 'unknown opcode')})")

    if nxt is not NULL:
        code.c("!unpack TOS")
        put(nxt)
        if call_nxt:
            code.c("!call TOS")
            if isinstance(nxt, FunctionType):
                code.i(CALL_FUNCTION, 0)
            elif isinstance(nxt, CodeType):
                put(f"morph_into:{f_code.co_name}")
                code.i(MAKE_FUNCTION, 0)
                code.i(CALL_FUNCTION, 0)
            else:
                raise ValueError(f"cannot call {nxt}")

    # now jump to the previously saved position
    code.c("!final jump")
    last_opcode = code.i(JUMP_ABSOLUTE, 0, jump_to=code.by_pos(p.pos + 2))

    code.c("!code")

    # add signature
    code.sign()

    if storage is not None and pin_storage:
        code.co_consts[storage_future_data_handle] = storage.dumps(storage)

    # finalize
    bytecode_data = code.get_bytecode()
    
    # determine the desired stack size
    s = 0
    preamble_stack_size = 0
    for i in code.iter_opcodes():
        s += i.get_stack_effect(jump=True)
        preamble_stack_size = max(preamble_stack_size, s)
        if i is last_opcode:
            break

    result = CodeType(
        0,
        0,
        0,
        len(code.co_varnames),
        max(f_code.co_stacksize, preamble_stack_size),
        flags,
        bytecode_data,
        tuple(code.co_consts),
        tuple(code.co_names),
        tuple(code.co_varnames),
        f_code.co_filename,  # TODO: something different should be here
        f_code.co_name,
        f_code.co_firstlineno,  # TODO: this has to be fixed
        f_code.co_lnotab,
    )
    for i in str(code).split("\n"):
        log_bytecode(i)
    return FunctionType(
        result,
        p.v_globals,
        f"morph_into:{p.code.co_name}",
    )


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
    logging.info(f"Morphing frame stack depth={len(frame_data)}")
    for frame_i, frame in enumerate(frame_data):
        logging.info(f"Morphing frame {frame_i + 1:d}/{len(frame_data)}")
        is_topmost = frame is frame_data[0]
        is_root = frame is frame_data[-1] and root
        tos = morph_execpoint(
            frame, tos,
            call_nxt=not is_topmost,
            pin_storage=is_root,
            module_globals=frame.v_globals if is_root else None,
            **kwargs
        )
    return tos
