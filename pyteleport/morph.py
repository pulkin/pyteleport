"""
Preparing morph bytecode.

- `morph_execpoint()`: prepares a bytecode in form of a function for a single stack item;
- `morph_stack()`: prepares a series of functions representing the state of the entire stack;
"""
import dis
import logging
from types import FunctionType
from typing import Optional
from functools import partial
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext

from .bytecode import Bytecode, disassemble, jump_multiplier
from .bytecode.primitives import AbstractInstruction, NoArgInstruction, ConstInstruction, NameInstruction, \
    NameInstructionBit, EncodedInstruction, ReferencingInstruction, FloatingCell
from .bytecode.minias import assign_stack_size
from .primitives import NULL
from .bytecode.opcodes import (
    POP_TOP, UNPACK_SEQUENCE, BUILD_TUPLE,
    LOAD_CONST, LOAD_FAST, LOAD_ATTR, LOAD_GLOBAL,
    STORE_FAST, STORE_NAME, STORE_GLOBAL, STORE_ATTR,
    JUMP_FORWARD, RETURN_VALUE,
    CALL_FUNCTION_EX,
    IMPORT_NAME, IMPORT_FROM, MAKE_FUNCTION,
    RAISE_VARARGS, GEN_START, PUSH_NULL, LOAD_METHOD,
    RESUME, MAKE_CELL, COPY_FREE_VARS, CALL, BINARY_SUBSCR, BINARY_OP, binary_op_arg,
    python_feature_block_stack,
    python_feature_load_global_null,
    python_feature_load_attr_method,
    python_feature_make_function_qualname,
    python_feature_simple_make_function,
    python_feature_call_null_swapped
)
from .util import log_bytecode
from .storage import transmission_engine
from .bytecode.patched_opcode import hasfree

EXCEPT_HANDLER = 257

if python_feature_block_stack:
    from .bytecode.opcodes import SETUP_FINALLY
if python_feature_call_null_swapped:
    # by coincidence, SWAP is available for all python versions where we need to swap NULL for CALL
    from .bytecode.opcodes import SWAP


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
    if python_feature_block_stack:
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


NOTSET = object()


@dataclass
class MorphCode(Bytecode):
    editing: int = 0

    def __post_init__(self):
        self.__editing_history__ = []

    def __enter__(self):
        self.__editing_history__.append(self.editing)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.editing = self.__editing_history__.pop()

    @classmethod
    def from_bytecode(cls, code: Bytecode) -> "MorphCode":
        return cls(
            instructions=code.instructions,
            exception_table=code.exception_table,
            current=code.current,
        )

    def get_marks(self):
        result = super().get_marks()
        if 0 <= self.editing < len(self.instructions):
            result[self.instructions[self.editing]] = "(e)"
        return result

    def insert_cell(self, cell: FloatingCell, at: Optional[int] = None):
        if at is not None:
            self.instructions.insert(at, cell)
        else:
            self.instructions.insert(self.editing, cell)
            self.editing += 1
        return cell

    def insert(self, instruction: AbstractInstruction, at: Optional[int] = None):
        return self.insert_cell(FloatingCell(instruction), at=at)

    def i(self, opcode: int, arg=NOTSET) -> FloatingCell:
        if opcode < dis.HAVE_ARGUMENT:
            if arg is not NOTSET:
                raise ValueError(f"no argument expected for {dis.opname[opcode]}; provided: {arg=}")
            result = NoArgInstruction(opcode)
        else:
            if arg is NOTSET:
                raise ValueError(f"argument expected for {dis.opname[opcode]}")
            if opcode in dis.hasconst:
                result = ConstInstruction(opcode, arg)
            elif opcode in dis.hasname + dis.haslocal + dis.hasfree:
                if not isinstance(arg, str):
                    raise ValueError(f"string argument expected for {dis.opname[opcode]}; provided: {arg=}")
                if (python_feature_load_global_null and opcode == LOAD_GLOBAL) or \
                        (python_feature_load_attr_method and opcode == LOAD_ATTR):
                    result = NameInstructionBit(opcode, arg, bit=False)
                else:
                    result = NameInstruction(opcode, arg)
            elif opcode in dis.hasjabs + dis.hasjrel:
                if not isinstance(arg, FloatingCell):
                    raise ValueError(f"cell argument expected for {dis.opname[opcode]}; provided: {arg=}")
                result = ReferencingInstruction(opcode, arg)
            else:
                if not isinstance(arg, int):
                    raise ValueError(f"integer argument expected for {dis.opname[opcode]}; provided: {arg=}")
                result = EncodedInstruction(opcode, arg)
        return self.insert(result)

    def c(self, *args):
        pass

    def sign(self):
        pass

    def put_except_handler(self) -> None:
        """
        Puts except handler and 3 items (NULL, NULL, None) on the stack.
        """
        # try:
        #     raise
        # except:
        #     POP, POP, POP
        #     ...
        towards = FloatingCell(NoArgInstruction(POP_TOP))
        self.i(SETUP_FINALLY, towards)
        self.i(RAISE_VARARGS, 0)
        self.insert_cell(towards)
        for i in range(2):
            self.i(POP_TOP)

    def put_null(self) -> None:
        """
        Puts a single NULL on the stack.
        """
        if PUSH_NULL is not None:
            self.i(PUSH_NULL)
        else:
            assert LOAD_METHOD is not None
            # any unbound method will work here
            # property.fget
            # POP
            self.i(LOAD_GLOBAL, "property")
            self.i(LOAD_METHOD, "fget")
            self.i(POP_TOP)

    def put_unpack(self, object_storage_name: str, object_storage: dict, tos) -> None:
        """
        Unpack an object from the storage to TOS.
        Assembles a bytecode to unpack an object from the
        storage accessed via its name. At the same time,
        registers the object in the storage.

        Parameters
        ----------
        object_storage_name
            The storage name.
        object_storage
            The storage itself.
        tos
            The object to put.
        """
        # storage_name[id(tos)]
        handle = id(tos)
        object_storage[handle] = tos
        self.i(LOAD_GLOBAL, object_storage_name)
        self.i(LOAD_CONST, handle)
        if BINARY_SUBSCR is not None:
            self.i(BINARY_SUBSCR)
        else:
            self.i(BINARY_OP, binary_op_arg["NB_SUBSCR"])

    def put_module(self, name: str, fromlist=None, level=0):
        """
        Simple module import.

        Parameters
        ----------
        name
            Module name.
        fromlist
            Module names to import.
        level
            Import level (absolute or relative).
        """
        self.i(LOAD_CONST, level)
        self.i(LOAD_CONST, fromlist)
        self.i(IMPORT_NAME, name)

    def unpack_storage(
        self,
        object_storage_name: str,
        object_storage_protocol: transmission_engine,
        object_data: bytes,
    ) -> FloatingCell:
        """
        Unpack the storage.

        Parameters
        ----------
        object_storage_name
            The name of the storage in builtins.
        object_storage_protocol
            A collection of functions governing initial serialization
            and de-serialization of the global storage dict.
        object_data
            The serialized data which object storage unpacks.

        Returns
        -------
        handle
            Position in `code.co_consts` where the serialized data is
            expected.
        """
        self.i(LOAD_CONST, object_storage_protocol.load_from_code.__code__)
        if python_feature_make_function_qualname:
            self.i(LOAD_CONST, "unpack")
        if python_feature_simple_make_function:
            self.i(MAKE_FUNCTION)
        else:
            self.i(MAKE_FUNCTION, 0)
        result = self.i(LOAD_CONST, object_data)
        if CALL is not None:
            self.i(CALL, 0)
        else:
            self.i(BUILD_TUPLE, 1)
            self.i(CALL_FUNCTION_EX, 0)
        # import builtins
        self.put_module("builtins")
        # builtins.morph_data = ...
        self.i(STORE_ATTR, object_storage_name)
        return result

    @contextmanager
    def callable_ctx(self):
        if PUSH_NULL is not None and python_feature_call_null_swapped:
            self.put_null()
        yield
        if PUSH_NULL is not None and not python_feature_call_null_swapped:
            self.put_null()


def assemble_entry_point(
        tos,
        object_storage,
        object_storage_protocol,
        object_storage_name: str = "morph_data",
):
    """
    Assembles the topmost service frame unpacking serialized data.

    Parameters
    ----------
    tos
        The top of the stack: a morph function that will be called.
    object_storage : dict, None
        Storage dictionary for python objects.
    object_storage_name : str
        Storage name in globals.
    object_storage_protocol : storage_protocol
        A collection of functions governing initial serialization
        and de-serialization of the global storage dict.

    Returns
    -------
    A code object running object de-serialization and invoking the tos.
    """
    code = MorphCode(instructions=[])

    # unpack storage into a name in globals
    load_storage_handle = code.unpack_storage(object_storage_name, object_storage_protocol, b"")

    # load tos
    with code.callable_ctx():
        code.put_unpack(object_storage_name, object_storage, tos)

    # call exec(tos)
    if CALL is not None:
        code.i(CALL, 0)
    else:
        code.i(BUILD_TUPLE, 0)
        code.i(CALL_FUNCTION_EX, 0)
    code.i(RETURN_VALUE)

    # populate serialized data
    load_storage_handle.instruction = ConstInstruction(
        load_storage_handle.instruction.opcode,
        object_storage_protocol.save_to_code(object_storage),
    )

    assign_stack_size(code.instructions)
    stack_size = max(i.metadata.stack_size or 0 for i in code.instructions)
    return code.assemble().to_code_object(stacksize=stack_size)


def morph_into(snapshot, tos, call_tos=False, flags=0):
    """
    Prepares a code object which morphs into the desired stack frame state
    and continues the execution afterwards.

    Parameters
    ----------
    snapshot : FrameSnapshot
        The frame snapshot to morph into.
    tos : object
        An item to put on top of the stack.
        Typically, appears as if this item was returned.
    call_tos : bool
        If True, calls tos.
    flags : int
        Code object flags.

    Returns
    -------
    result : FunctionType
        The resulting morph.
    """
    logging.debug("Assembling morph ...")
    for i in str(snapshot).split("\n"):
        logging.debug(i)
    code = MorphCode.from_bytecode(disassemble(snapshot.code, f_lasti=snapshot.f_lasti))
    lookup_orig = {
        i.metadata.source.offset: i
        for i in code.instructions
    }
    # execute the header
    if code.instructions[0].instruction.opcode == GEN_START:
        # Leave the header as-is
        code.editing = 1
    elif RESUME is not None:
        transaction = []
        for i, cell in enumerate(code.instructions):
            if cell.instruction.opcode in (MAKE_CELL, COPY_FREE_VARS):
                # remove because cells are supplied
                transaction.append(i)
            elif cell.instruction.opcode == RESUME:
                code.editing = i + 1
                break
        else:
            code.print(log_bytecode)
            raise ValueError("Expected a RESUME opcode but found none")
        for i in transaction[::-1]:
            del code.instructions[i]
            code.editing -= 1
        # add COPY_FREE_VARS if needed
        arg = len({id(cell.instruction.arg) for cell in code.instructions if cell.instruction.opcode in hasfree})
        if arg:
            code.i(COPY_FREE_VARS, arg)
    else:
        code.editing = 0
    f_code = snapshot.code
    code.c("--------------")
    code.c("Morph preamble")
    code.c("--------------")

    # locals
    for obj_collection, known_as, store_opcode in [
        (zip(snapshot.code.co_varnames, snapshot.fast_locals), "locals", STORE_FAST),
    ]:
        code.c(f"!unpack {known_as}")
        for obj_name, obj_in_collection in obj_collection:
            if obj_in_collection is not NULL:
                code.i(LOAD_CONST, obj_in_collection)
                code.i(store_opcode, obj_name)

    # load block and value stacks
    code.c("!unpack stack")
    stack_items = _iter_stack(snapshot.v_stack, snapshot.block_stack)
    for item, is_value in stack_items:
        if is_value:
            if item is NULL:
                code.put_null()
            else:
                code.i(LOAD_CONST, item)
        else:
            if item.type == SETUP_FINALLY:
                code.i(SETUP_FINALLY, lookup_orig[item.handler * jump_multiplier])
            elif item.type == EXCEPT_HANDLER:
                assert next(stack_items) == (NULL, True)  # traceback
                assert next(stack_items) == (NULL, True)  # value
                assert next(stack_items) == (None, True)  # type
                code.put_except_handler()
            else:
                raise NotImplementedError(f"Unknown block type={item.type} ({dis.opname.get(item.type, 'unknown opcode')})")

    if tos is not NULL:
        code.c("!unpack TOS")
        tos_context = code.callable_ctx if call_tos else nullcontext
        with tos_context():
            code.i(LOAD_CONST, tos)
        if call_tos:
            # TODO: move calling
            code.c("!call TOS")
            if CALL is not None:
                code.i(CALL, 0)
            else:
                code.i(BUILD_TUPLE, 0)
                code.i(CALL_FUNCTION_EX, 0)

    # now jump to the previously saved position
    if code.current is not None:
        code.c("!final jump")
        code.i(JUMP_FORWARD, code.instructions[code.instructions.index(code.current) + 1])

    code.c("-----------------")
    code.c("Original bytecode")
    code.c("-----------------")

    # add signature
    code.sign()

    # finalize
    assign_stack_size(code.instructions, code.exception_table)
    stack_size = max(i.metadata.stack_size or 0 for i in code.instructions)
    code.print(log_bytecode)
    assembled = code.assemble()
    result = assembled.to_code_object(
        stacksize=stack_size,
        flags=flags,
        freevars=assembled.cells,
        filename=f_code.co_filename,  # TODO: something different should be here
        name=f_code.co_name,
        qualname=getattr(f_code, "co_qualname", f_code.co_name),
        firstlineno=f_code.co_firstlineno,  # TODO: this has to be fixed
        linetable=f_code.co_lnotab,
    )

    return FunctionType(
        code=result,
        globals=snapshot.v_globals,
        name=f"morph_into:{snapshot.code.co_name}",
        closure=tuple(snapshot.v_cells),
    )


def morph_stack(frame_data, tos=None, **kwargs):
    """
    Morphs the stack.

    Parameters
    ----------
    frame_data : list
        States of all individual frames.
    tos : object
        Top-of-stack object for the executing frame.
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
        tos = morph_into(
            frame, tos,
            call_tos=not is_topmost,
            **kwargs
        )
    return tos
