"""
Preparing morph bytecode.

- `morph_execpoint()`: prepares a bytecode in form of a function for a single stack item;
- `morph_stack()`: prepares a series of functions representing the state of the entire stack;
"""
import dis
import logging
from types import CodeType, FunctionType
from typing import Optional
from functools import partial
from dataclasses import dataclass

from .bytecode import Bytecode, disassemble, jump_multiplier
from .bytecode.primitives import AbstractInstruction, NoArgInstruction, ConstInstruction, NameInstruction, \
    NameInstruction2, EncodedInstruction, ReferencingInstruction, FloatingCell
from .bytecode.minias import assign_stack_size
from .primitives import NULL
from .bytecode.opcodes import (
    POP_TOP, UNPACK_SEQUENCE, BINARY_SUBSCR, BUILD_TUPLE,
    LOAD_CONST, LOAD_FAST, LOAD_ATTR, LOAD_METHOD, LOAD_GLOBAL,
    STORE_FAST, STORE_NAME, STORE_GLOBAL, STORE_ATTR,
    JUMP_FORWARD,
    CALL_FUNCTION_EX,
    IMPORT_NAME, IMPORT_FROM, MAKE_FUNCTION,
    RAISE_VARARGS,
    guess_entering_stack_size, python_feature_block_stack, python_feature_gen_start_opcode,
    python_feature_resume_opcode, python_feature_load_global_null, python_feature_make_function_qualname,
    python_feature_put_null
)
from .util import log_bytecode
from .storage import transmission_engine

EXCEPT_HANDLER = 257

# cpython/Lib/test/test_code.py
if python_feature_block_stack:
    code_object_args = (
        "argcount", "posonlyargcount", "kwonlyargcount", "nlocals", "stacksize", "flags", "code", "consts", "names",
        "varnames", "filename", "name", "firstlineno", "linetable", "freevars", "cellvars",
    )  # no exceptiontable
else:
    code_object_args = (
        "argcount", "posonlyargcount", "kwonlyargcount", "nlocals", "stacksize", "flags", "code", "consts", "names",
        "varnames", "filename", "name", "qualname", "firstlineno", "linetable", "exceptiontable", "freevars", "cellvars",
    )  # qualname as well

if python_feature_gen_start_opcode:
    from .bytecode.opcodes import GEN_START
if python_feature_block_stack:
    from .bytecode.opcodes import SETUP_FINALLY
if python_feature_load_global_null:
    from .bytecode.opcodes import PUSH_NULL


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
        return cls(code.instructions, current=code.current)

    def get_marks(self):
        result = super().get_marks()
        if 0 <= self.editing < len(self.instructions):
            result[self.instructions[self.editing]] = "✎✎✎"
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

    def i(self, opcode: int, arg=NOTSET, bit: bool = False) -> FloatingCell:
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
                if python_feature_load_global_null and opcode == LOAD_GLOBAL:
                    result = NameInstruction2(opcode, arg, bit)
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
        if python_feature_put_null:
            self.i(PUSH_NULL)
        else:
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
        self.i(BINARY_SUBSCR)

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
        if python_feature_load_global_null:
            self.put_null()
        self.i(LOAD_CONST, object_storage_protocol.load_from_code.__code__)
        if python_feature_make_function_qualname:
            self.i(LOAD_CONST, "unpack")
        self.i(MAKE_FUNCTION, 0)
        result = self.i(LOAD_CONST, object_data)
        self.i(BUILD_TUPLE, 1)
        self.i(CALL_FUNCTION_EX, 0)
        # import builtins
        self.put_module("builtins")
        # builtins.morph_data = ...
        self.i(STORE_ATTR, object_storage_name)
        return result

    def i_print(self, what: str):
        """
        Instruct to print something.

        Parameters
        ----------
        what
            The string to print.
        """
        self.i(LOAD_GLOBAL, "print", bit=1)
        self.i(LOAD_CONST, (what,))
        self.i(LOAD_CONST, {"flush": True})
        self.i(CALL_FUNCTION_EX, 1)
        self.i(POP_TOP)


def morph_into(p, nxt, call_nxt=False, object_storage=None, object_storage_name="morph_data",
               object_storage_protocol=None, module_globals=None, flags=0):
    """
    Prepares a code object which morphs into the desired stack frame state
    and continues the execution afterwards.

    Parameters
    ----------
    p : FrameSnapshot
        The frame snapshot to morph into.
    nxt : object
        An item to put on top of the stack.
        Typically, appears as if this item was returned.
    call_nxt : bool
        If True, calls `nxt` without arguments,
        assuming `nxt` is a code object without
        arguments. Use it to develop the call
        stack.
    object_storage : dict, None
        Storage dictionary for python objects.
    object_storage_name : str
        Storage name in globals.
    object_storage_protocol : storage_protocol
        A collection of functions governing initial serialization
        and de-serialization of the global storage dict.
    module_globals : list
        An optional list of execpoints to initialize module globals.
    flags : int
        Code object flags.

    Returns
    -------
    result : FunctionType
        The resulting morph.
    """
    logging.debug("Assembling morph ...")
    for i in str(p).split("\n"):
        logging.debug(i)
    logging.debug(f"  {object_storage=}")
    logging.debug(f"  {object_storage_name=}")
    logging.debug(f"  {object_storage_protocol=}")
    code = MorphCode.from_bytecode(disassemble(p.code, pos=p.pos))
    lookup_orig = {
        i.metadata.source.offset: i
        for i in code.instructions
    }
    if python_feature_resume_opcode or (python_feature_gen_start_opcode and code.instructions[0].instruction.opcode == GEN_START):
        # Leave the header as-is
        code.editing = 1
    else:
        code.editing = 0
    f_code = p.code
    code.c("--------------")
    code.c("Morph preamble")
    code.c("--------------")

    if object_storage is not None:
        if object_storage_protocol is not None:
            logging.debug(f"Storage will be loaded here into builtins as '{object_storage_name}'")
            code.c("!unpack object storage")
            load_storage_handle = code.unpack_storage(object_storage_name, object_storage_protocol, b"to be replaced")

        put = partial(code.put_unpack, object_storage_name, object_storage)
    else:
        put = partial(code.i, LOAD_CONST)

    # locals
    for obj_collection, known_as, store_opcode in [
        (zip(p.code.co_varnames, p.v_locals), "locals", STORE_FAST),
    ]:
        code.c(f"!unpack {known_as}")
        for obj_name, obj_in_collection in obj_collection:
            if obj_in_collection is not NULL:
                put(obj_in_collection)
                code.i(store_opcode, obj_name)

    # globals
    for obj_collection, known_as, store_opcode in [
        # (p.v_locals, "locals", STORE_FAST),
        (module_globals, "globals", STORE_NAME),
    ]:
        if obj_collection is not None and len(obj_collection) > 0:
            code.c(f"!unpack {known_as}")
            klist, vlist = zip(*obj_collection.items())
            put(vlist)
            code.i(UNPACK_SEQUENCE, len(vlist))
            for k in klist:
                # k = v
                code.i(store_opcode, k)

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
                code.i(SETUP_FINALLY, lookup_orig[item.handler * jump_multiplier])
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
                code.i(BUILD_TUPLE, 0)
                code.i(CALL_FUNCTION_EX, 0)
            elif isinstance(nxt, CodeType):
                if python_feature_make_function_qualname:
                    put(f"morph_into:{f_code.co_name}")
                code.i(MAKE_FUNCTION, 0)
                code.i(BUILD_TUPLE, 0)
                code.i(CALL_FUNCTION_EX, 0)
            else:
                raise ValueError(f"cannot call {nxt}")

    # now jump to the previously saved position
    if p.current_opcode is not None:
        code.c("!final jump")
        code.i(JUMP_FORWARD, code.instructions[code.instructions.index(code.current) + 1])

    code.c("-----------------")
    code.c("Original bytecode")
    code.c("-----------------")

    # add signature
    code.sign()

    if object_storage is not None and object_storage_protocol is not None:
        load_storage_handle.instruction = ConstInstruction(
            load_storage_handle.instruction.opcode,
            object_storage_protocol.save_to_code(object_storage),
        )

    # finalize
    starting = code.instructions[0]
    starting.metadata.stack_size = guess_entering_stack_size(starting.instruction.opcode)
    assign_stack_size(code.instructions, clean_start=False)
    code.print(log_bytecode)
    assembled = code.assemble()
    bytecode_data = bytes(assembled)

    init_args = dict(
        argcount=0,
        posonlyargcount=0,
        kwonlyargcount=0,
        nlocals=len(assembled.varnames),
        stacksize=max(i.metadata.stack_size or 0 for i in code.instructions),
        flags=flags,
        code=bytecode_data,
        consts=tuple(assembled.consts),
        names=tuple(assembled.names),
        varnames=tuple(assembled.varnames),
        freevars=tuple(assembled.cells),
        cellvars=tuple(),
        filename=f_code.co_filename,  # TODO: something different should be here
        name=f_code.co_name,
        firstlineno=f_code.co_firstlineno,  # TODO: this has to be fixed
        linetable=f_code.co_lnotab,
        exceptiontable=None,
    )
    if "qualname" in code_object_args:
        init_args["qualname"] = f_code.co_qualname
    if "exceptiontable" in code_object_args:
        if f_code.co_exceptiontable:
            raise ValueError(str(f_code.co_exceptiontable))
        init_args["exceptiontable"] = f_code.co_exceptiontable
    init_args = tuple(init_args[f"{i}"] for i in code_object_args)
    result = CodeType(*init_args)

    return FunctionType(
        result,
        p.v_globals,
        name=f"morph_into:{p.code.co_name}",
        closure=tuple(p.v_cells),
    )


def morph_stack(frame_data, tos=None, object_storage_protocol=None, root_unpack_globals=False, **kwargs):
    """
    Morphs the stack.

    Parameters
    ----------
    frame_data : list
        States of all individual frames.
    tos : object
        Top-of-stack object for the executing frame.
    object_storage_protocol : storage_protocol
        If specified, the root frame unpacks the object storage.
    root_unpack_globals : bool
        If True, unpacks globals in the root frame.
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
        is_root = frame is frame_data[-1]
        tos = morph_into(
            frame, tos,
            call_nxt=not is_topmost,
            object_storage_protocol=object_storage_protocol if is_root else None,
            module_globals=frame.v_globals if is_root and root_unpack_globals else None,
            **kwargs
        )
    return tos
