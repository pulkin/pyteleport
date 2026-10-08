from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from dis import get_instructions as dis_get_instructions, _get_code_object, Instruction
from functools import partial
from io import StringIO
import logging
from types import CodeType, FrameType
from typing import Callable, Optional, Iterable, Iterator, Sequence

from .primitives import AbstractBytecodePrintable, FixedCell, FloatingCell, EncodedInstruction, ReferencingInstruction, \
    NoArgInstruction, ConstInstruction, NameInstruction, jump_multiplier, ExceptionCodeBlock
from .util import IndexStorage, NameStorage, Cell, log_iter
from .sequence_assembler import LookBackSequence, assemble as assemble_sequence
from .opcodes import guess_entering_stack_size, RETURN_VALUE, \
    python_feature_exceptiontable, interrupting, python_feature_f_lasti_is_offset, LOAD_DEREF, \
    python_feature_free_locals_plus, python_feature_all_locals_plus
from .exceptiontable import unpack_exception_table
from .patched_opcode import EXTENDED_ARG, HAVE_ARGUMENT, opmap, hasjrel, hasjabs, hasconst, hasname, haslocal, hasfree, opname

NOP = opmap["NOP"]


jrel_bw = {
    i
    for i in hasjrel
    if "JUMP_BACKWARD" in opname[i]
}


def offset_to_jump(opcode: int, offset: int, next_pos: Optional[int], x: int = jump_multiplier) -> int:
    """
    Computes jump argument from the provided offset information.

    Parameters
    ----------
    opcode
        The jumping opcode.
    offset
        Jump destination.
    next_pos
        Offset of the instruction following the jump opcode.
    x
        The jump multiplier.

    Returns
    -------
    The resulting argument.
    """
    if opcode in hasjabs:
        return offset // x
    elif opcode in hasjrel:
        result = (offset - next_pos) // x
        if opcode in jrel_bw:
            result = - result
        return result
    else:
        raise ValueError(f"{opcode=} {opname[opcode]} is not jumping")


def jump_to_offset(opcode: int, arg: int, next_pos: Optional[int], x: int = jump_multiplier) -> int:
    """
    Computes jump argument from the provided offset information.

    Parameters
    ----------
    opcode
        The jumping opcode.
    arg
        Jump argument.
    next_pos
        Offset of the instruction following the jump opcode.
    x
        The jump multiplier.

    Returns
    -------
    The resulting argument.
    """
    if opcode in hasjabs:
        return arg * x
    elif opcode in hasjrel:
        if opcode in jrel_bw:
            arg = - arg
        return arg * x + next_pos
    else:
        raise ValueError(f"{opcode=} {opname[opcode]} is not jumping")


def iter_slots(source, exception_table: Optional[Iterable[ExceptionCodeBlock]] = None) -> Iterator[FixedCell]:
    """
    Generates slots from the raw bytecode data.

    Parameters
    ----------
    source
        The source of instructions.
    exception_table
        An exception table if python uses it.

    Yields
    ------
    Bytecode slots with instructions inside.
    """
    by_pos = {
        instruction.offset: FixedCell(
            offset=instruction.offset,
            is_jump_target=instruction.is_jump_target,
            instruction=EncodedInstruction(
                opcode=instruction.opcode,
                arg=instruction.arg or 0,
            ),
        )
        for instruction in source
    }

    if exception_table is not None:
        for item in exception_table:
            item.map(by_pos)

    yield from by_pos.values()


def get_instructions(code: CodeType) -> Iterator[Instruction]:
    """
    A replica of `dis.get_instructions` with minor
    modifications.

    Parameters
    ----------
    code
        The code to parse instructinos from.

    Yields
    ------
    Individual instructions.
    """
    for instruction in dis_get_instructions(code):
        if instruction.arg is None:
            arg = code.co_code[instruction.offset + 1]
            instruction = instruction._replace(arg=arg)
        yield instruction


def iter_extract(source) -> tuple[Iterable[FixedCell], Optional[list[ExceptionCodeBlock]], CodeType]:
    """
    Iterates over bytecodes from the source.

    Parameters
    ----------
    source
        Anything with the bytecode.

    Returns
    ------
    Bytecode iterator and the corresponding code object.
    """
    code_obj = _get_code_object(source)
    exception_table = None
    if python_feature_exceptiontable:
        exception_table = unpack_exception_table(code_obj)
    return iter_slots(get_instructions(code_obj), exception_table), exception_table, code_obj


def filter_ext_arg(source: Iterable[FixedCell]) -> Iterator[FixedCell]:
    """
    Filters out NOP and EXT_ARG.

    Parameters
    ----------
    source
        The source of bytecode instructions.

    Yields
    ------
    Filtered instructions.
    """
    head = None
    for slot in source:
        if head is not None:
            # all references need to be to head
            assert not slot.is_jump_target
        else:
            head = slot

        if slot.instruction.opcode is not EXTENDED_ARG:
            slot = FixedCell(
                offset=head.offset,
                is_jump_target=head.is_jump_target,
                instruction=slot.instruction,
            )
            head = None
            yield slot


def iter_dis_arg_to_offset(source: Iterable[FixedCell]) -> Iterator[FixedCell]:
    """
    Computes jump offsets from jump arguments.

    Parameters
    ----------
    source
        The source of bytecode slots.

    Yields
    ------
    Bytecode where all jump arguments are offsets.
    """
    for fixed_cell in source:
        instruction = fixed_cell.instruction

        if instruction.opcode in hasjabs or instruction.opcode in hasjrel:
            if instruction.opcode in hasjabs or instruction.opcode in hasjrel:
                fixed_cell.instruction = EncodedInstruction(
                    opcode=instruction.opcode,
                    arg=jump_to_offset(
                        instruction.opcode,
                        instruction.arg,
                        fixed_cell.offset + instruction.size_full,
                    ),
                )
        yield fixed_cell


def iter_dis_build_references(
        source: Iterable[FixedCell],
        exception_table: Optional[Iterable[ExceptionCodeBlock]] = None,
) -> Iterator[FloatingCell]:
    """
    Computes jumps.

    Parameters
    ----------
    source
        The source of bytecode slots.
    exception_table
        An exception table if python uses it.

    Yields
    ------
    FloatingCell
        The resulting cell with the referencing information.
    """
    lookup: dict[int, FloatingCell] = defaultdict(lambda: FloatingCell(instruction=None))

    for fixed_cell in source:
        instruction = fixed_cell.instruction

        # if it is a jump process it and create jump destination if not exists
        jumps_to = None
        if instruction.opcode in hasjabs or instruction.opcode in hasjrel:
            jumps_to = lookup[instruction.arg]
            instruction = ReferencingInstruction(
                opcode=instruction.opcode,
                arg=jumps_to,
            )

        floating_cell = lookup[fixed_cell.offset]
        floating_cell.instruction = instruction

        if jumps_to is not None:
            jumps_to.referenced_by.append(floating_cell)

        yield floating_cell

    if exception_table is not None:
        for item in exception_table:
            item.map(dict(lookup), key=lambda i: i.offset)


def iter_dis_args(
        source: Iterable[FloatingCell],
        consts: Sequence[object],
        names: Sequence[str],
        varnames: Sequence[str],
        cellnames: Sequence[str],
) -> Iterator[FloatingCell]:
    """
    Pipes instructions from the input and computes object arguments.

    Parameters
    ----------
    source
        The source of bytecode instructions.
    consts
        A list of constants.
    names
        A list of names.
    varnames
        A list of local names.
    cellnames
        A list of cells.

    Yields
    ------
    Instructions with computed args.
    """
    for slot in source:
        instruction = slot.instruction
        if isinstance(instruction, EncodedInstruction):
            opcode = instruction.opcode
            arg = instruction.arg

            if opcode < HAVE_ARGUMENT:
                result = NoArgInstruction(opcode, arg)
            else:
                if opcode in hasconst:
                    result = ConstInstruction(opcode, consts[arg])
                elif opcode in hasname:
                    result = NameInstruction.from_args(opcode, arg, names)
                elif opcode in haslocal:
                    if python_feature_all_locals_plus:
                        result = NameInstruction.from_args(opcode, arg, (*varnames, *cellnames))
                    else:
                        result = NameInstruction.from_args(opcode, arg, varnames)
                elif opcode in hasfree:
                    if python_feature_free_locals_plus:
                        result = NameInstruction.from_args(opcode, arg, (*varnames, *cellnames))
                    else:
                        result = NameInstruction.from_args(opcode, arg, cellnames)
                else:
                    result = EncodedInstruction(opcode, arg)

            # crack LOAD_FAST_LOAD_FAST into two separate opcodes
            if not isinstance(result, list):
                result = [result]
            for instr in result:
                slot.instruction = instr
                yield slot
                slot = FloatingCell(None)
        else:
            yield slot


def iter_dis(
        source: Iterable[FixedCell],
        consts: Sequence[object],
        names: Sequence[str],
        varnames: Sequence[str],
        cellnames: Sequence[str],
        exception_table: Optional[Iterable[ExceptionCodeBlock]] = None,
        current: Optional[FixedCell] = None,
) -> Iterator[FloatingCell]:
    """
    Disassembles encoded instructions.
    The reverse of iter_as.

    Parameters
    ----------
    source
        The source of encoded instructions.
    consts
    names
    varnames
    cellnames
        Constant and name collections.
    exception_table
        An exception table if python uses it.
    current
        Corresponds to currently executed opcode.

    Yields
    ------
    FloatingCell
        The resulting cell with the referencing information.
    """
    cell_fixed = Cell()

    for i, result in enumerate(iter_dis_args(
            iter_dis_build_references(
                iter_dis_arg_to_offset(
                    filter_ext_arg(
                        log_iter(source, cell_fixed),
                    ),
                ),
                exception_table=exception_table,
            ),
            consts,
            names,
            varnames,
            cellnames,
    )):
        fixed: FixedCell = cell_fixed.value
        result.metadata.source = fixed
        result.metadata.uid = i
        result.metadata.mark_current = fixed is current

        yield result


def iter_as_args(
        source: Iterable[FloatingCell],
        consts: IndexStorage,
        names: NameStorage,
        varnames: NameStorage,
        cellnames: NameStorage,
        dry_run: bool = False,
) -> Iterator[FloatingCell]:
    """
    Pipes instructions from the input and assembles their
    object and name arguments.

    Parameters
    ----------
    source
        The source of bytecode instructions.
    consts
        Constant storage (modified by this iterator).
    names
        Name storage (modified by this iterator).
    varnames
        Variable name storage (modified by this iterator).
    cellnames
        Cell name storage (modified by this iterator).
    dry_run
        If True, updates all storages without modifying iterator slots.

    Yields
    ------
    Instructions with assembled args.
    """
    for slot in source:
        instruction = slot.instruction
        opcode = instruction.opcode

        if isinstance(instruction, ConstInstruction):
            result = instruction.encode(consts)
        elif isinstance(instruction, NameInstruction):
            if opcode in hasname:
                result = instruction.encode(names)
            elif opcode in haslocal:
                result = instruction.encode(varnames)
            elif opcode in hasfree:
                result = instruction.encode(cellnames)
            else:
                raise ValueError(f"unknown name instruction to process: {instruction}")
        elif isinstance(instruction, NoArgInstruction):
            result = instruction.encode()
        elif isinstance(instruction, (ReferencingInstruction, EncodedInstruction)):
            result = instruction
        else:
            raise ValueError(f"unknown instruction to process: {instruction}")

        if not dry_run:
            slot.instruction = result

        yield slot


def as_jumps(
        source: Iterable[FloatingCell],
        exception_table: Optional[Iterable[ExceptionCodeBlock]] = None,
) -> list[FixedCell]:
    """
    Pipes instructions from the input and assembles
    jump destinations.

    Parameters
    ----------
    source
        The source of bytecode instructions.
    exception_table
        An exception table if python uses it.

    Returns
    -------
    Instructions with assembled jumps.
    """

    class CellToken:
        def __init__(self, cell: FloatingCell, cell_lookup: dict[FloatingCell, "CellToken"]):
            instruction = cell.instruction
            self.backward_reference_token = None
            if isinstance(instruction, ReferencingInstruction):
                try:
                    self.backward_reference_token = cell_lookup[instruction.arg]
                except KeyError:
                    pass
                instruction = EncodedInstruction(
                    opcode=instruction.opcode,
                    arg=0,
                )

            elif not isinstance(instruction, EncodedInstruction):
                raise ValueError(f"cannot init with instruction: {instruction}")

            self.cell = FixedCell(
                offset=0,
                is_jump_target=bool(cell.referenced_by),
                instruction=instruction,
            )
            self.earlier_references_to_here = ref = []
            for i in cell.referenced_by:
                try:
                    ref.append(cell_lookup[i])
                except KeyError:
                    pass

        def update_sequentially(self, prev: Optional["CellToken"]):
            # update offset
            if prev is None:
                self.cell.offset = 0
            else:
                self.cell.offset = prev.cell.offset + prev.cell.instruction.size_full
            # if jump: update arg
            if self.backward_reference_token is not None:
                self.update_jump(self.backward_reference_token)

        def update_jump(self, reference: "CellToken") -> bool:
            opcode = self.cell.instruction.opcode
            arg = offset_to_jump(
                opcode,
                reference.cell.offset,
                self.cell.offset + self.cell.instruction.size_full,
                jump_multiplier,
            )
            old_size = self.cell.instruction.n_bytes_arg
            self.cell.instruction = EncodedInstruction(
                opcode=opcode,
                arg=arg,
            )
            return self.cell.instruction.n_bytes_arg != old_size

    source = list(source)
    lookup = {}
    for floating in source:
        lookup[floating] = CellToken(floating, lookup)

    result = LookBackSequence(lookup[i] for i in source)
    assemble_sequence(result)
    result.reset()
    if exception_table is not None:
        lookup = {k: v.cell for k, v in lookup.items()}
        for item in exception_table:
            item.map(lookup)
    return list(i.cell for _, i in result)


def iter_as(
        source: Iterable[FloatingCell],
        consts: Optional[Sequence] = None,
        names: Optional[Sequence] = None,
        varnames: Optional[Sequence] = None,
        cells: Optional[Sequence] = None,
        exception_table: Optional[Iterable[ExceptionCodeBlock]] = None,
) -> tuple[
    Iterable[FixedCell],
    IndexStorage,
    NameStorage,
    NameStorage,
    NameStorage,
]:
    """
    Assembles decoded instructions.
    The reverse of iter_dis.

    Parameters
    ----------
    source
        The source of decoded instructions.
    consts
        Initial constants.
    names
    varnames
    cells
        Initial names.
    exception_table
        An exception table if python uses it.

    Returns
    -------
    The resulting bytecode, consts, names, varnames, and cellnames.
    """
    consts = IndexStorage(consts or [])
    names = NameStorage(names or [])
    varnames = NameStorage(varnames or [])
    cellnames = NameStorage(cells or [])
    if python_feature_free_locals_plus or python_feature_all_locals_plus:
        # locals_plus is not empty: some bytecodes need to offset the argument by the size of varnames
        # to understand how many names we have, we process the bytecode in dry run which populates every storage
        source = list(source)
        list(iter_as_args(
            source,
            consts,
            names,
            varnames,
            cellnames,
            dry_run=True,
        ))
        # now, every storage has been populated so we know how many varnames do we have
        # lock adding new values and put the needed offset
        consts.read_only = names.read_only = varnames.read_only = cellnames.read_only = True
        cellnames.name_offset = len(varnames)
    return as_jumps(
        iter_as_args(
            source,
            consts,
            names,
            varnames,
            cellnames,
        ),
        exception_table=exception_table,
    ), consts, names, varnames, cellnames


def assign_fixed_stack_size(
        source: list[FloatingCell],
        exception_table: Optional[Iterable[ExceptionCodeBlock]] = None,
) -> None:
    """
    Determine fixed point for stack sizes and assign them.

    Parameters
    ----------
    source
        A list of instructions.
    exception_table
        An exception table if python uses it.
    """
    # the first instruction has a fixed stack size
    starting = source[0]
    starting.metadata.stack_size = guess_entering_stack_size(starting.instruction.opcode)
    if exception_table is not None:
        # exception handlers also have a fixed stack size
        for item in exception_table:
            item.target.metadata.stack_size = item.stack_size


def assign_stack_size(
        source: list[FloatingCell],
        exception_table: Optional[Iterable[ExceptionCodeBlock]] = None,
) -> None:
    """
    Computes and assigns stack size per instruction.
    The computed values are available in `item.metadata.stack_size`.

    Parameters
    ----------
    source
        Bytecode instructions.
    exception_table
        An exception table if python uses it.
    """
    # assign fixed first
    assign_fixed_stack_size(source, exception_table)
    # figure out starting points
    chains = []
    for i, (cell, nxt) in enumerate(zip(source[:-1], source[1:])):
        if cell.metadata.stack_size is not None and nxt.metadata.stack_size is None:
            chains.append(i)

    while chains:
        new_chains = []

        for starting_point in chains:
            for cell, nxt in zip(source[starting_point:], source[starting_point + 1:]):

                if isinstance(cell.instruction, ReferencingInstruction):
                    distant_stack_size = cell.metadata.stack_size + cell.instruction.get_stack_effect(jump=True)
                    distant = cell.instruction.arg
                    if distant.metadata.stack_size is None:
                        distant.metadata.stack_size = distant_stack_size
                        new_chains.append(source.index(distant))
                    else:
                        assert distant_stack_size == distant.metadata.stack_size, \
                            f"stack size computed from {cell} to {distant} (jump) mismatch: " \
                            f"{distant_stack_size} vs previous {distant.metadata.stack_size}"

                if cell.instruction.opcode not in interrupting:
                    next_stack_size = cell.metadata.stack_size + cell.instruction.get_stack_effect(jump=False)
                    if nxt.metadata.stack_size is None:
                        if nxt.instruction.opcode == RETURN_VALUE:
                            assert next_stack_size == 1, f"non-zero stack at RETURN_VALUE: {next_stack_size}"
                        try:
                            nxt.metadata.stack_size = next_stack_size
                        except ValueError as e:
                            raise ValueError(
                                f"Failed unwinding the stack size; bytecode following (failing instruction marked)\n"
                                f"{ObjectBytecode(source, exception_table=exception_table, current=nxt).to_string()}") from e
                    else:
                        assert next_stack_size == nxt.metadata.stack_size, \
                            f"stack size computed from {cell} to {nxt} (step) mismatch: " \
                            f"{next_stack_size} vs previous {nxt.metadata.stack_size}"
                else:
                    break

        chains = new_chains


@dataclass
class AbstractBytecode:
    """An abstract bytecode"""
    instructions: list[AbstractBytecodePrintable]

    def get_marks(self):
        raise NotImplementedError

    def print(self, line_printer: Callable = print) -> None:
        """
        Prints the bytecode.

        Parameters
        ----------
        line_printer
            A function printing lines.
        """
        marks = self.get_marks()
        for i in self.instructions:
            mark = marks.get(i, '').rjust(3)
            lines = iter(i.pprint().split("\n"))
            line_printer(f"{mark} {next(lines)}")
            for line in lines:
                line_printer(f"    {line}" if line else "")
        line_printer("(bytecode ends)")

    def to_string(self) -> str:
        """Prints the bytecode and return the print"""
        buffer = StringIO()
        self.print(partial(print, file=buffer))
        return buffer.getvalue()


def verify_instructions(instructions: list[FloatingCell]):
    """
    Verifies the integrity of the disassembled instructions.

    Parameters
    ----------
    instructions
        The instructions to check.
    """
    counts = Counter(instructions)
    duplicates = {k: v for k, v in counts.items() if v != 1}
    if duplicates:
        raise ValueError(f"duplicate cells: {duplicates}")
    for i, floating in enumerate(instructions):
        if floating.instruction is None:
            raise ValueError(f"empty cell: {floating}")
        if isinstance(floating.instruction, ReferencingInstruction):
            target = floating.instruction.arg
            if target not in counts:
                raise ValueError(f"instruction references outside the bytecode:\n"
                                 f"  instruction {floating}\n"
                                 f"  target {target}\n"
                                 f"  source {floating.metadata.source}\n"
                                 f"bytecode follows\n"
                                 f"{ObjectBytecode(instructions).to_string()}")
            if floating not in target.referenced_by:
                raise ValueError(f"instruction target does not contain the reverse reference:\n"
                                 f"  instruction {floating}\n"
                                 f"  target {target}\n"
                                 f"  source {floating.metadata.source}\n"
                                 f"  referenced by {target.referenced_by}\n"
                                 f"bytecode follows\n"
                                 f"{ObjectBytecode(instructions).to_string()}")


@dataclass
class ObjectBytecode(AbstractBytecode):
    instructions: list[FloatingCell]
    exception_table: Optional[list[ExceptionCodeBlock]] = None
    current: Optional[FloatingCell] = None
    """
    An object bytecode.

    Parameters
    ----------
    code
        A list of opcode cells with object arguments.
    exception_table
        An exception table if python uses it.
    current
        Current bytecode operation.
    """

    def get_marks(self):
        return {self.current: ">>>"}

    @classmethod
    def from_iterable(
            cls,
            source: Iterable[FloatingCell],
            exception_table: Optional[list[ExceptionCodeBlock]] = None,
            compute_stack_size: bool = True,
            verify: bool = True,
    ):
        instructions = []
        current = None
        for c in source:
            instructions.append(c)
            if c.metadata.mark_current:
                current = c

        if verify:
            verify_instructions(instructions)

        if compute_stack_size:
            assign_stack_size(instructions, exception_table)

        return cls(
            instructions=instructions,
            exception_table=exception_table,
            current=current,
        )

    def recompute_references(self):
        """
        Re-computes references across the bytecode.
        """
        references: dict[FloatingCell, list[FloatingCell]] = defaultdict(list)
        for i in self.instructions:
            if isinstance(i.instruction, ReferencingInstruction):
                references[i.instruction.arg].append(i)
        for i in self.instructions:
            i.referenced_by = references[i]

    def assemble(self, **kwargs) -> "AssembledBytecode":
        """
        Assembles the bytecode.

        Parameters
        ----------
        kwargs
            Arguments to `iter_as`.

        Returns
        -------
        Assembled bytecode.
        """
        self.recompute_references()
        cell = Cell()
        exception_table = self.exception_table
        if exception_table is not None:
            exception_table = [replace(i) for i in exception_table]
        code_iter, consts, names, varnames, cells = iter_as(
            log_iter(self.instructions, cell),
            exception_table=exception_table,
            **kwargs
        )
        current = None
        code = []
        for fixed in code_iter:
            code.append(fixed)
            if cell.value.metadata.mark_current:
                current = fixed
        return AssembledBytecode(
            code,
            consts,
            names,
            varnames,
            cells,
            exception_table=exception_table,
            current=current,
        )


@dataclass
class AssembledBytecode(AbstractBytecode):
    instructions: list[FixedCell]
    consts: IndexStorage
    names: NameStorage
    varnames: NameStorage
    cells: NameStorage
    exception_table: Optional[list[ExceptionCodeBlock]] = None
    current: Optional[FixedCell] = None
    """
    An assembled bytecode.
    
    Parameters
    ----------
    code
        A list of opcode cells.
    consts
    names
    varnames
    cells
        Object and name storage.
    exception_table
        An exception table if python uses it.
    current
        Current instruction.
    """

    def get_marks(self):
        return {self.current: ">>>"}

    @classmethod
    def from_code_object(cls, source, f_lasti=None):
        """
        Turns code objects into assembled bytecode.

        Parameters
        ----------
        source
            The source for the bytecode.
        f_lasti
            Current opcode indicator.

        Returns
        -------
        Assembled bytecode.
        """
        cells, exception_table, code_obj = iter_extract(source)
        cells = list(cells)
        current = None

        if f_lasti is None and isinstance(source, FrameType):
            f_lasti = source.f_lasti

        current_condition = None
        if f_lasti is not None:
            if python_feature_f_lasti_is_offset:
                def current_condition(c):
                    return c.offset + c.instruction.size_ext_arg_prefix == f_lasti
            else:
                def current_condition(c):
                    # note that f_lasti is basically always two bytes before the next instruction:
                    # f_lasti points to cache for opcodes with caches
                    return c.offset + c.instruction.size_full == f_lasti + 2

        if current_condition is not None:
            for c in cells:
                if current_condition(c):
                    current = c
                    break
            else:
                AssembledBytecode(
                    cells,
                    IndexStorage(code_obj.co_consts),
                    NameStorage(code_obj.co_names),
                    NameStorage(code_obj.co_varnames),
                    NameStorage(code_obj.co_cellvars + code_obj.co_freevars),
                    exception_table,
                ).print(logging.debug)
                raise ValueError(
                    f"{f_lasti=} does not align with any opcode location"
                )
        return AssembledBytecode(
            cells,
            IndexStorage(code_obj.co_consts),
            NameStorage(code_obj.co_names),
            NameStorage(code_obj.co_varnames),
            NameStorage(code_obj.co_cellvars + code_obj.co_freevars),
            exception_table,
            current=current,
        )

    def disassemble(self) -> ObjectBytecode:
        """
        Disassembles the bytecode.

        Returns
        -------
        The disassembled bytecode.
        """
        exception_table = self.exception_table
        if exception_table is not None:
            exception_table = list(replace(i) for i in exception_table)
        return ObjectBytecode.from_iterable(
            iter_dis(
                self.instructions,
                self.consts,
                self.names,
                self.varnames,
                self.cells,
                exception_table,
                current=self.current,
            ),
            exception_table,
        )

    def __bytes__(self):
        return b''.join(bytes(i.instruction) for i in self.instructions)


def disassemble(source, f_lasti=None) -> ObjectBytecode:
    """
    Disassembles any bytecode source.

    Parameters
    ----------
    source
        The bytecode source.
    f_lasti
        Current opcode indicator.

    Returns
    -------
    The disassembled bytecode.
    """
    return AssembledBytecode.from_code_object(source, f_lasti=f_lasti).disassemble()
