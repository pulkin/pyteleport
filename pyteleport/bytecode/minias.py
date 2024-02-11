from collections import Counter, defaultdict
from dataclasses import dataclass
from dis import get_instructions as dis_get_instructions, _get_code_object, Instruction
from functools import partial
from io import StringIO
from opcode import EXTENDED_ARG, HAVE_ARGUMENT, opmap, hasjrel, hasjabs, hasconst, hasname, haslocal, hasfree, opname
from types import CodeType, FrameType
from typing import Callable, Optional, Iterable, Iterator, Sequence

from .primitives import AbstractBytecodePrintable, FixedCell, FloatingCell, EncodedInstruction, ReferencingInstruction, \
    NoArgInstruction, ConstInstruction, NameInstruction, jump_multiplier, no_step_opcodes
from .util import IndexStorage, NameStorage, Cell, log_iter
from .sequence_assembler import LookBackSequence, assemble as assemble_sequence
from .opcodes import guess_entering_stack_size, RETURN_VALUE

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


def iter_slots(source) -> Iterator[FixedCell]:
    """
    Generates slots from the raw bytecode data.

    Parameters
    ----------
    source
        The source of instructions.

    Yields
    ------
    Bytecode slots with instructions inside.
    """
    for instruction in source:
        yield FixedCell(
            offset=instruction.offset,
            is_jump_target=instruction.is_jump_target,
            instruction=EncodedInstruction(
                opcode=instruction.opcode,
                arg=instruction.arg or 0,
            )
        )


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


def iter_extract(source) -> tuple[Iterable[FixedCell], CodeType]:
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
    return iter_slots(get_instructions(code_obj)), code_obj


def filter_nop(source: Iterable[FixedCell], keep_nop: bool = False) -> Iterator[FixedCell]:
    """
    Filters out NOP and EXT_ARG.
    Corrects offsets but does not apply extended args.

    Parameters
    ----------
    source
        The source of bytecode instructions.
    keep_nop
        If True, keeps NOP opcodes.

    Yields
    ------
    Every instruction, except no-op and extended
    args.
    """
    head = None
    for slot in source:
        if not keep_nop and slot.instruction.opcode == NOP:
            continue
        if slot.instruction.opcode == EXTENDED_ARG:
            if head is None:
                head = slot
            else:
                assert not slot.is_jump_target
        else:
            if head is not None:
                assert not slot.is_jump_target
                slot = FixedCell(
                    offset=head.offset,
                    is_jump_target=head.is_jump_target,
                    instruction=slot.instruction,
                )
                head = None
            yield slot


def iter_dis_jumps(source: Iterable[FixedCell]) -> Iterator[FloatingCell]:
    """
    Computes jumps.

    Parameters
    ----------
    source
        The source of bytecode slots.

    Yields
    ------
    FloatingCell
        The resulting cell with the referencing information.
    """
    lookup: dict[int, FloatingCell] = {}

    stack_size = 0

    for fixed_cell in source:
        original = fixed_cell.instruction

        # determine the jump destination, if any
        jump_destination = None
        if original.opcode in hasjabs or original.opcode in hasjrel:
            jump_destination = jump_to_offset(
                original.opcode,
                original.arg,
                fixed_cell.offset + original.size_ext,
            )

        # replace with the jump instruction
        jumps_to = None
        if jump_destination is not None:
            try:
                jumps_to = lookup[jump_destination]
            except KeyError:
                jumps_to = lookup[jump_destination] = FloatingCell(
                    instruction=None,
                )

            instruction = ReferencingInstruction(
                opcode=original.opcode,
                arg=jumps_to,
            )
        else:
            instruction = original

        floating_cell = None
        if fixed_cell.is_jump_target:
            try:
                # check if already in lookup and replace (fw jump)
                floating_cell = lookup[fixed_cell.offset]
            except KeyError:
                pass  # add floating_cell to lookup later (bw jump)
            else:
                floating_cell.instruction = instruction

        if floating_cell is None:
            floating_cell = FloatingCell(
                instruction=instruction,
            )
            stack_size += original.get_stack_effect(jump=False)
            if fixed_cell.is_jump_target:
                lookup[fixed_cell.offset] = floating_cell

        if jumps_to is not None:
            jumps_to.referenced_by.append(floating_cell)

        yield floating_cell


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
        A llist of cells.

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
                    result = NameInstruction(opcode, varnames[arg])
                elif opcode in hasfree:
                    result = NameInstruction(opcode, cellnames[arg])
                else:
                    result = EncodedInstruction(opcode, arg)

            slot.instruction = result
        yield slot


def iter_dis(
        source: Iterable[FixedCell],
        consts: Sequence[object],
        names: Sequence[str],
        varnames: Sequence[str],
        cellnames: Sequence[str],
        keep_nop: bool = False,
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
    keep_nop
        If True, yields NOP as they are found
        in the original bytecode.
    current
        Corresponds to currently executed opcode.

    Yields
    ------
    FloatingCell
        The resulting cell with the referencing information.
    """
    cell_fixed = Cell()

    for i, result in enumerate(iter_dis_args(
            iter_dis_jumps(filter_nop(
                log_iter(source, cell_fixed),
                keep_nop=keep_nop,
            )),
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
        cellnames: NameStorage
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
    varnames
    cellnames
        Name storages (modified by this iterator).

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

        slot.instruction = result

        yield slot


def as_jumps(source: Iterable[FloatingCell]) -> list[FixedCell]:
    """
    Pipes instructions from the input and assembles
    jump destinations.

    Parameters
    ----------
    source
        The source of bytecode instructions.

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
                self.cell.offset = prev.cell.offset + prev.cell.instruction.size_ext
            # if jump: update arg
            if self.backward_reference_token is not None:
                self.update_jump(self.backward_reference_token)

        def update_jump(self, reference: "CellToken") -> bool:
            opcode = self.cell.instruction.opcode
            arg = offset_to_jump(
                opcode,
                reference.cell.offset,
                self.cell.offset + self.cell.instruction.size_ext,
                jump_multiplier,
            )
            old_size = self.cell.instruction.size_arg
            self.cell.instruction = EncodedInstruction(
                opcode=opcode,
                arg=arg,
            )
            return self.cell.instruction.size_arg != old_size

    source = list(source)
    lookup = {}
    for floating in source:
        lookup[floating] = CellToken(floating, lookup)

    result = LookBackSequence(lookup[i] for i in source)
    assemble_sequence(result)
    result.reset()
    return list(i.cell for _, i in result)


def iter_as(
        source: Iterable[FloatingCell],
        consts: Optional[Sequence] = None,
        names: Optional[Sequence] = None,
        varnames: Optional[Sequence] = None,
        cells: Optional[Sequence] = None,
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

    Returns
    -------
    The resulting bytecode, consts, names, varnames, and cellnames.
    """
    consts = IndexStorage(consts or [])
    names = NameStorage(names or [])
    varnames = NameStorage(varnames or [])
    cellnames = NameStorage(cells or [])
    return as_jumps(iter_as_args(
        source,
        consts,
        names,
        varnames,
        cellnames,
    )), consts, names, varnames, cellnames


def assign_stack_size(source: list[FloatingCell], clean_start: bool = True) -> None:
    """
    Computes and assigns stack size per instruction.
    The computed values are available in `item.metadata.stack_size`.

    Parameters
    ----------
    source
        Bytecode instructions.
    clean_start
        If True, wipes previously computed stack sizes, if any.
    """
    if not len(source):
        return
    if clean_start:
        for i in source:
            i.metadata.stack_size = None
        starting = source[0]
        starting.metadata.stack_size = guess_entering_stack_size(starting.instruction.opcode)

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

                if cell.instruction.opcode not in no_step_opcodes:
                    next_stack_size = cell.metadata.stack_size + cell.instruction.get_stack_effect(jump=False)
                    if nxt.metadata.stack_size is None:
                        if nxt.instruction.opcode == RETURN_VALUE:
                            assert next_stack_size == 1, f"non-zero stack at RETURN_VALUE: {next_stack_size}"
                        try:
                            nxt.metadata.stack_size = next_stack_size
                        except ValueError as e:
                            raise ValueError(
                                f"Failed unwinding the stack size; bytecode following (failing instruction marked)\n"
                                f"{ObjectBytecode(source, current=nxt).to_string()}") from e
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
            line_printer(f"{mark} {i.pprint()}")

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
    current: Optional[FloatingCell] = None
    """
    An object bytecode.

    Parameters
    ----------
    code
        A list of opcode cells with object arguments.
    current
        Current bytecode operation.
    """

    def get_marks(self):
        return {self.current: ">>>"}

    @classmethod
    def from_iterable(
            cls,
            source: Iterable[FloatingCell],
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
            assign_stack_size(instructions)

        return cls(
            instructions=instructions,
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
        code_iter, consts, names, varnames, cells = iter_as(log_iter(self.instructions, cell), **kwargs)
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
            current=current,
        )


@dataclass
class AssembledBytecode(AbstractBytecode):
    instructions: list[FixedCell]
    consts: IndexStorage
    names: NameStorage
    varnames: NameStorage
    cells: NameStorage
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
    current
        Current instruction.
    """

    def get_marks(self):
        return {self.current: ">>>"}

    @classmethod
    def from_code_object(cls, source, f_lasti=None, pos=None):
        """
        Turns code objects into assembled bytecode.

        Parameters
        ----------
        source
            The source for the bytecode.
        f_lasti
        pos
            Current opcode indicators. Cannot specify both.

        Returns
        -------
        Assembled bytecode.
        """
        if f_lasti is not None and pos is not None:
            raise ValueError(f"specify either f_lasti or pos but not both")
        cells, code_obj = iter_extract(source)
        cells = list(cells)
        current = None

        if f_lasti is None and pos is None and isinstance(source, FrameType):
            f_lasti = source.f_lasti

        current_condition = None
        if f_lasti is not None:
            def current_condition(c):
                return c.following_offset == f_lasti + 2
        if pos is not None:
            def current_condition(c):
                return c.offset == pos

        if current_condition is not None:
            for c in cells:
                if current_condition(c):
                    current = c
                    break
            else:
                raise ValueError(
                    f"{f_lasti=} does not align with any opcode location"
                    if f_lasti is not None
                    else
                    f"{pos=} does not align with any opcode location"
                )
        return AssembledBytecode(
            cells,
            IndexStorage(code_obj.co_consts),
            NameStorage(code_obj.co_names),
            NameStorage(code_obj.co_varnames),
            NameStorage(code_obj.co_cellvars + code_obj.co_freevars),
            current=current,
        )

    def disassemble(self, keep_nop=False) -> ObjectBytecode:
        """
        Disassembles the bytecode.

        Parameters
        ----------
        keep_nop
            If True, won't drop NOPs.

        Returns
        -------
        The disassembled bytecode.
        """
        return ObjectBytecode.from_iterable(
            iter_dis(
                self.instructions,
                self.consts,
                self.names,
                self.varnames,
                self.cells,
                keep_nop=keep_nop,
                current=self.current,
            )
        )

    def __bytes__(self):
        return b''.join(bytes(i.instruction) for i in self.instructions)


def disassemble(source, f_lasti=None, pos=None, keep_nop=False) -> ObjectBytecode:
    """
    Disassembles any bytecode source.

    Parameters
    ----------
    source
        The bytecode source.
    f_lasti
    pos
        Current opcode indicators. Cannot specify both.
    keep_nop
        If True, yields collects as they are found in the original bytecode.

    Returns
    -------
    The disassembled bytecode.
    """
    return AssembledBytecode.from_code_object(source, f_lasti=f_lasti, pos=pos).disassemble(keep_nop=keep_nop)
