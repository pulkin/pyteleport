from collections.abc import Sequence
from dataclasses import dataclass, field
from dis import opname as dis_opname, stack_effect
from math import ceil
from opcode import HAVE_ARGUMENT, EXTENDED_ARG, opname
from typing import Optional

from shutil import get_terminal_size

from .opcodes import LOAD_GLOBAL, python_feature_cache, python_feature_jump_2x, python_feature_load_global_null
from .printing import truncate, int_diff
from .util import IndexStorage, NameStorage

if python_feature_jump_2x:
    jump_multiplier = 2
else:
    jump_multiplier = 1

if python_feature_cache:
    from opcode import _inline_cache_entries
else:
    _inline_cache_entries = (0,) * 256

max_opname_len = max(map(len, dis_opname))
max_op_len = max_opname_len + 38
no_step_opcodes = set()
for _name in "JUMP_ABSOLUTE", "JUMP_FORWARD", "JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "RETURN_VALUE", "RERAISE", "RAISE_VARARGS":
    try:
        no_step_opcodes.add(opname.index(_name))
    except ValueError:
        pass


def byte_len(i: int) -> int:
    return int(ceil((i or 1).bit_length() / 8))


class AbstractBytecodePrintable:
    """
    Anything that can be printed instruction-like.
    """

    def pprint(self):
        """Subclasses ensure it is properly printable"""
        raise NotImplementedError


@dataclass(frozen=True)
class AbstractInstruction(AbstractBytecodePrintable):
    opcode: int
    """
    A parent class for any instruction.

    Parameters
    ----------
    opcode
        Instruction opcode.
    """

    def __post_init__(self):
        assert isinstance(self.opcode, int)
        assert 0 <= self.opcode < 0x100

    @property
    def opname(self) -> str:
        return dis_opname[self.opcode]

    @property
    def size_tail(self):
        return 2 * _inline_cache_entries[self.opcode]

    @property
    def size(self):
        return self.size_tail + 2

    def __str__(self):
        return self.opname

    def pprint(self, width: int = max_opname_len):
        return truncate(self.opname, width)


@dataclass(frozen=True)
class AbstractArgInstruction(AbstractInstruction):
    opcode: int
    """
    A base class for instructions with an argument.
    Subclasses are required to have an arg field.

    Parameters
    ----------
    opcode
        Instruction opcode.
    """

    def get_stack_effect(self, jump: bool = False) -> int:
        raise NotImplementedError

    def __str_arg__(self):
        return str(self.arg)

    def __str__(self):
        return f"{self.opname}({self.__str_arg__()})"

    def pprint(self, width: int = max_op_len, opname_width: int = max_opname_len):
        arg_width = width - opname_width - 1
        if arg_width < 4:
            return super().pprint(width)
        return f"{truncate(self.opname, opname_width).ljust(opname_width)} {truncate(self.__str_arg__(), arg_width)}"


@dataclass(frozen=True)
class EncodedInstruction(AbstractArgInstruction):
    arg: int
    """
    Instruction with an encoded (positive integer) argument.

    Parameters
    ----------
    opcode
        Instruction opcode.
    arg
        Integer argument.
    """

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.arg, int)
        assert 0 <= self.arg, f"negative arg: {self.arg}"

    @property
    def size_arg(self):
        return byte_len(self.arg)

    @property
    def size_ext(self):
        return self.size + 2 * (self.size_arg - 1)

    def get_stack_effect(self, jump: bool = False) -> int:
        if self.opcode < HAVE_ARGUMENT:
            arg = None
        else:
            arg = self.arg
        return stack_effect(self.opcode, arg, jump=jump)

    def __bytes__(self):
        arg = self.arg.to_bytes(self.size_arg, 'big')
        result = []
        for a in arg[:-1]:
            result.append(EXTENDED_ARG)
            result.append(a)
        result.append(self.opcode)
        result.append(arg[-1])
        return bytes(result) + b'\x00' * self.size_tail


@dataclass(eq=False)
class FixedCell(AbstractBytecodePrintable):
    offset: int
    is_jump_target: bool
    instruction: Optional[EncodedInstruction] = None
    """
    An instruction cell at a specific offset, possibly
    occupied by an instruction.

    Parameters
    ----------
    offset
        Instruction offset.
    is_jump_target
        If True, indicates that this slot is referenced.
    instruction
        An instruction occupying this slot.
    """

    def __str__(self):
        result = f"FixedCell[pos={self.offset}]({str(self.instruction)})"
        if self.is_jump_target:
            result += "*"
        return result

    def pprint(self, width: int = 0, offset_width: int = 4):
        if width == 0:
            width, _ = get_terminal_size()
        instr_width = width - offset_width - 1
        if self.instruction is None:
            inner = truncate("None", instr_width, left="<", right=">")
        else:
            inner = self.instruction.pprint(width=instr_width)
        offset = truncate(str(self.offset), offset_width, suffix="..")
        return f"{offset.rjust(offset_width)} {inner}"


@dataclass(frozen=True)
class NoArgInstruction(AbstractInstruction):
    arg: int = 0
    """
    An instruction that does not require any
    arguments.

    Parameters
    ----------
    opcode
        Instruction opcode.
    arg
        Instruction argument. Simply keeps
        whatever argument provided, without
        any meaning assigned.
    """

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.arg, int)
        assert 0 <= self.arg

    def get_stack_effect(self, jump: bool = False) -> int:
        assert not jump
        return stack_effect(self.opcode, None)

    def encode(self) -> EncodedInstruction:
        return EncodedInstruction(self.opcode, self.arg)


@dataclass(frozen=True)
class NameInstruction(AbstractArgInstruction):
    arg: str
    """
    Instruction with a name (string) argument.

    Parameters
    ----------
    opcode
        Instruction opcode.
    arg
        Name argument.
    """

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.arg, str)

    def get_stack_effect(self, jump: bool = False) -> int:
        assert not jump
        return stack_effect(self.opcode, 0)

    @staticmethod
    def from_args(code: int, arg: int, lookup: Sequence[str]):
        if python_feature_load_global_null and code == LOAD_GLOBAL:
            return NameInstruction2(code, lookup[arg >> 1], bool(arg % 2))
        else:
            return NameInstruction(code, lookup[arg])

    def encode(self, storage: NameStorage) -> EncodedInstruction:
        return EncodedInstruction(self.opcode, storage.store(self.arg))


@dataclass(frozen=True)
class NameInstruction2(NameInstruction):
    bit: bool
    """
    A flavor of NameInstruction with a special meaning of the arg lowest bit.
    Used for LOAD_GLOBAL in 3.11+.

    Parameters
    ----------
    opcode
        Instruction opcode.
    arg
        Name argument.
    bit
        The lowest bit (stands for loading NULL in LOAD_GLOBAL).
    """

    def get_stack_effect(self, jump: bool = False) -> int:
        return super().get_stack_effect(jump=jump) + self.bit

    def encode(self, storage: NameStorage) -> EncodedInstruction:
        return EncodedInstruction(self.opcode, storage.store(self.arg) << 1 + self.bit)


@dataclass(frozen=True)
class ConstInstruction(AbstractArgInstruction):
    arg: object
    """
    Instruction with a constant (object) argument.

    Parameters
    ----------
    opcode
        Instruction opcode.
    arg
        Object argument.
    """

    def get_stack_effect(self, jump: bool = False) -> int:
        assert not jump
        return stack_effect(self.opcode, 0)

    def __str_arg__(self):
        return repr(self.arg)

    def encode(self, storage: IndexStorage) -> EncodedInstruction:
        return EncodedInstruction(self.opcode, storage.store(self.arg))


@dataclass(eq=False)
class FloatingMetadata:
    uid: Optional[object] = None
    source: Optional[FixedCell] = None
    _stack_size: Optional[int] = None
    mark_current: Optional[bool] = None
    """
    Metadata for the floating cell.
    
    Parameters
    ----------
    source
        The corresponding fixed cell for this floating cell.
    stack_size
        The number of items in the value stack *before*
        the corresponding instruction is executed.
    mark_current
        Marks this instruction as "current" (i.e. to be
        executed next).
    """

    @property
    def stack_size(self) -> int:
        return self._stack_size

    @stack_size.setter
    def stack_size(self, value: Optional[int]):
        if value is not None and value < 0:
            raise ValueError(f"trying to set stack_size to negative {value=}")
        self._stack_size = value


@dataclass(eq=False)
class FloatingCell(AbstractBytecodePrintable):
    instruction: Optional[AbstractInstruction]
    referenced_by: list["FloatingCell"] = None
    metadata: FloatingMetadata = field(default_factory=FloatingMetadata)
    """
    A bytecode slot without any specific offset but
    instead keeping track of references to this slot.

    Parameters
    ----------
    instruction
        The instruction in this cell.
    referenced_by
        A list of references.
    metadata
        Optional metadata for this instruction.
    """

    def __post_init__(self):
        if self.referenced_by is None:
            self.referenced_by = []

    @property
    def is_jump_target(self):
        return bool(self.referenced_by)

    def swap_with(self, another: "FloatingCell") -> None:
        """
        Reference another floating cell.
        This makes `self` safe to delete.

        Parameters
        ----------
        another
            Another cell to reference.
        """
        for i in set(self.referenced_by):
            i.relink_to(another)
        assert len(self.referenced_by) == 0

    def __str__(self):
        uid = ""
        if self.metadata.uid is not None:
            uid = f"[pos={self.metadata.uid}]"
        result = f"FloatingCell{uid}({str(self.instruction)})"
        if self.is_jump_target:
            result += "*"
        return result

    def pprint(self, width: int = 0, width_stack_size: int = 16, width_uid=4):
        if width == 0:
            width, _ = get_terminal_size()
        instr_width = width - width_stack_size - width_uid - 2
        if self.instruction is None:
            result = truncate("None", instr_width, left="<", right=">")
        else:
            result = self.instruction.pprint(width=instr_width)
        if self.metadata.stack_size is None or self.instruction is None:
            stack_size = " ?"
        else:
            try:
                delta = self.instruction.get_stack_effect(jump=False)
            except AssertionError:
                delta = 0
            stack_size = int_diff(self.metadata.stack_size, max(0, delta), max(0, -delta), width_stack_size)
        uid = ""
        if self.metadata.uid is not None:
            uid = truncate(str(self.metadata.uid), width_uid, suffix="..")
        uid = uid.rjust(width_uid)
        return f"{uid} {result.ljust(instr_width)} {stack_size}"


@dataclass(frozen=True)
class ReferencingInstruction(AbstractArgInstruction):
    arg: FloatingCell
    """
    Instruction referencing another one through the
    argument.

    Parameters
    ----------
    opcode
        Instruction opcode.
    arg
        Reference to another instruction.
    """

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.arg, FloatingCell)

    def get_stack_effect(self, jump: bool = False) -> int:
        if self.opcode in no_step_opcodes:
            assert jump
        return stack_effect(self.opcode, 0, jump=jump)

    def __str_arg__(self, size: Optional[int] = None):
        base = ["to"]
        uid = self.arg.metadata.uid
        if uid is not None:
            base.append(str(uid))
        if isinstance(self.arg.instruction, ReferencingInstruction):
            base.append(self.arg.instruction.opname)
        else:
            base.append(str(self.arg.instruction))
        return truncate(" ".join(base), size)
