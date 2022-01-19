from dataclasses import dataclass
from types import FunctionType
from itertools import count

import dis
from dis import HAVE_ARGUMENT, stack_effect
import sys

from .bytecode import (
    LOAD_CONST,
    RETURN_VALUE,
    EXTENDED_ARG,
    NOP,
    POP_JUMP_IF_FALSE,
    interrupting,
    resuming,
)
from .util import unique_name


def long2bytes(l):
    result = tuple(map(int, l.to_bytes((l.bit_length() + 7) // 8, byteorder="big")))
    assert len(result) < 5
    if len(result) == 0:
        return 0,
    return result


def _trunc(s, l):
    if len(s) <= l:
        return s
    else:
        return s[:l-3] + "..."


def get_jump_multiplier() -> int:
    """
    Computes jump multiplier.

    Returns
    -------
    Jump multiplier.
    """
    def _challenge():
        if something:
            pass
    bytecode = _challenge.__code__.co_code
    assert bytecode[2] == POP_JUMP_IF_FALSE
    assert bytecode[-4:] == bytes((LOAD_CONST, 0, RETURN_VALUE, 0))
    arg = bytecode[3]
    target = len(bytecode) - 4
    if arg == target:
        return 1
    elif arg * 2 == target:
        return 2
    else:
        dis.dis(_challenge, file=sys.stderr)
        sys.stderr.flush()
        raise RuntimeError(f"Failed to determine jump multiplier with arg={arg} and target={target}")


jump_multiplier = get_jump_multiplier()


@dataclass
class Instruction:
    """Represents a single opcode"""
    opcode: int
    arg: int
    pos: int = 0
    len: int = 2
    jump_to: "Instruction" = None
    stack_size: int = None

    @property
    def is_jrel(self):
        return self.opcode in dis.hasjrel

    @property
    def is_jump(self):
        return self.opcode in dis.hasjabs

    @property
    def is_any_jump(self):
        return self.is_jrel or self.is_jump

    @property
    def pos_last(self):
        return self.pos + self.len - 2

    def get_stack_effect(self, jump=None):
        result = self.opcode in resuming
        if self.opcode < HAVE_ARGUMENT:
            return result + stack_effect(self.opcode)
        else:
            if self.is_any_jump:
                return result + stack_effect(self.opcode, self.arg, jump=jump)
            else:
                return result + stack_effect(self.opcode, self.arg)

    def get_stack_after(self, jump=None):
        return self.stack_size + self.get_stack_effect(jump=jump)

    @property
    def bytes(self):
        arg_bytes = long2bytes(max(self.arg, 0))
        result = []
        for i in arg_bytes[:-1]:
            result.extend((EXTENDED_ARG, i))
        result.extend((self.opcode, arg_bytes[-1]))
        return bytes(result)

    def compute_jump(self):
        if self.is_jrel:
            return self.arg * jump_multiplier + self.pos_last + 2
        elif self.is_jump:
            return self.arg * jump_multiplier
        else:
            return None

    def assert_valid(self, prev, lookup=None):
        assert self.arg >= 0, f"arg is negative: {self.arg}"
        assert len(self.bytes) == self.len, f"len is invalid: len({repr(self.bytes)}) != {self.len}"
        if prev is not None:
            assert prev.pos + prev.len == self.pos, f"pos is invalid: {prev.pos}(prev.pos) + {prev.len}(prev.len) != {self.pos}(self.pos)"
        else:
            assert self.pos == 0, f"pos is non-zero: {self.pos}"
        if lookup is not None and self.is_any_jump:
            jump_points_to = lookup.get(self.compute_jump(), None)
            assert jump_points_to is self.jump_to, f"jump_to is invalid: {repr(self.jump_to)} vs {repr(jump_points_to)}"

    def __str__(self):
        return f"{self.pos:>6d} {_trunc(dis.opname[self.opcode], 18):<18} {self.arg:<16d}"

    def __repr__(self):
        return f"{dis.opname[self.opcode]}({self.arg}, pos={self.pos}, len={self.len})"


@dataclass
class Comment:
    """Represents a comment"""
    text: str

    @property
    def printable_text(self):
        if len(self.text) < 35:
            return self.text
        return f"{self.text[:32]}..."

    def __repr__(self):
        return f"       {self.printable_text}".ljust(42)


class CList(list):
    def index_store(self, x, create_new=False):
        if create_new:
            x = unique_name(x, self)
        try:
            return self.index(x)
        except ValueError:
            self.append(x)
            return len(self) - 1
    __call__ = index_store


class Bytecode(list):
    def __init__(self, opcodes, co_names, co_varnames, co_consts, pos=None):
        super().__init__(opcodes)
        if pos is None:
            pos = len(self)
        self.pos = pos
        self.co_names = CList(co_names)
        self.co_varnames = CList(co_varnames)
        self.co_consts = CList(co_consts)

    @classmethod
    def disassemble(cls, arg, **kwargs):
        if isinstance(arg, FunctionType):
            arg = arg.__code__
        code = arg.co_code

        # Attempt to read source code
        marks = dis.findlinestarts(arg)
        try:
            lines = open(arg.co_filename, 'r').readlines()  # [arg.co_firstlineno - 1:] ??
            marks = list((i_opcode, i_line, lines[i_line - 1].strip()) for (i_opcode, i_line) in marks)
        except (TypeError, OSError, IndexError):
            marks = None

        result = cls([], arg.co_names, arg.co_varnames, arg.co_consts, **kwargs)
        arg = 0
        _len = 0
        for pos, (opcode, _arg) in enumerate(zip(code[::2], code[1::2])):
            arg = arg * 0x100 + _arg
            _len += 2
            if opcode != EXTENDED_ARG:
                start_pos = pos * 2 - _len + 2

                if marks is not None:
                    for i_mark, (mark_opcode, mark_lineno, mark_text) in enumerate(marks):
                        if mark_opcode <= start_pos:
                            result.c(f"{mark_lineno:<3d} {mark_text}")
                        else:
                            marks = marks[i_mark:]
                            break
                    else:
                        marks = []

                result.i(opcode, arg, start_pos, _len)
                arg = _len = 0
        result.eval_jumps()
        result.eval_stack()
        return result

    def i(self, opcode, arg=None, *args, **kwargs):
        if isinstance(opcode, Instruction):
            i = opcode
        else:
            i = Instruction(opcode, arg, *args, **kwargs)
        self.insert(self.pos, i)
        self.pos += 1
        return i

    def c(self, text):
        i = Comment(text)
        self.insert(self.pos, i)
        self.pos += 1
        return i

    def I(self, opcode, arg, *args, create_new=False, **kwargs):
        if opcode in dis.hasconst:
            return self.i(opcode, self.co_consts(arg, create_new=create_new), *args, **kwargs)
        elif opcode in dis.hasname:
            return self.i(opcode, self.co_names(arg, create_new=create_new), *args, **kwargs)
        elif opcode in dis.haslocal:
            return self.i(opcode, self.co_varnames(arg, create_new=create_new), *args, **kwargs)
        elif opcode in dis.hasjrel + dis.hasjabs:
            result = self.i(opcode, None)
            result.jump_to = arg
            return result
        else:
            raise ValueError(f"Unknown opcode: {dis.opname[opcode]}")

    def nop(self, arg):
        arg = bytes(arg)
        for i in arg:
            self.i(NOP, int(i))

    def iter_opcodes(self, start=None):
        if start is None:
            iterator = iter(self)
        else:
            iterator = iter(self[self.index(start):])
        for i in iterator:
            if isinstance(i, Instruction):
                yield i

    def by_pos(self, pos):
        for i in self.iter_opcodes():
            if i.pos == pos:
                return i
        else:
            raise ValueError(f"Instruction with pos={pos} not found")

    def eval_jumps(self):
        lookup = {i.pos: i for i in self.iter_opcodes()}
        for i in self.iter_opcodes():
            if i.is_any_jump and i.arg is not None:
                i.jump_to = lookup[i.compute_jump()]

    def eval_stack(self):
        for i_i, i in enumerate(self.iter_opcodes()):
            i.stack_size = 0 if i_i == 0 else None

        updated = True

        def _maybe_set_stack(_op: Instruction, _stack: int):
            nonlocal updated
            assert _stack >= 0, f"Negative stack size {_stack} at {_op.pos} for code\n{self}"
            if _op.opcode == RETURN_VALUE:
                assert _stack == 1, f"Stack size {_stack} != 1 for RETURN_VALUE"
            if _op.stack_size is not None:
                assert _op.stack_size == _stack, f"Failed to match stack_size={_stack} against previously assigned value {_op.stack_size} at pos {_op.pos} for code\n{self}"
            else:
                _op.stack_size = _stack
                updated = True

        while updated:
            updated = False
            prev_instruction = None
            for i in self.iter_opcodes():
                # no-jump
                if prev_instruction is not None and prev_instruction.stack_size is not None and \
                        prev_instruction.opcode not in interrupting:
                    _maybe_set_stack(i, prev_instruction.get_stack_after(jump=False))

                # jump
                if i.stack_size is not None and i.is_any_jump:
                    _maybe_set_stack(i.jump_to, i.get_stack_after(jump=True))

                prev_instruction = i

    def assign_pos(self):
        pos = 0
        for i in self.iter_opcodes():
            i.pos = pos
            pos += i.len

    def assign_jump_args(self):
        for i in self.iter_opcodes():
            if i.is_any_jump and i.jump_to is not None:
                if i.is_jump:
                    i.arg = i.jump_to.pos // jump_multiplier
                elif i.is_jrel:
                    i.arg = (i.jump_to.pos - i.pos_last - 2) // jump_multiplier

    def assign_len(self):
        for i in self.iter_opcodes():
            i.len = len(i.bytes)

    def assert_valid(self):
        prev = None
        lookup = {i.pos: i for i in self.iter_opcodes()}
        for opcode in self.iter_opcodes():
            opcode.assert_valid(prev, lookup=lookup)
            prev = opcode

    def get_bytecode(self):
        for i in range(5):
            self.assign_jump_args()
            self.assign_len()
            self.assign_pos()
            try:
                self.assert_valid()
                break
            except AssertionError:
                pass
        else:
            self.assert_valid()  # re-raise
        return b''.join(i.bytes for i in self.iter_opcodes())

    def __str__(self):
        lookup = {id(i): i_i for i_i, i in enumerate(self)}
        connections = []
        for _ in self:
            connections.append([])
        for i_i, i in enumerate(self):
            if isinstance(i, Instruction) and i.jump_to is not None:
                i_j = lookup[id(i.jump_to)]
                i_mn = min(i_i, i_j)
                i_mx = max(i_i, i_j)
                occupied = set(sum(connections[i_mn:i_mx+1], []))
                for slot in count():
                    if slot not in occupied:
                        break
                for _ in range(i_mn, i_mx+1):
                    connections[_].append(slot)
        lines = []
        for i_i, (i, c, c_prev, c_next) in enumerate(zip(
                self, connections, [[]] + connections[:-1], connections[1:] + [[]])):
            if len(c) == 0:
                lines.append(str(i))
            else:
                _str = []
                for _ in range(max(c) + 1):
                    if _ in c:
                        if _ in c_prev and _ in c_next:
                            _str.append("┃")
                        elif _ in c_prev and _ not in c_next:
                            _str.append("┛")
                        elif _ not in c_prev and _ in c_next:
                            _str.append("┓")
                        else:
                            _str.append("@")
                    else:
                        _str.append(" ")
                lines.append((">" if i_i == self.pos else "") + str(i) + ''.join(_str))

            if isinstance(i, Instruction):
                represented, arg = _repr_arg(i.opcode, i.arg, self)
                if represented:
                    arg_repr = repr(arg)
                    if len(arg_repr) > 24:
                        arg_repr = f"<{type(arg).__name__} instance>"
                    lines[-1] = lines[-1] + f" ({arg_repr})"
                if i.stack_size is not None:
                    lines[-1] += f" stack={i.stack_size:d}"
        return '\n'.join(lines)


def _repr_arg(opcode, arg, code):
    if opcode in dis.hasconst:
        return True, code.co_consts[arg]
    elif opcode in dis.haslocal:
        return True, code.co_varnames[arg]
    elif opcode in dis.hasname:
        return True, code.co_names[arg]
    else:
        return False, arg


def _repr_opcode(opcode, arg, code):
    head = f"{dis.opname[opcode]:>20} {arg: 3d}"
    represented, val = _repr_arg(opcode, arg, code)
    if represented:
        return f"{head} {'(' + repr(val) + ')':<12}"
    else:
        return f"{head}" + " " * 13


def _dis(code_obj, alt=None):
    code = code_obj.co_code
    if alt is None:
        alt = code
    result = list(zip(code[::2], code[1::2], alt[::2], alt[1::2]))
    result_repr = []
    for i, (opc_old, arg_old, opc_new, arg_new) in enumerate(result):
        i *= 2
        if (opc_new, arg_new) == (opc_old, arg_old):
            result_repr.append((f"{i: 3d} {_repr_opcode(opc_new, arg_new, code_obj)}",))
        else:
            result_repr.append(("\033[94m", f"{i: 3d} {_repr_opcode(opc_new, arg_new, code_obj)} {_repr_opcode(opc_old, arg_old, code_obj)}", "\033[0m"))
    return result_repr


def cdis(code_obj, alt=None):
    return "\n".join(''.join(i) for i in _dis(code_obj, alt=alt))



