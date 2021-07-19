from dataclasses import dataclass, field
from types import CodeType, FunctionType
from itertools import count

import dis
locals().update(dis.opmap)


def long2bytes(l):
    return tuple(map(int, l.to_bytes((l.bit_length() + 7) // 8, byteorder="big")))


@dataclass
class Instruction:
    """Represents a single opcode"""
    opcode: int
    arg: int
    pos: int = 0
    len: int = 2
    jump_target: list = field(default_factory=list)
    jump_to: object = None

    @property
    def is_jrel(self):
        return self.opcode in dis.hasjrel

    @property
    def is_jump(self):
        return self.opcode in dis.hasjabs

    @property
    def bytes(self):
        arg_bytes = long2bytes(self.arg)
        if len(arg_bytes) == 0:
            arg_bytes = [0]
        assert len(arg_bytes) * 2 == self.len, f"len({arg_bytes}) != {self.len}"
        result = []
        for i in arg_bytes[:-1]:
            result.extend((EXTENDED_ARG, i))
        result.extend((self.opcode, arg_bytes[-1]))
        return bytes(result)

    def __repr__(self):
        return f"{self.pos:>6d} {dis.opname[self.opcode]:<18} {self.arg:<16d}"


class CList(list):
    def index_store(self, x):
        try:
            return self.index(x)
        except ValueError:
            self.append(x)
            return len(self) - 1
    __call__ = index_store


class Bytecode(list):
    def __init__(self, opcodes, names, varnames, consts):
        super().__init__(opcodes)
        self.pos = len(self)
        self.names = CList(names)
        self.varnames = CList(varnames)
        self.consts = CList(consts)

    @staticmethod
    def disassemble(arg):
        if isinstance(arg, FunctionType):
            arg = arg.__code__
        code = arg.co_code
        result = Bytecode([], arg.co_names, arg.co_varnames, arg.co_consts)
        arg = 0
        _len = 0
        for pos, (opcode, _arg) in enumerate(zip(code[::2], code[1::2])):
            arg = arg * 0x100 + _arg
            _len += 2
            if opcode != EXTENDED_ARG:
                result.i(opcode, arg, pos * 2 - _len + 2, _len)
                arg = _len = 0
        result.eval_jumps()
        return result

    def i(self, opcode, arg=None, *args, **kwargs):
        if isinstance(opcode, Instruction):
            i = opcode
        else:
            i = Instruction(opcode, arg, *args, **kwargs)
        self.insert(self.pos, i)
        self.pos += 1
        return i

    def I(self, opcode, arg, *args, **kwargs):
        if opcode in dis.hasconst:
            return self.i(opcode, self.consts(arg), *args, **kwargs)
        elif opcode in dis.hasname:
            return self.i(opcode, self.names(arg), *args, **kwargs)
        elif opcode in dis.haslocal:
            return self.i(opcode, self.varnames(arg), *args, **kwargs)
        else:
            raise ValueError(f"Unknown opcode: {dis.opnames[opcode]}")

    def nop(self, arg):
        arg = bytes(arg)
        for i in arg:
            self.i(NOP, int(i))

    def eval_jumps(self):
        lookup = {i.pos: i for i in self}
        for i in self:
            if i.is_jrel:
                target = lookup[i.arg + i.pos + 2]
            elif i.is_jump:
                target = lookup[i.arg]
            else:
                target = None
            if target is not None:
                target.jump_target.append(i)
                i.jump_to = target

    def assign_pos(self):
        pos = 0
        result = False
        for i in self:
            if i.pos != pos:
                result = True
            i.pos = pos
            pos += i.len
        return result

    def assign_jump_args(self):
        for i in self:
            if i.is_jump:
                i.arg = i.jump_to.pos
            elif i.is_jrel:
                i.arg = i.jump_to.pos - i.pos - 2

    def assign_len(self):
        for i in self:
            i.len = 2 * max(1, len(long2bytes(i.arg)))

    def get_bytecode(self):
        for i in range(4):
            self.assign_jump_args()
            self.assign_len()
            if self.assign_pos() is False:
                break
        else:
            raise ValueError("Failed to re-assemble")
        return b''.join(i.bytes for i in self)

    def __str__(self):
        lookup = {id(i): i_i for i_i, i in enumerate(self)}
        connections = []
        for i in self:
            connections.append([])
        for i_i, i in enumerate(self):
            if i.jump_to is not None:
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
        for i, c, c_prev, c_next in zip(self, connections, [[]] + connections[:-1], connections[1:] + [[]]):
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
                lines.append(str(i) + ''.join(_str))
        return '\n'.join(lines)


def _repr_opcode(opcode, arg, code):
    head = f"{dis.opname[opcode]:>20} {arg: 3d}"
    if opcode == LOAD_CONST:
        return f"{head} {'(' + repr(code.co_consts[arg]) + ')':<12}"
    elif opcode in (LOAD_FAST, STORE_FAST):
        return f"{head} {code.co_varnames[arg]:<12}"
    elif opcode in (LOAD_NAME, STORE_NAME):
        return f"{head} {code.co_names[arg]:<12}"
    elif opcode in (LOAD_GLOBAL, STORE_GLOBAL):
        return f"{head} {'(' + repr(code.co_names[arg]) + ')':<12}"
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



