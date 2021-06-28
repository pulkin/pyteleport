import inspect
import struct
import dis
import ctypes
from collections import namedtuple
from types import CodeType, FunctionType
import logging

import dill

from mem_view import Mem

locals().update(dis.opmap)


def _overlapping(s1, l1, s2, l2):
    e1 = s1 + l1
    e2 = s2 + l2
    return s1 < e2 and s2 < e1


def _repr_opcode(opcode, arg, code):
    head = f"{dis.opname[opcode]:>20} {arg: 3d}"
    if opcode == LOAD_CONST:
        return f"{head} {'(' + repr(code.co_consts[arg]) + ')':<12}"
    elif opcode in (LOAD_FAST, STORE_FAST):
        return f"{head} {code.co_varnames[arg]:<12}"
    elif opcode in (LOAD_GLOBAL, STORE_GLOBAL):
        return f"{head} {'(' + repr(code.co_names[arg]) + ')':<12}"
    else:
        return f"{head}" + " " * 13


class CodePatcher(dict):
    def __init__(self, code):
        self._code = code

    def __str__(self):
        return f"CodePatcher(code={self._code})"

    def _diff(self):
        code = self._code.co_code
        result = list(zip(code[::2], code[1::2]))
        for pos, patch in self.items():
            assert pos % 2 == 0
            pos //= 2
            for offset, (opc, arg) in enumerate(zip(patch[::2], patch[1::2])):
                result[pos + offset] = (*result[pos + offset], opc, arg)
        result_repr = []
        for i, line in enumerate(result):
            i *= 2
            if len(line) == 2:
                opc_new, arg_new = line
                result_repr.append((f"{i: 3d} {_repr_opcode(opc_new, arg_new, self._code)}",))
            else:
                opc_old, arg_old, opc_new, arg_new = line
                result_repr.append(("\033[94m", f"{i: 3d} {_repr_opcode(opc_new, arg_new, self._code)} {_repr_opcode(opc_old, arg_old, self._code)}", "\033[0m"))
        return result_repr

    def commit(self):
        logging.debug(f"Commit patch to <{self._code.co_name}>")
        for i in self._diff():
            logging.debug(''.join(i))
        code = self._code.co_code
        code_view = Mem.view(code)
        for pos, patch in self.items():
            assert len(patch) <= len(code), f"len(patch) = {len(patch)} > len(code) = {len(code)}"
            assert 0 <= pos <= len(code) - len(patch), f"Index {pos:d} out of range [0, {len(code) - len(patch)}]"
            code_view[pos:pos + len(patch)] = patch
        self.clear()

    @property
    def last_opcode(self):
        return self._code.co_code[-2]

    def __setitem__(self, pos, patch):
        patch = bytes(patch)
        code = self._code.co_code
        assert len(patch) <= len(code), f"len(patch) = {len(patch)} > len(code) = {len(code)}"
        assert 0 <= pos <= len(code) - len(patch), f"Index {pos:d} out of range [0, {len(code) - len(patch)}]"
        for _pos, _other in self.items():
            if _overlapping(pos, len(patch), _pos, len(_other)):
                raise ValueError("Patches overlap")
        super().__setitem__(pos, patch)

    def patch(self, patch, pos):
        self[pos] = patch

    def __len__(self):
        return len(self._code.co_code)


class FramePatcher(CodePatcher):
    def __init__(self, frame):
        self._frame = frame
        super().__init__(frame.f_code)

    def __str__(self):
        return f"FramePatcher(frame={self._frame})"

    @property
    def pos(self):
        return self._frame.f_lasti

    def _diff(self):
        result_ = super()._diff()
        result = []
        for i, l in enumerate(result_):
            if 2 * i == self.pos:
                if l[0].startswith('\033'):
                    result.append(('\033[92m', *l[1:]))
                else:
                    result.append(('\033[92m', *l, '\033[0m'))
            else:
                result.append(l)
        return result

    @property
    def current_opcode(self):
        return self._code.co_code[self.pos]

    def patch_current(self, patch, pos):
        return self.patch(patch, pos + self.pos)


def expand_long(c):
    result = []
    for opcode, val in zip(c[::2], c[1::2]):
        if not val:
            result.extend([opcode, val])
        else:
            bts = val.to_bytes((val.bit_length() + 7) // 8, byteorder="big")
            assert len(bts) < 5
            for b in bts[:-1]:
                result.extend([EXTENDED_ARG, b])
            result.extend([opcode, bts[-1]])
    return bytes(result)


def _jump_absolute(i):
    return expand_long([JUMP_ABSOLUTE, i])


def _pvaluestack(frame):
    pframe = id(frame)
    # https://github.com/python/cpython/blob/46b16d0bdbb1722daed10389e27226a2370f1635/Include/cpython/frameobject.h#L17
    pvaluestack_star = pframe + 0x40
    pframe_mem = Mem(pvaluestack_star, 0x08)
    pvaluestack, = struct.unpack("P", pframe_mem[:])
    return pvaluestack


def get_value_stack(frame, stack_top, expand=0):
    stack_bot = _pvaluestack(frame)
    stack_view = Mem(stack_bot, (frame.f_code.co_stacksize + expand) * 8)[:]
    result = []
    for i in range(0, len(stack_view), 8):
        obj_ref = int.from_bytes(stack_view[i:i + 8], "little")
        if obj_ref == stack_top:
            return result
        result.append(obj_ref)
    raise RuntimeError("Failed to determine stack top")


def collect_objects(oids):
    return [ctypes.cast(i, ctypes.py_object).value for i in oids]


class Beacon:
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<beacon>"


class ExecPoint(namedtuple("ExecPoint", ("code", "pos", "v_stack", "v_locals"))):
    slots = ()
    def __repr__(self):
        code = self.code
        return f'Code {code.co_name} at "{code.co_filename}"+{code.co_firstlineno} @{self.pos:d}\n  stack: {self.v_stack}\n  locals: {self.v_locals}'


class Worm:
    def __init__(self, patcher=None, nxt=None):
        self.patcher = patcher
        self.nxt = nxt

    def _jump_top(self, f_next):
        """Jumps to the top of the frame and calls the next function"""
        if self.patcher.pos == 0:
            logging.debug(f"{self}._jump_top (already at the top)")
            # already at the top: execute the payload
            logging.debug(f"  ⏎ {f_next} (directly)")
            return f_next()

        else:
            logging.debug(f"{self}._jump_top (will jump to the top of the frame)")
            # jump to the beginning of the bytecode ...
            self.patcher.patch_current(_jump_absolute(0), 2)
            # ... and call the payload
            self.patcher.patch([
                CALL_FUNCTION, 0,
            ], 0)
            self.patcher.commit()
            logging.debug(f"  ⏎ {f_next}")
            return f_next

    def __call__(self):
        if self.patcher is None:
            self.patcher = FramePatcher(inspect.currentframe().f_back)
        return self._jump_top(self._payload)

    def _payload(self):
        logging.debug(f"  ⏎ {self.nxt} (finally)")
        return self.nxt

    def __str__(self):
        if self.patcher is None:
            return f"<{self.__class__.__name__} (unassigned)>"
        else:
            return f"<{self.__class__.__name__} -> '{self.patcher._code.co_name}'>"


class RestoreBytecodeWorm(Worm):
    def __init__(self, patcher=None, nxt=None, code=None, pos="continue"):
        super().__init__(patcher, nxt)
        if code is None:
            if patcher is None:
                raise ValueError("Either patcher or code has to be specified")
            code = bytearray(self.patcher._frame.f_code.co_code)
        self._snapshot_bytecode = code
        if pos == "return":
            for i, instr in enumerate(self._snapshot_bytecode[::2]):
                if instr == RETURN_VALUE:
                    self._snapshot_pos = 2 * (i - 1)
                    break
            else:
                raise ValueError("RETURN_VALUE not found")
        elif pos == "continue":
            self._snapshot_pos = self.patcher.pos
        else:
            self._snapshot_pos = pos

    def _payload(self):
        logging.debug(f"{self}._payload (jump to {self._snapshot_pos:d})")
        # jump to the original bytecode position ...
        if self.patcher.pos != self._snapshot_pos - 2:
            self.patcher.patch_current(_jump_absolute(self._snapshot_pos), 2)
        # ... and call restore_bytecode
        self.patcher.patch([
            CALL_FUNCTION, 0
        ], self._snapshot_pos)
        self.patcher.commit()
        logging.debug(f"  ⏎ {self._payload1}")
        return self._payload1

    def _payload1(self):
        logging.debug(f"{self}._payload1 (restore the original bytecode state)")
        # re-write the bytecode from scratch and return the previously saved value
        self.patcher.patch(self._snapshot_bytecode, 0)
        self.patcher.commit()
        return super()._payload()


class ExitWorm(Worm):
    def __init__(self, patcher=None, nxt=None, nxt_args=None):
        super().__init__(patcher, nxt)
        self.nxt_args = tuple(nxt_args) if nxt_args is not None else ()

    def _payload(self):
        if self.nxt is not None:
            logging.debug(f"{self}._payload (call {self.nxt}({', '.join(repr(i) for i in self.nxt_args)}))")
            self.nxt(*self.nxt_args)
        logging.debug(f"{self}._payload (return)")
        self.patcher.patch_current([RETURN_VALUE, 0], 2)
        self.patcher.commit()


class ValueStackWorm(Worm):
    def __init__(self, destination, patcher=None, nxt=None):
        super().__init__(patcher, nxt)
        self.destination = destination
        self.beacon = Beacon()

    def _payload(self):
        logging.debug(f"{self}._payload (place a beacon)")
        self.patcher.patch_current([
            UNPACK_SEQUENCE, 2,
            CALL_FUNCTION, 0,  # calls _payload1
            CALL_FUNCTION, 0,  # calls whatever follows
        ], 2)
        self.patcher.commit()
        logging.debug(f"  ⏎ beacon, {self._payload1}")
        return self._payload1, self.beacon

    def _payload1(self):
        logging.debug(f"{self}._payload1 (collect the stack)")
        frame = self.patcher._frame
        name = frame.f_code.co_name
        self.destination(self, collect_objects(get_value_stack(frame, id(self.beacon), expand=1)))  # this might corrupt memory
        return super()._payload()


class Snapshot(list):
    def _notify(self, worm, stack):
        logging.info(f"{worm} reporting {stack}")
        ep = self.sdata.pop(0)._asdict()
        ep["v_stack"] = stack
        self.append(ExecPoint(**ep))

    def inject(self, frame, finalize=None):
        logging.debug("Start frame serialization")
        self.clear()
        if frame is None:
            logging.info("  no frame specified")
            frame = 1
        if isinstance(frame, int):
            logging.info(f"  taking frame #{frame:d}")
            _frame = inspect.currentframe()
            for i in range(frame):
                _frame = _frame.f_back
            frame = _frame

        logging.info(f"  frame: {frame}")

        frame_stack = []
        sdata = []
        while frame is not None:
            sdata.append(ExecPoint(
                code=frame.f_code,
                pos=frame.f_lasti,
                v_stack=None,
                v_locals=frame.f_locals.copy(),
            ))
            frame_stack.append(frame)
            frame = frame.f_back
        logging.info(f"  {len(frame_stack):d} frames detected")
        self.sdata = sdata

        logging.info(f"  {len(frame_stack):d} frames collected")

        rtn = None
        prev_worm = None
        logging.info("Deploying patches ...")
        for frame in frame_stack:
            logging.info(f"Deploying into {frame} ...")
            patcher = FramePatcher(frame)
            if frame is not frame_stack[-1] or finalize is None:
                w_restore = RestoreBytecodeWorm(patcher=patcher, pos="return")
            else:
                w_restore = ExitWorm(patcher=patcher, nxt=finalize, nxt_args=[self])
            w_vstack = ValueStackWorm(destination=self._notify, patcher=patcher, nxt=w_restore)
            logging.info(f"  patching ...")
            pass_value = w_vstack()  # injects the head
            if prev_worm is None:
                rtn = pass_value
            else:
                prev_worm.nxt = pass_value
            prev_worm = w_restore
        return rtn

    def compose_morph(self):
        prev = lambda: None
        for i in self:
            prev = morph_execpoint(i, nxt=prev)
        return prev

    def __str__(self):
        return '\n'.join(("Snapshot", *map(str, self)))

def snapshot(frame, **kwargs):
    if frame is None:
        frame = 2
    return Snapshot().inject(frame, **kwargs)

# gi_frame


def morph_execpoint(p, nxt=None):
    logging.info(f"Preparing a morph into execpoint {p} ...")
    f_code = p.code
    new_code = []
    new_consts = list(f_code.co_consts)

    # locals
    for k, v in p.v_locals.items():
        dst = f_code.co_varnames.index(k)
        # k = v
        new_code.extend([
            LOAD_CONST, len(new_consts),
            STORE_FAST, dst,
        ])
        new_consts.append(v)

    # stack
    for v in p.v_stack:
        new_code.extend([
            LOAD_CONST, len(new_consts),
        ])
        new_consts.append(v)

    # restores one step back and ensures nxt can be called without arguments
    assert p.code.co_code[p.pos] in (CALL_FUNCTION, CALL_FUNCTION_KW)
    _code = bytearray(p.code.co_code)
    _code[p.pos] = CALL_FUNCTION  # call nxt ...
    _code[p.pos + 1] = 0  # ... with no arguments
    worm = RestoreBytecodeWorm(code=_code, pos=p.pos - 2, nxt=nxt)

    # worm()
    new_code.extend([
        LOAD_CONST, len(new_consts),
        CALL_FUNCTION, 0,
    ])
    new_consts.append(worm)

    new_code = expand_long(new_code)
    if len(new_code) < len(f_code.co_code):
        new_code += bytes([NOP, 0]) * ((len(f_code.co_code) - len(new_code)) // 2)

    code = CodeType(
        f_code.co_argcount,  # argcount=0,
        f_code.co_posonlyargcount,  # posonlyargcount?
        f_code.co_kwonlyargcount,  # kwonlyargcount=0,
        f_code.co_nlocals,  # nlocals=1,
        f_code.co_stacksize + 1,  # stacksize=1,
        f_code.co_flags,  # flags=0,
        new_code,  # bytes([LOAD_CONST, 0, RETURN_VALUE, 0]),  # codestring=bytes([LOAD_CONST, 0, RETURN_VALUE, 0]),
        tuple(new_consts),  # consts=(None,),
        f_code.co_names,  # names=(),
        f_code.co_varnames,  # varnames=(),
        f_code.co_filename,  # filename='',
        f_code.co_name,  # name='',
        f_code.co_firstlineno,  # firstlineno=0,
        f_code.co_lnotab,  # lnotab=b'',
    )
    return FunctionType(code, globals())

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    entered_c = 0
    exited_c = 0

    def a():
        def b():
            def c():
                global entered_c, exited_c
                entered_c += 1
                result = "hello"
                snapshot(None, finalize=lambda x: dill.dump(x, open("state.pickle", 'wb')))
                exited_c += 1
                return result + " world"
            return len(c()) + float("3.5")
        return 5 * (3 + b())
    state = a()
    with open("state.pickle", 'wb') as f:
        dill.dump(state, f)
    assert (entered_c, exited_c) == (1, 0)
    morph = state.compose_morph()
    assert morph() == 87.5
    assert (entered_c, exited_c) == (1, 1)

