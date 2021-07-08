import inspect
import struct
import dis
import ctypes
from collections import namedtuple
from functools import partial
from types import CodeType, FunctionType
import logging

from importlib._bootstrap_external import _code_to_timestamp_pyc

import subprocess
from tempfile import NamedTemporaryFile
import os

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


class CodePatcher(dict):
    def __init__(self, code):
        self._code = code

    def __str__(self):
        return f"CodePatcher(code={self._code})"

    def _diff(self):
        _new = list(self._code.co_code)
        for pos, patch in self.items():
            _new[pos:pos + len(patch)] = patch
        return _dis(self._code, alt=_new)

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

    def post(self):
        pass

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
        self.post()
        logging.debug(f"  ⏎ {self.nxt} (finally)")
        return self.nxt

    def __str__(self):
        if self.patcher is None:
            return f"<{self.__class__.__name__} (patcher not assigned)>"
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
            w_restore = RestoreBytecodeWorm(patcher=patcher, pos="return")
            if frame is frame_stack[-1] and finalize is not None:
                w_restore.post = partial(finalize, self)
            w_vstack = ValueStackWorm(destination=self._notify, patcher=patcher, nxt=w_restore)
            logging.info(f"  patching ...")
            pass_value = w_vstack()  # injects the head
            if prev_worm is None:
                rtn = pass_value
            else:
                prev_worm.nxt = pass_value
            prev_worm = w_restore
        return rtn

    def compose_morph(self, pack=False):
        def prev():
            logging.info("Finally: restoring fake stack calls")
            frame = inspect.currentframe().f_back
            for st in self:
                patcher = FramePatcher(frame)
                patcher.patch(st.code.co_code, 0)
                patcher.commit()
                frame = frame.f_back
        for i in self:
            prev = morph_execpoint(i, prev, pack=pack)
        return prev

    def __str__(self):
        return '\n'.join(("Snapshot", *map(str, self)))


def snapshot(frame, **kwargs):
    if frame is None:
        frame = 2
    return Snapshot().inject(frame, **kwargs)


def save(fname, fmt="pill", pack=False):
    assert fmt in ("pill", "pyc")
    if fmt == "pyc":
        pack = True
    if isinstance(fname, str):
        fname = open(fname, 'w+b')

    if fmt == "pill":
        def serializer(obj):
            dill.dump(obj.compose_morph(pack=pack), fname)

    elif fmt == "pyc":
        def serializer(obj):
            code = obj.compose_morph(pack=pack).__code__
            fname.write(_code_to_timestamp_pyc(code))

    return snapshot(
        inspect.currentframe().f_back,
        finalize=serializer,
    )


def dummy_teleport():
    pyc_file = NamedTemporaryFile(suffix="pyc")
    def dummy(obj):
        code = obj.compose_morph(pack=True).__code__
        pyc_file.write(_code_to_timestamp_pyc(code))
        p = subprocess.run(["python", pyc_file.name], env={"PYTHONPATH": ".:" + os.environ.get("PYTHONPATH", "")})
        exit(p.returncode)
    return snapshot(
        inspect.currentframe().f_back,
        finalize=dummy,
    )


def load(fname):
    with open(fname, 'rb') as f:
        dill.load(f)()


class CList(list):
    def index_store(self, x):
        try:
            return self.index(x)
        except ValueError:
            self.append(x)
            return len(self) - 1
    __call__ = index_store

# gi_frame


def morph_execpoint(p, nxt, pack=False):
    logging.info(f"Preparing a morph into execpoint {p} pack={pack} ...")
    f_code = p.code
    new_code = []
    new_consts = CList(f_code.co_consts)
    new_names = CList(f_code.co_names)
    new_varnames = CList(f_code.co_varnames)
    new_stacksize = f_code.co_stacksize

    if pack:
        import dill
        unpacker = new_varnames('.:loads:.')  # non-alphanumeric = unlikely to exist as a proper variable
        new_code.extend([
            LOAD_CONST, new_consts(0),
            LOAD_CONST, new_consts(('loads',)),
            IMPORT_NAME, new_names('dill'),
            IMPORT_FROM, new_names('loads'),
            STORE_FAST, unpacker,
        ])

        def _unpack(_what):
            new_code.extend([
                LOAD_FAST, unpacker,
                LOAD_CONST, new_consts(_what),
                CALL_FUNCTION, 1,
            ])

    logging.info("  locals ...")
    if len(p.v_locals) > 0:
        v_locals_k, v_locals_v = zip(*p.v_locals.items())
        if pack:
            _unpack(dill.dumps(v_locals_v))
        else:
            new_code.extend([
                LOAD_CONST, new_consts(v_locals_v),
            ])
        new_code.extend([
            UNPACK_SEQUENCE, len(v_locals_v),
        ])
        for k in v_locals_k:
            # k = v
            new_code.extend([
                STORE_FAST, new_varnames(k),
            ])
        new_stacksize = max(new_stacksize, len(v_locals_v))

    # stack
    if len(p.v_stack) > 0:
        v_stack = p.v_stack[::-1]
        if pack:
            _unpack(dill.dumps(v_stack))
        else:
            new_code.extend([
                LOAD_CONST, new_consts(v_stack),
            ])
        new_code.extend([
            UNPACK_SEQUENCE, len(v_stack),
        ])

    # restores one step back and ensures nxt can be called without arguments
    assert p.code.co_code[p.pos] in (CALL_FUNCTION, CALL_FUNCTION_KW)
    _code = bytearray(p.code.co_code)
    _code[p.pos] = CALL_FUNCTION  # call nxt ...
    _code[p.pos + 1] = 0  # ... with no arguments
    worm = RestoreBytecodeWorm(code=_code, pos=p.pos - 2, nxt=nxt)

    # worm()
    if pack:
        _unpack(dill.dumps(worm))
    else:
        new_code.extend([
            LOAD_CONST, new_consts(worm),
        ])
    new_code.extend([
        CALL_FUNCTION, 0,
        RETURN_VALUE, 0,
    ])

    new_code = expand_long(new_code)
    if len(new_code) < len(f_code.co_code):
        new_code += bytes([NOP, 0]) * ((len(f_code.co_code) - len(new_code)) // 2)

    code = CodeType(
        0,
        0,
        0,
        len(new_varnames),
        new_stacksize + 1,
        f_code.co_flags,
        new_code,
        tuple(new_consts),
        tuple(new_names),
        tuple(new_varnames),
        f_code.co_filename,
        f_code.co_name,
        f_code.co_firstlineno,
        f_code.co_lnotab,
    )
    logging.info("resulting morph:\n" + "\n".join(''.join(i) for i in _dis(code)))
    return FunctionType(code, globals())

