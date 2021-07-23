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
import base64
from shlex import quote
from pathlib import Path
import dill

from .mem_view import Mem
from .minias import _dis, Bytecode

locals().update(dis.opmap)


def _overlapping(s1, l1, s2, l2):
    e1 = s1 + l1
    e2 = s2 + l2
    return s1 < e2 and s2 < e1


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


class ExecPoint(namedtuple("ExecPoint", ("code", "pos", "v_stack", "v_locals", "v_globals", "v_builtins"))):
    slots = ()
    def __repr__(self):
        code = self.code
        return f'Code {code.co_name} at "{code.co_filename}"+{code.co_firstlineno} @{self.pos:d}\n  stack: {self.v_stack}\n  locals: {self.v_locals}'


def p_jump_to(patcher, pos, f_next):
    """
    Patch: jump to position.

    Parameters
    ----------
    patcher : FramePatcher
    pos : int
        Position to set.
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    logging.debug(f"jump_to {pos:d}: patching ...")
    if patcher.pos != pos - 2:
        patcher.patch_current(_jump_absolute(pos), 2)  # jump to the original bytecode position
    patcher.patch([CALL_FUNCTION, 0], pos)  # call next
    patcher.commit()
    logging.debug(f"jump_to {pos:d}: ⏎ {f_next}")
    return f_next


def p_maybe_jump_head(patcher, f_next):
    """
    Patch: jumps to the top of the frame.

    Parameters
    ----------
    patcher : FramePatcher
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    if patcher.pos == 0:
        return f_next()  # already at the top: execute next
    else:
        return p_jump_to(patcher, 0, f_next)


def p_set_bytecode(patcher, bytecode, post, f_next):
    """
    Patch: set the bytecode contents.

    Parameters
    ----------
    patcher : FramePatcher
    bytecode : bytearray
        Bytecode to overwrite.
    post : Callable
        Call this before returning.
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    logging.debug(f"set_bytecode: patching ...")
    patcher.patch(bytecode, 0)  # re-write the bytecode from scratch
    patcher.commit()
    if post is not None:
        post()
    logging.debug(f"set_bytecode: ⏎ {f_next}")
    return f_next


def p_place_beacon(patcher, beacon, f_next):
    """
    Patch: places the beacon.

    Parameters
    ----------
    patcher : FramePatcher
    beacon
        Beacon to place.
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    logging.debug(f"place_beacon {beacon}: patching ...")
    patcher.patch_current([
        UNPACK_SEQUENCE, 2,
        CALL_FUNCTION, 0,  # calls _payload1
        CALL_FUNCTION, 0,  # calls whatever follows
    ], 2)
    patcher.commit()
    logging.debug(f"place_beacon {beacon}: ⏎ ({f_next}, {beacon})")
    return f_next, beacon


class Worm:
    def __init__(self, patcher=None, nxt=None):
        self.patcher = patcher
        self.nxt = nxt

    def post(self):
        pass

    def _jump_top(self, f_next):
        return p_maybe_jump_head(self.patcher, f_next)

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
        return p_jump_to(self.patcher, self._snapshot_pos, self._payload1)

    def _payload1(self):
        return p_set_bytecode(self.patcher, self._snapshot_bytecode, self.post, self.nxt)


class ValueStackWorm(Worm):
    def __init__(self, destination, patcher=None, nxt=None):
        super().__init__(patcher, nxt)
        self.destination = destination
        self.beacon = Beacon()

    def _payload(self):
        return p_place_beacon(self.patcher, self.beacon, self._payload1)

    def _payload1(self):
        logging.debug(f"{self}._payload1 (collect the stack)")
        self.destination(self, collect_objects(get_value_stack(self.patcher._frame, id(self.beacon), expand=1)))  # this might corrupt memory
        return super()._payload()


def snapshot(frame, finalize):
    """
    Snapshot the stack starting from the given frame.
    This is a destructive operation: all stack frames
    will be patched and will return.

    Parameters
    ----------
    frame : FrameObject
        Top of the stack frame.
    finalize : Callable
        Where to return the result.

    Returns
    -------
    rtn : object
        An object that has to be returned to the TOS frame
        to initiate frame collection.
    """
    # determine the frame to start with
    logging.debug("Start frame serialization")
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

    result = []

    notify_current = 0
    def notify(_, stack):
        """A callback to save stack items"""
        nonlocal notify_current
        logging.info(f"Received object stack #{notify_current:d}: {len(stack):d} items")

        result[notify_current] = result[notify_current]._replace(v_stack=stack)
        notify_current += 1

    prev_globals = None
    prev_builtins = None
    w_restore = None

    while frame is not None:  # iterate over frame stack
        logging.info(f"Frame: {frame}")

        # check globals and builtins
        if prev_globals is None:
            prev_globals = frame.f_globals
        else:
            assert prev_globals is frame.f_globals
        if prev_builtins is None:
            prev_builtins = frame.f_builtins
        else:
            assert prev_builtins is frame.f_builtins

        # save locals, globals, etc.
        logging.info("  saving locals ...")
        result.append(ExecPoint(
            code=frame.f_code,
            pos=frame.f_lasti,
            v_stack=None,
            v_locals=frame.f_locals.copy(),
            v_globals=prev_globals,
            v_builtins=prev_builtins,
        ))

        # prepare patchers
        logging.info(f"  patching the bytecode ...")
        original_code = bytearray(frame.f_code.co_code)  # store the original bytecode
        # note that bytearray is intentional to guarantee the copy
        patcher = FramePatcher(frame)
        w_vstack = ValueStackWorm(destination=notify, patcher=patcher)  # determines the value stack
        ignition = w_vstack()  # patch now and save the object initiating collection
        if w_restore is not None:
            w_restore.nxt = ignition  # make the top frame return ignition
        else:
            rtn_value = ignition  # otherwise save it to return from this method
        w_restore = RestoreBytecodeWorm(patcher=patcher, code=original_code, pos="return")  # restores the bytecode
        w_vstack.nxt = w_restore  # chain these two internally

        if frame.f_back is None:
            w_restore.post = partial(finalize, result)  # process the result

        frame = frame.f_back  # next frame

    logging.info("Ready to collect frames")
    return rtn_value


def morph_execpoint(p, nxt, pack=None, unpack=None, _globals=False):
    """
    Prepares a code object which morphs into the desired state
    and continues the execution afterwards.

    Parameters
    ----------
    p : execpoint
        The execution point to morph into.
    nxt : CodeType
        The code object which develops the stack further.
    pack : Callable, None
        A method turning objects into bytes (serializer)
        locally.
    unpack : tuple, None
        A 2-tuple `(module_name, method_name)` specifying
        the method that morph uses to unpack the data.
    _globals : bool
        If True, unpacks globals.

    Returns
    -------
    result : CodeType
        The resulting morph.
    """
    assert pack is None and unpack is None or pack is not None and unpack is not None,\
        "Either both or none pack and unpack arguments should be specified"
    logging.info(f"Preparing a morph into execpoint {p} pack={pack is not None} ...")
    code = Bytecode.disassemble(p.code)
    code.pos = 0
    code.nop(b'mrph')  # signature
    f_code = p.code
    new_stacksize = f_code.co_stacksize

    if pack:
        unpack_mod, unpack_method = unpack
        # from {module_name} import {load_name}
        unpack = code.varnames('.:unpack:.')  # non-alphanumeric = unlikely to exist as a proper variable
        code.I(LOAD_CONST, 0)
        code.I(LOAD_CONST, (unpack_method,))
        code.I(IMPORT_NAME, unpack_mod)
        code.I(IMPORT_FROM, unpack_method)
        code.i(STORE_FAST, unpack)

        def _LOAD(_what):
            code.i(LOAD_FAST, unpack)
            code.I(LOAD_CONST, pack(_what))
            code.i(CALL_FUNCTION, 1)
    else:
        def _LOAD(_what):
            code.I(LOAD_CONST, _what)

    scopes = [(p.v_locals, STORE_FAST, "locals")]
    if _globals:
        scopes.append((p.v_globals, STORE_GLOBAL, "globals"))
    for _dict, _STORE, log_name in scopes:
        logging.info(f"  {log_name} ...")
        if len(_dict) > 0:
            klist, vlist = zip(*_dict.items())
            _LOAD(vlist)
            code.i(UNPACK_SEQUENCE, len(vlist))
            for k in klist:
                # k = v
                code.I(_STORE, k)
            new_stacksize = max(new_stacksize, len(vlist))

    # stack
    if len(p.v_stack) > 0:
        v_stack = p.v_stack[::-1]
        _LOAD(v_stack)
        code.i(UNPACK_SEQUENCE, len(v_stack))

    if nxt is not None:
        # call nxt which is a code object

        # load code object
        _LOAD(nxt)
        code.I(LOAD_CONST, None)  # function name
        code.i(MAKE_FUNCTION, 0)  # turn code object into a function
        code.i(CALL_FUNCTION, 0)  # call it
    else:
        code.I(LOAD_CONST, None)  # fake nxt returning None

    # now jump to the previously saved position
    target_pos = p.pos + 2  # p.pos points to the last executed opcode
    # find the instruction ...
    for jump_target in code:
        if jump_target.pos == target_pos:
            break
    else:
        raise RuntimeError
    # ... and jump to it (the argument will be determined after re-assemblling the bytecode)
    code.i(JUMP_ABSOLUTE, 0, jump_to=jump_target)

    code = CodeType(
        0,
        0,
        0,
        len(code.varnames),
        new_stacksize + 1,
        0,
        code.get_bytecode(),
        tuple(code.consts),
        tuple(code.names),
        tuple(code.varnames),
        f_code.co_filename,  # TODO: something smarter should be here
        f_code.co_name,
        f_code.co_firstlineno,
        f_code.co_lnotab,
        )
    logging.info("resulting morph:\n" + "\n".join(''.join(i) for i in _dis(code)))
    return code


def morph_stack(frame_data, **kwargs):
    """
    Morphs the stack.

    Parameters
    ----------
    frame_data : list
        States of all individual frames.
    kwargs
        Arguments to morph_execpoint.

    Returns
    -------
    result : CodeType
        The resulting morph for the root frame.
    """
    prev = None
    for i, frame in enumerate(frame_data):
        logging.info(f"Preparing morph #{i:d}")
        prev = morph_execpoint(frame, prev, _globals=frame is frame_data[-1], **kwargs)
    return prev


def dump(file, **kwargs):
    """
    Serialize the runtime into a file and exit.

    Parameters
    ----------
    file : File
        The file to write to.
    kwargs
        Arguments to `dill.dump`.
    """
    def serializer(stack_data):
        dill.dump(FunctionType(morph_stack(stack_data), globals()), file, **kwargs)
    return snapshot(
        inspect.currentframe().f_back,
        finalize=serializer,
    )
load = dill.load


def bash_inline_create_file(name, contents):
    """
    Turns a file into bash command.

    Parameters
    ----------
    name : str
        File name.
    contents : bytes
        File contents.

    Returns
    -------
    result : str
        The resulting command that creates this file.
    """
    return f"echo {quote(base64.b64encode(contents).decode())} | base64 -d > {quote(name)}"


def shell_teleport(*shell_args, python="python", before="cd $(mktemp -d)",
        pyc_fn="payload.pyc", shell_delimeter="; ", pack_file=bash_inline_create_file,
        pack_object=dill.dumps, unpack_object=("dill", "loads"),
        _frame=None, **kwargs):
    """
    Teleport into another shell.

    Parameters
    ----------
    shell_args
        Arguments to a shell where python is found.
    python : str
        Python executable in the shell.
    before : str, list
        Shell commands to be run before anything else.
    pyc_fn : str
        Temporary filename to save the bytecode to.
    shell_delimeter : str
        Shell delimeter to chain multiple commands.
    pack_file : Callable
        A function `f(name, contents)` turning a file
        into shell-friendly assembly.
    pack_object : Callable, None
        A method turning objects into bytes (serializer)
        locally.
    unpack_object : tuple, None
        A 2-tuple `(module_name, method_name)` specifying
        the method that morph uses to unpack the data.
    _frame
        The frame to collect.
    kwargs
        Other arguments to `subprocess.run`.

    Returns
    -------
    None
    """
    payload = []
    if not isinstance(before, (list, tuple)):
        payload.append(before)
    else:
        payload.extend(before)

    def _teleport(stack_data):
        """Will be executed after the snapshot is done."""
        logging.info("Snapshot done, composing morph ...")
        code = morph_stack(stack_data, pack=pack_object, unpack=unpack_object)  # compose the code object
        logging.info("Creating pyc ...")
        files = {pyc_fn: _code_to_timestamp_pyc(code)}  # turn it into pyc
        for k, v in files.items():
            payload.append(pack_file(k, v))  # turn files into shell commands
        payload.append(f"{python} {pyc_fn}")  # execute python

        # pipe the output and exit
        logging.info("Executing the payload ...")
        p = subprocess.run([*shell_args, shell_delimeter.join(payload)], text=True, **kwargs)
        exit(p.returncode)

    # proceed to snapshotting
    return snapshot(
        inspect.currentframe().f_back if _frame is None else _frame,
        finalize=_teleport,
    )
bash_teleport = shell_teleport


def dummy_teleport(**kwargs):
    """A dummy teleport into another python process in current environment."""
    return bash_teleport("bash", "-c", _frame=inspect.currentframe().f_back, **kwargs)
