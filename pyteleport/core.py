import inspect
import dis
import ctypes
from collections import namedtuple
import functools
from itertools import groupby
from types import CodeType, FunctionType
import logging
import struct
from importlib._bootstrap_external import _code_to_timestamp_pyc

import subprocess
import base64
from shlex import quote
import dill
import sys

from .mem_view import Mem
from .py import (
    JX,
    ptr_frame_stack_bottom,
    ptr_frame_stack_top,
    ptr_frame_block_stack_bottom,
    ptr_frame_block_stack_top,
)
from .minias import _dis, long2bytes
from .morph import morph_stack
from .primitives import NULL


locals().update(dis.opmap)


class partial(functools.partial):
    def __str__(self):
        return f"partial({self.func}, positional={len(self.args)}, kw=[{', '.join(self.keywords)}])"
    __repr__ = __str__


def _overlapping(s1, l1, s2, l2):
    e1 = s1 + l1
    e2 = s2 + l2
    return s1 < e2 and s2 < e1


class CodePatcher(dict):
    """Collects and applies patches to bytecodes."""
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
            logging.log(5, ''.join(i))
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
    """Collects and applies patches to bytecodes."""
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
    """Expands opcode arguments if they do not fit byte"""
    result = []
    for opcode, val in zip(c[::2], c[1::2]):
        if not val:
            result.extend([opcode, val])
        else:
            bts = long2bytes(val)
            for b in bts[:-1]:
                result.extend([EXTENDED_ARG, b])
            result.extend([opcode, bts[-1]])
    return bytes(result)


def get_value_stack_from_beacon(frame, beacon, expand=0, null=NULL()):
    """
    Collects frame stack using beacon as
    an indicator of stack top.

    Parameters
    ----------
    frame : FrameObject
        Frame to process.
    beacon : int
        Value on top of the stack.
    expand : int
    null
        The NULL object replacement.

    Returns
    -------
    stack : list
        Stack contents.
    """
    logging.debug(f"collecting stack for {frame} with beacon 0x{beacon:016x}")
    stack_bot = ptr_frame_stack_bottom(frame)
    stack_view = Mem(stack_bot, (frame.f_code.co_stacksize + expand) * 8)
    logging.debug(f"  contents:\n{stack_view}")
    stack_view = stack_view[:]
    result = []
    for i in range(0, len(stack_view), 8):
        obj_ref = int.from_bytes(stack_view[i:i + 8], "little")
        logging.debug(f"  object at 0x{obj_ref:016x} ...")
        if obj_ref == beacon:
            logging.debug(f"    <beacon>")
            return result
        if obj_ref == 0:
            result.append(null)
            logging.debug(f"    (null)")
        else:
            result.append(ctypes.cast(obj_ref, ctypes.py_object).value)
            logging.debug(f"    {repr(result[-1])}")
    raise RuntimeError("Failed to determine stack top")


def get_value_stack(frame):
    """
    Collects frame stack for generator objects.

    Parameters
    ----------
    frame : FrameObject
        Frame to process.

    Returns
    -------
    stack : list
        Stack contents.
    """
    stack_bot = ptr_frame_stack_bottom(frame)
    stack_top = ptr_frame_stack_top(frame)
    data = Mem(stack_bot, stack_top - stack_bot)[:]
    result = []
    for i in range(0, len(data), 8):
        obj_ref = int.from_bytes(data[i:i + 8], "little")
        result.append(ctypes.cast(obj_ref, ctypes.py_object).value)
    return result


block_stack_item = namedtuple('block_stack_item', ('type', 'handler', 'level'))


def get_block_stack(frame):
    """
    Collects block stack.

    Parameters
    ----------
    frame : FrameObject
        Frame to process.

    Returns
    -------
    stack : list
        Block stack contents.
    """
    fr = ptr_frame_block_stack_bottom(frame)
    to = ptr_frame_block_stack_top(frame)
    size = to - fr
    size4 = size // 4
    result = struct.unpack("i" * size4, Mem(fr, size)[:])
    result = tuple(block_stack_item(*x) for x in zip(result[::3], result[1::3], result[2::3]))
    return result


class FrameSnapshot(namedtuple("FrameSnapshot", ("scope", "code", "pos", "v_stack", "v_locals", "v_globals", "v_builtins", "block_stack"))):
    """A snapshot of python frame"""
    slots = ()
    def __repr__(self):
        code = self.code
        contents = []
        for i in "v_stack", "v_locals", "v_globals", "v_builtins", "block_stack":
            v = getattr(self, i)
            if v is None:
                contents.append(f"{i}: not set")
            else:
                contents.append(f"{i}: {len(v):d}")
        return f'FrameSnapshot {self.scope} -> {code.co_name} "{code.co_filename}"+{code.co_firstlineno} @{self.pos:d} {" ".join(contents)}'


def p_jump_to(pos, patcher, f_next, jx=JX):
    """
    Patch: jump to position.

    Parameters
    ----------
    pos : int
        Position to set.
    patcher : FramePatcher
    f_next : Callable
    jx : int
        Jump address multiplier.

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    if patcher.pos == pos - 2:
        if f_next is not None:
            return f_next()  # already at the top: execute next
    else:
        logging.debug(f"jump_to {pos:d}: patching ...")
        if patcher.pos != pos - 2:
            patcher.patch_current(expand_long([JUMP_ABSOLUTE, pos // jx]), 2)
        patcher.patch([CALL_FUNCTION, 0], pos)  # call next
        patcher.commit()
        logging.debug(f"jump_to {pos:d}: ⏎ {f_next}")
        return f_next


def p_set_bytecode(bytecode, post, patcher, f_next):
    """
    Patch: set the bytecode contents.

    Parameters
    ----------
    bytecode : bytearray
        Bytecode to overwrite.
    post : Callable
        Call this before returning.
    patcher : FramePatcher
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


def p_place_beacon(beacon, patcher, f_next):
    """
    Patch: places the beacon.

    Parameters
    ----------
    beacon
        Beacon to place.
    patcher : FramePatcher
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    beacon
        The beacon object.
    """
    logging.debug(f"place_beacon {beacon}: patching ...")
    patcher.patch_current([
        UNPACK_SEQUENCE, 2,
        CALL_FUNCTION, 0,  # calls f_next
        CALL_FUNCTION, 0,  # calls what f_next returns
    ], 2)
    patcher.commit()
    logging.debug(f"place_beacon {beacon}: ⏎ ({f_next}, {beacon})")
    return f_next, beacon


def p_exit_block_stack(block_stack, patcher, f_next):
    """
    Patch: exits the block stack.

    Parameters
    ----------
    block_stack
        State of the block stack.
    patcher : FramePatcher
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    logging.debug(f"exit block stack {len(block_stack):d} times")
    patcher.patch_current(
        [POP_BLOCK, 0] * len(block_stack) + [CALL_FUNCTION, 0], 2
    )
    patcher.commit()
    logging.debug(f"exit block stack: ⏎ {f_next}")
    return f_next


def snapshot(frame, finalize, method="inject"):
    """
    Snapshot the stack starting from the given frame.

    Parameters
    ----------
    frame : FrameObject
        Top of the stack frame.
    finalize : Callable
        Where to return the result.
    method : {"inject", "direct"}
        Method to use for the stack:
        * `inject`: makes a snapshot of an active stack by
          patching stack frames and running bytecode snippets
          inside. The stack is destroyed and the result is
          returned into `finalize` function (required).
        * `direct`: makes a snapshot of an inactive stack
          by reading FrameObject structure fields. Can only
          be used with generator frames.

    Returns
    -------
    rtn : object
        Depending on the method, this is either the snapshot
        itself or an object that has to be returned to the
        subject frame to initiate invasive frame collection.
    """
    assert method in {"inject", "direct"}
    if method == "inject" and finalize is None:
        raise ValueError("For method='inject' finalize has to set")
    # determine the frame to start with
    logging.debug(f"Start frame serialization; mode: {'active' if finalize is not None else 'inactive'}")
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
    if method == "inject":  # prepare to recieve data from patched frames
        beacon = object()  # beacon object

        notify_current = 0
        def notify(frame, f_next):
            """A callback to save stack items"""
            nonlocal notify_current, beacon
            logging.debug(f"Identify/collect object stack ...")
            result[notify_current] = result[notify_current]._replace(
                v_stack=get_value_stack_from_beacon(frame, id(beacon), expand=1))  # this might corrupt memory
            logging.info(f"  received {len(result[notify_current].v_stack):d} items")
            notify_current += 1
            return f_next

        chain = []  # holds a chain of patches and callbacks

    prev_globals = None
    prev_builtins = None

    while frame is not None:  # iterate over frame stack
        logging.info(f"Frame: {frame}")

        # check builtins
        if prev_builtins is None:
            prev_builtins = frame.f_builtins
        else:
            assert prev_builtins is frame.f_builtins

        # save locals, globals, etc.
        logging.info("  saving snapshot ...")
        fs = FrameSnapshot(
            scope=inspect.getmodule(frame),
            code=frame.f_code,
            pos=frame.f_lasti,
            v_stack=None if method == "inject" else get_value_stack(frame),
            v_locals=frame.f_locals.copy(),
            v_globals=frame.f_globals,
            v_builtins=frame.f_builtins,
            block_stack=get_block_stack(frame),
        )
        logging.info(f"    scope: {fs.scope}")
        logging.info(f"    code: {fs.code}")
        logging.info(f"    pos: {fs.pos}")
        logging.info(f"    stack: {len(fs.v_stack) if fs.v_stack is not None else 'none'}")
        logging.info(f"    locals: {len(fs.v_locals)}")
        logging.info(f"    globals: {len(fs.v_globals)}")
        logging.info(f"    builtins: {len(fs.v_builtins)}")
        logging.info(f"    block_stack:")
        if len(fs.block_stack):
            for i in fs.block_stack:
                logging.info(f"      {i}")
        else:
            logging.info("      (empty)")
        result.append(fs)

        if method == "inject":  # prepare patchers
            logging.info(f"  patching the bytecode ...")
            original_code = bytearray(frame.f_code.co_code)  # store the original bytecode
            rtn_pos = original_code[::2].index(RETURN_VALUE) * 2  # figure out where it returns
            # note that the bytearray is intentional to guarantee the copy
            patcher = FramePatcher(frame)

            p_jump_to(0, patcher, None)  # make room for patches immediately
            chain.append(partial(p_place_beacon, beacon, patcher))  # place the beacon
            chain.append(partial(notify, frame))  # collect value stack
            chain.append(partial(p_jump_to, 0, patcher))  # jump once more to make more room
            chain.append(partial(p_exit_block_stack, fs.block_stack, patcher))  # exit from "finally" statements
            chain.append(partial(p_jump_to, rtn_pos - 2, patcher))  # jump 1 opcode before return
            chain.append(partial(
                p_set_bytecode,
                original_code,
                None
                if frame.f_back is not None
                else partial(finalize, result),
                patcher
            ))  # restore the bytecode

        frame = frame.f_back  # next frame

    if method == "inject":  # chain patches
        prev = None
        for i in chain[::-1]:
            prev = partial(i, prev)
        logging.info("Ready to collect frames")
        return prev

    else:
        logging.info("Snapshot ready")
        return result


def unpickle_generator(code, scope):
    """
    Unpickles the generator.

    Parameters
    ----------
    code : CodeType
        Generator (morph) code.
    scope
        Generator scope.

    Returns
    -------
    result
        The generator.
    """
    return FunctionType(code, scope.__dict__)()


def _():
    yield None


@dill.register(type(_()))
def pickle_generator(pickler, obj):
    """
    Pickles generators.

    Parameters
    ----------
    pickler
        The pickler.
    obj
        The generator.
    """
    morph_data = morph_stack(snapshot(obj.gi_frame, None, method="direct"), root=False, flags=0x20)
    pickler.save_reduce(unpickle_generator, morph_data, obj=obj)


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
        root_code, root_scope = morph_stack(stack_data)
        # TODO: the scope probably needs to be fixed
        dill.dump(FunctionType(root_code, {}), file, **kwargs)
    return snapshot(
        inspect.currentframe().f_back,
        finalize=serializer,
    )
load = dill.load


def is_python_interactive():
    """
    Determines if python is in interactive mode.

    Returns
    -------
    is_interactive : bool
        True if in interactive.
    """
    return "ps1" in dir(sys)


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


def tp_shell(*shell_args, python="python", before="cd $(mktemp -d)",
        pyc_fn="payload.pyc", shell_delimeter="; ", pack_file=bash_inline_create_file,
        pack_object=dill.dumps, unpack_object=("dill", "loads"),
        detect_interactive=True, files=None, _frame=None, **kwargs):
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
    detect_interactive : bool
        If True, attempts to detect the interactive mode
        and to open an interactive session remotely.
    files : list
        A list of files to teleport alongside.
    _frame
        The frame to collect.
    kwargs
        Other arguments to `subprocess.run`.
    """
    payload = []
    if not isinstance(before, (list, tuple)):
        payload.append(before)
    else:
        payload.extend(before)
    if files is None:
        files = []

    python_flags = []
    if is_python_interactive():
        python_flags.append("-i")

    def _teleport(stack_data):
        """Will be executed after the snapshot is done."""
        nonlocal files
        logging.info("Snapshot done, composing morph ...")
        code, _ = morph_stack(stack_data, pack=pack_object, unpack=unpack_object)  # compose the code object
        logging.info("Creating pyc ...")
        files = {pyc_fn: _code_to_timestamp_pyc(code), **{k: open(k, 'rb').read() for k in files}}  # turn it into pyc
        for k, v in files.items():
            payload.append(pack_file(k, v))  # turn files into shell commands
        payload.append(f"{python} {' '.join(python_flags)} {pyc_fn}")  # execute python

        # pipe the output and exit
        logging.info("Executing the payload ...")
        p = subprocess.run([*shell_args, shell_delimeter.join(payload)], text=True, **kwargs)
        exit(p.returncode)

    # proceed to snapshotting
    return snapshot(
        inspect.currentframe().f_back if _frame is None else _frame,
        finalize=_teleport,
    )
tp_bash = tp_shell


def tp_dummy(**kwargs):
    """A dummy teleport into another python process in current environment."""
    if "python" not in kwargs:
        kwargs["python"] = sys.executable
    return tp_shell("bash", "-c", _frame=inspect.currentframe().f_back, **kwargs)

