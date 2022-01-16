import inspect
from collections import namedtuple
import functools
from types import CodeType, FunctionType, GeneratorType
import logging
from importlib._bootstrap_external import _code_to_timestamp_pyc

import subprocess
import base64
from shlex import quote
import dill
import sys
import os
from pathlib import Path

from .mem_view import Mem
from .frame import get_value_stack, get_block_stack
from .minias import _dis, long2bytes, Bytecode, jump_multiplier
from .morph import morph_stack
from .bytecode import (
    EXTENDED_ARG,
    JUMP_ABSOLUTE,
    CALL_FUNCTION,
    UNPACK_SEQUENCE,
    POP_BLOCK,
    RETURN_VALUE,
)


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
        super().__init__()
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


def predict_stack_size(frame):
    """
    Attempts to predict the stack size of the frame
    by analyzing the bytecode.

    Parameters
    ----------
    frame : FrameObject
        Frame to process.

    Returns
    -------
    size : int
        The size of the value stack
    """
    code = Bytecode.disassemble(frame.f_code)
    opcode = code.by_pos(frame.f_lasti + 2)
    code.pos = code.index(opcode)  # for presentation
    logging.debug(f"Bytecode disassembly pos={opcode.pos}")
    for i in str(code).split("\n"):
        logging.debug(i)
    if opcode.stack_size is None:
        raise ValueError("Stack size information is not available")
    return opcode.stack_size - 1  # the returned value is not there yet


class FrameSnapshot(namedtuple("FrameSnapshot", ("scope", "code", "pos", "v_stack", "v_locals", "v_globals",
                                                 "v_builtins", "block_stack"))):
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


def p_check_integrity(patcher, f_next):
    """
    Checks the integrity of the frame.

    Parameters
    ----------
    patcher : FramePatcher
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    raise NotImplemented


def interactive_patcher(fun):
    """
    Decorates patchers.

    Parameters
    ----------
    fun : Callable
        The patching function.

    Returns
    -------
    Decorated patcher.
    """
    @functools.wraps(fun)
    def _wrapped(*args, f_next=None, **kwargs):
        action = fun(*args, **kwargs)
        if action in (None, "return") or f_next is None:
            logging.debug(f"  ⏎ {f_next}")
            return f_next
        elif action == "inline":
            logging.debug(f"  ⏎ {f_next}()")
            return f_next()
        elif isinstance(action, tuple):
            logging.debug(f"  ⏎ {f_next}, *action={action}")
            return f_next, *action
        else:
            raise ValueError(f"Unknown action returned: {action}")
    return _wrapped


@interactive_patcher
def p_jump_to(pos, patcher):
    """
    Jump to bytecode position.

    Parameters
    ----------
    pos : int
        Position to set.
    patcher : FramePatcher
    """
    if patcher.pos == pos - 2:
        return "inline"
    else:
        logging.debug(f"PATCH: jump_to {pos:d}")
        if patcher.pos != pos - 2:
            patcher.patch_current(expand_long([JUMP_ABSOLUTE, pos // jump_multiplier]), 2)
        patcher.patch([CALL_FUNCTION, 0], pos)  # call next
        patcher.commit()


@interactive_patcher
def p_set_bytecode(bytecode, patcher):
    """
    Patch: set the bytecode contents.

    Parameters
    ----------
    bytecode : bytearray
        Bytecode to overwrite.
    patcher : FramePatcher
    """
    logging.debug(f"PATCH: set_bytecode")
    patcher.patch(bytecode, 0)  # re-write the bytecode from scratch
    patcher.commit()
    return "inline"


@interactive_patcher
def p_place_beacon(beacon, patcher):
    """
    Patch: places the beacon.

    Parameters
    ----------
    beacon
        Beacon to place.
    patcher : FramePatcher

    Returns
    -------
    f_next : Callable
        Next function to call.
    beacon
        The beacon object.
    """
    logging.debug(f"PATCH: place_beacon {beacon}")
    patcher.patch_current([
        UNPACK_SEQUENCE, 2,
        CALL_FUNCTION, 0,  # calls f_next
        CALL_FUNCTION, 0,  # calls what f_next returns
    ], 2)
    patcher.commit()
    return beacon,


@interactive_patcher
def p_exit_block_stack(block_stack, patcher):
    """
    Patch: exits the block stack.

    Parameters
    ----------
    block_stack
        State of the block stack.
    patcher : FramePatcher

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    logging.debug(f"PATCH: exit block stack x{len(block_stack):d}")
    patcher.patch_current(
        [POP_BLOCK, 0] * len(block_stack) + [CALL_FUNCTION, 0], 2
    )
    patcher.commit()


def normalize_frames(topmost_frame):
    """
    Prepares a list of frames (top to bottom) to serialize.

    Parameters
    ----------
    topmost_frame : FrameObject, int
        The topmost frame to start from.

    Returns
    -------
    result : list
        A list of frames top to bottom.
    """
    if isinstance(topmost_frame, int):
        frame = inspect.currentframe()
        for i in range(topmost_frame):
            frame = frame.f_back
    else:
        frame = topmost_frame

    result = [frame]
    while frame.f_back is not None:
        frame = frame.f_back
        result.append(frame)
    return result


def snapshot_frame(frame):
    """
    Make a snapshot of locals, globals and other information.

    Parameters
    ----------
    frame : FrameObject
        The frame to snapshot.

    Returns
    -------
    result : FrameSnapshot
        The resulting snapshot.
    """
    return FrameSnapshot(
        scope=inspect.getmodule(frame),
        code=frame.f_code,
        pos=frame.f_lasti,
        v_stack=None,
        v_locals=frame.f_locals.copy(),
        v_globals=frame.f_globals,
        v_builtins=frame.f_builtins,
        block_stack=get_block_stack(frame),
    )


def prepare_patch_chain(frames, snapshots):
    """
    Prepares patches to restore the value stack using
    a beacon object.

    Parameters
    ----------
    frames : list
        Stack to work with.
    snapshots : list
        A list of FrameSnapshots where to write
        snapshots to.

    Returns
    -------
    patches : list
        A list of patches.
    """
    beacon = object()  # the beacon object
    notify_current = 0  # a variable that holds position in the stack

    def notify(_frame, f_next):
        """A callback to save stack items"""
        nonlocal notify_current, beacon
        logging.debug(f"Identify/collect object stack ...")
        snapshots[notify_current] = snapshots[notify_current]._replace(
            v_stack=get_value_stack(_frame, until=beacon))
        logging.info(f"  received {len(snapshots[notify_current].v_stack):d} items")
        notify_current += 1
        return f_next

    chain = []  # holds a chain of patches and callbacks
    for frame, snapshot_data in zip(frames, snapshots):
        original_code = bytearray(frame.f_code.co_code)  # store the original bytecode
        rtn_pos = original_code[::2].index(RETURN_VALUE) * 2  # figure out where it returns

        # note that the bytearray is intentional to guarantee the copy
        patcher = FramePatcher(frame)

        chain.append(partial(p_jump_to, 0, patcher))  # make room for further patches
        chain.append(partial(p_place_beacon, beacon, patcher))  # place the beacon
        chain.append(partial(notify, frame))  # collect value stack
        chain.append(partial(p_jump_to, 0, patcher))  # jump once more to make more room
        chain.append(partial(p_exit_block_stack, snapshot_data.block_stack, patcher))  # exit from "finally" statements
        chain.append(partial(p_jump_to, rtn_pos - 2, patcher))  # jump 1 opcode before return
        chain.append(partial(p_set_bytecode, original_code, patcher))  # restore the bytecode
    return chain


def chain_patches(patches):
    """
    Chains multiple patches into a single function.

    Parameters
    ----------
    patches : list
        A list of patch functions to chain.

    Returns
    -------
    result : Callable
        The resulting function to call.
    """
    prev = patches[-1]
    for i in patches[-2::-1]:
        prev = partial(i, f_next=prev)
    return prev


def snapshot(frame, finalize, stack_method=None):
    """
    Snapshot the stack starting from the given frame.

    Parameters
    ----------
    frame : FrameObject, int
        Topmost frame.
    finalize : Callable
        If specified, returns the result into this function
        and terminates.
    stack_method : {"inject", "direct", "predict"}
        Method to use for the stack:
        * `inject`: makes a snapshot of an active stack by
          patching stack frames and running bytecode snippets
          inside. The stack is destroyed and the result is
          returned into `finalize` function (required).
        * `direct`: makes a snapshot of an inactive stack
          by reading FrameObject structure fields. Can only
          be used with generator frames.
        * `predict`: attempts to analyze the bytecode and to
          derive the stack size based on bytecode instruction
          sequences.

    Returns
    -------
    rtn : object
        Depending on the method, this is either the snapshot
        itself or an object that has to be returned to the
        topmost frame to initiate frame collection through
        bytecode injection.
    """
    if stack_method is None:
        stack_method = "inject"
    assert stack_method in ("inject", "direct", "predict")
    if stack_method == "inject" and finalize is None:
        raise ValueError("For method='inject' finalize has to be set")

    # determine frame stack
    if isinstance(frame, int):
        frame += 1
    frames = normalize_frames(frame)
    logging.debug(f"Snapshot {len(frames)} frame(s) using stack_method='{stack_method}'")
    for i, f in enumerate(frames):
        logging.info(f"  frame #{i:02d}: {f}")

    result = []
    prev_builtins = None
    for frame in frames:
        logging.info(f"Frame: {frame}")

        # check builtins
        if prev_builtins is None:
            prev_builtins = frame.f_builtins
        else:
            assert prev_builtins is frame.f_builtins

        # save locals, globals, etc.
        logging.info("  saving snapshot ...")
        fs = snapshot_frame(frame)
        if stack_method == "direct":
            fs = fs._replace(v_stack=get_value_stack(frame))
        elif stack_method == "predict":
            fs = fs._replace(v_stack=get_value_stack(frame, depth=predict_stack_size(frame)))
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

    if stack_method == "inject":  # prepare patchers
        chain = prepare_patch_chain(frames, result)
        chain.append(partial(finalize, result))
        logging.info("Ready to collect frames")
        return chain_patches(chain)()

    else:
        logging.info("Snapshot ready")
        if finalize is not None:
            return finalize(result)
        else:
            return result


def unpickle_generator(code, scope):
    """
    Restores a generator.

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


@dill.register(GeneratorType)
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
    morph_data = morph_stack(snapshot(obj.gi_frame, None, stack_method="direct"), root=False, flags=0x20)
    pickler.save_reduce(unpickle_generator, morph_data, obj=obj)


def dump(file, stack_method=None, **kwargs):
    """
    Serialize the runtime into a file and exit.

    Parameters
    ----------
    file : File
        The file to write to.
    stack_method
        Stack collection method.
    kwargs
        Arguments to `dill.dump`.
    """
    def serializer(stack_data):
        root_code, root_scope = morph_stack(stack_data)
        # TODO: the scope probably needs to be fixed
        dill.dump(FunctionType(root_code, {}), file, **kwargs)
        file.close()
        os._exit(0)
    return snapshot(
        inspect.currentframe().f_back,
        finalize=serializer,
        stack_method=stack_method,
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
             pyc_fn="payload.pyc", shell_delimiter="; ", pack_file=bash_inline_create_file,
             pack_object=dill.dumps, unpack_object=("dill", "loads"),
             detect_interactive=True, files=None, stack_method=None,
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
    shell_delimiter : str
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
    stack_method
        Stack collection method.
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
    if detect_interactive and is_python_interactive():
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
        p = subprocess.run([*shell_args, shell_delimiter.join(payload)], text=True, **kwargs)
        os._exit(p.returncode)

    # proceed to snapshotting
    return snapshot(
        inspect.currentframe().f_back if _frame is None else _frame,
        finalize=_teleport,
        stack_method=stack_method,
    )


tp_bash = tp_shell


def tp_dummy(**kwargs):
    """A dummy teleport into another python process in current environment."""
    if "python" not in kwargs:
        kwargs["python"] = sys.executable
    if "env" not in kwargs:
        # make module search path exactly as it is here
        kwargs["env"] = {"PYTHONPATH": ':'.join(str(Path(i).resolve()) for i in sys.path)}
    return tp_shell("bash", "-c", _frame=inspect.currentframe().f_back, **kwargs)
