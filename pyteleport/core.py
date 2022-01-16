import inspect
from collections import namedtuple
from functools import partial
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

from .frame import get_value_stack, get_block_stack
from .minias import Bytecode
from .morph import morph_stack
from .inject import prepare_patch_chain, chain_patches


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
