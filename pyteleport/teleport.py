from importlib._bootstrap_external import _code_to_timestamp_pyc
import subprocess
import base64
from shlex import quote
from pathlib import Path
import dill
import logging
import os
import sys
import inspect

from .util import is_python_interactive
from .snapshot import morph_stack, snapshot


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
