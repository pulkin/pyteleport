from importlib._bootstrap_external import _code_to_timestamp_pyc
import subprocess
import base64
from shlex import quote
from pathlib import Path
import logging
import sys
import inspect

from .util import is_python_interactive, exit
from .snapshot import morph_stack, snapshot
from .dill_tools import dumps, portable_loads


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


def fork_shell(*shell_args, python="python", before="cd $(mktemp -d)", wait="wait",
               pyc_fn="payload_{}.pyc", shell_delimiter="; ", non_blocking_delimiter="& ",
               pack_file=bash_inline_create_file, pack_object=dumps, unpack_object=portable_loads,
               detect_interactive=True, files=None, stack_method=None, n=1,
               _skip=1, **kwargs):
    """
    Fork into another shell.

    Parameters
    ----------
    shell_args
        Arguments to start a new shell.
    python : str
        Python executable in the shell.
    before : str, list
        Shell commands to be run before running python.
    wait : str
        Wait command.
    pyc_fn : str
        Temporary filename to save the bytecode to.
    shell_delimiter : str
        Shell delimiter to chain multiple commands.
    non_blocking_delimiter : str
        Shell delimiter to chain multiple commands without blocking.
    pack_file : Callable
        A function `f(name, contents)` turning a file
        into a shell-friendly assembly.
    pack_object : Callable, None
        A method (serializer) turning objects into bytes
        locally.
    unpack_object : Callable, None
        A method (deserializer) turning bytes into objects
        remotely. It does not have to rely on globals.
    detect_interactive : bool
        If True, attempts to detect the interactive mode
        and to open an interactive session remotely while
        piping stdio into this python process.
    files : list
        A list of files to teleport alongside.
    stack_method : str
        Stack collection method.
    n : {int, Iterator}
        If specified, spawns multiple processes within the
        same shell. For integers, spawns the specified
        number of processes identified by `range(n)`. For
        iterators, spawns a process per each value yielded.
    _skip : int
        The count of stack frames to skip.
    kwargs
        Other arguments to `subprocess.run`.

    Returns
    -------
    process
        The resulting process.
    """
    payload = []
    if not isinstance(before, (list, tuple)):
        payload.append(before)
    else:
        payload.extend(before)
    if files is None:
        files = []

    python_flags = []
    interactive_mode = detect_interactive and is_python_interactive()
    if interactive_mode:
        python_flags.append("-i")
    python = ' '.join([python, *python_flags])

    logging.info(f"Making a snapshot (skip {_skip} frames) ...")
    frame = inspect.currentframe()
    for i in range(_skip):
        frame = frame.f_back

    stack_data = snapshot(frame, stack_method=stack_method)

    if isinstance(n, int):
        n = range(n)
    files = {k: open(k, 'rb').read() for k in files}  # read all files

    payload_python = []
    for i, tos in enumerate(n):
        logging.info(f"Composing morph #{i} ...")
        morph_fun = morph_stack(stack_data, tos=tos, pack=pack_object, unpack=unpack_object)  # compose the code object
        logging.info("Creating pyc ...")
        files[pyc_fn.format(i)] = _code_to_timestamp_pyc(morph_fun.__code__)  # turn it into pyc
        payload_python.append(f"{python} {pyc_fn.format(i)}")  # execute python
    if interactive_mode and len(payload_python) > 1:
        raise ValueError("Multiple payloads are not compatible with interactive mode")

    # turn files into shell commands
    for k, v in files.items():
        payload.append(pack_file(k, v))
    if len(payload_python) > 1:
        payload_python.append(wait)  # block
        payload.append(non_blocking_delimiter.join(payload_python))  # execute python(s)
    else:
        payload.append(payload_python[0])

    # pipe the output and exit
    shell_args = [*shell_args, shell_delimiter.join(payload)]
    printable = (' '.join(shell_args)).split(' ')
    logging.info(f"Executing in subprocess\n"
                 f"  {' '.join(i if len(i) < 24 else i[:8] + '...' + i[-8:] for i in printable)}")
    return subprocess.run(shell_args, text=True, **kwargs)


def tp_shell(*args, **kwargs):
    """Teleports into another shell and pipes output"""
    kwargs["_skip"] = kwargs.get("_skip", 1) + 1
    exit(fork_shell(*args, **kwargs).returncode)


tp_bash = tp_shell


def tp_dummy(**kwargs):
    """A dummy teleport into another python process in current environment."""
    if "python" not in kwargs:
        kwargs["python"] = sys.executable
    if "env" not in kwargs:
        # make module search path exactly as it is here
        kwargs["env"] = {"PYTHONPATH": ':'.join(str(Path(i).resolve()) for i in sys.path)}
    kwargs["_skip"] = kwargs.get("_skip", 1) + 1
    return tp_shell("bash", "-c", **kwargs)
