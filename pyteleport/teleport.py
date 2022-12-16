"""
Forking and teleporting using shell functions.

- `fork_shell()`: fork into shell;
- `tp_shell()`: teleport into shell and exit;
"""
from importlib._bootstrap_external import _code_to_timestamp_pyc
import subprocess
import base64
from shlex import quote
from pathlib import Path
import logging
import sys
import inspect
import socket

from .util import is_python_interactive, exit, format_binary
from .morph import morph_stack
from .snapshot import snapshot
from .storage import in_code_transmission_engine


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


def pyteleport_skip_stack(will_call):
    return inspect.getfullargspec(will_call).kwonlydefaults["_skip"] + 1


def fork_shell(*shell_args, python="python", before="cd $(mktemp -d)", wait="wait",
               pyc_fn="payload_{}.pyc", shell_delimiter="; ", non_blocking_delimiter="& ",
               pack_file=bash_inline_create_file, object_storage_protocol=in_code_transmission_engine,
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
    object_storage_protocol : storage_protocol
        A collection of functions governing initial serialization
        and de-serialization of the global storage dict.
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
    object_storage = {}
    payload = []
    if not isinstance(before, (list, tuple)):
        payload.append(before)
    else:
        payload.extend(before)
    if files is None:
        files = []
    if object_storage_protocol.on_startup is not None:
        logging.info("Deploying a socket to communicate with payloads")
        sock = socket.socket()
        sock.settimeout(0.1)
        sock.bind(('', 0))
        sock.listen()
        host, port = sock.getsockname()
        logging.info(f"{host}:{port}")
        # update shell arguments in case port forwarding is needed
        shell_args = tuple(i.format(host=host, port=port) for i in shell_args)

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
        logging.info(f"Assembling pyc #{i} ...")
        morph_fun = morph_stack(stack_data, tos=tos, object_storage=object_storage,
                                object_storage_protocol=object_storage_protocol,
                                root_unpack_globals=True)  # compose the morph fun
        logging.info("Creating pyc ...")
        pyc = _code_to_timestamp_pyc(morph_fun.__code__)
        logging.debug(f"  file size: {format_binary(len(pyc))}")
        files[pyc_fn.format(i)] = pyc
        if object_storage_protocol.on_startup is not None:
            payload_python.append(f"{python} {pyc_fn.format(i)} {port}")
        else:
            payload_python.append(f"{python} {pyc_fn.format(i)}")
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
    result = subprocess.Popen(shell_args, text=True, **kwargs)
    if object_storage_protocol.on_startup is not None:
        logging.info("Connecting to payload(s) ...")
        payloads_served = 0
        while payloads_served < len(n):
            try:
                conn, addr = sock.accept()
            except TimeoutError:
                pass
            else:
                with conn:
                    payloads_served += 1
                    logging.info(f"  accepted {addr} {payloads_served}/{len(n)}, serving ...")
                    object_storage_protocol.on_startup(object_storage, conn)
            if result.poll() is not None:
                logging.info(f"Subprocess terminated before all payloads served (served: {payloads_served})")
                break
        else:
            logging.info("All payloads served")
        sock.close()
    if result.wait() > 0:
        raise subprocess.SubprocessError(f"Remote pyteleport process exited with code {result.returncode}")
    return result


def tp_shell(*args, _skip=pyteleport_skip_stack(fork_shell), **kwargs):
    """Teleports into another shell and pipes output"""
    exit(fork_shell(*args, _skip=_skip, **kwargs).returncode)


tp_bash = tp_shell


def tp_dummy(dry_run=False, _skip=pyteleport_skip_stack(tp_shell), **kwargs):
    """A dummy teleport into another python process in current environment."""
    if dry_run:
        return
    if "python" not in kwargs:
        kwargs["python"] = sys.executable
    if "env" not in kwargs:
        # make module search path exactly as it is here
        kwargs["env"] = {"PYTHONPATH": ':'.join(str(Path(i).resolve()) for i in sys.path)}
    return tp_shell("bash", "-c", _skip=_skip, **kwargs)
