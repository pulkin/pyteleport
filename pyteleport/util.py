import sys
import os
from functools import partial
from logging import log


BYTECODE_LOG_LEVEL = 5
log_bytecode = partial(log, BYTECODE_LOG_LEVEL)


def is_python_interactive() -> bool:
    """
    Determines if python is in interactive mode.

    Returns
    -------
    is_interactive : bool
        True if in interactive.
    """
    return "ps1" in dir(sys)


def unique_name(prefix: str, collection) -> str:
    """
    Prepares a unique name that is not
    in the collection (yet).

    Parameters
    ----------
    prefix
        The prefix to use.
    collection
        Name collection.

    Returns
    -------
    A unique name.
    """
    if prefix not in collection:
        return prefix
    for i in range(len(collection) + 1):
        candidate = f"{prefix}{i:d}"
        if candidate not in collection:
            return candidate


def exit(code: int = 0, flush_stdio: bool = True):
    """
    Exits the interpreter.

    Parameters
    ----------
    code
        Exit code.
    flush_stdio
        If True, flushes stdio.
    """
    if flush_stdio:
        for i in (sys.stdout, sys.stderr):
            i.flush()
    os._exit(code)
