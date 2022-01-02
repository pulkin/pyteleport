from ..core import get_value_stack_from_beacon, get_block_stack

import logging
import os
import sys
from inspect import currentframe


pid_on_init = os.getpid()
local_environ = os.environ.copy()
local_environ["PYTHONPATH"] = ":".join([os.getcwd(), *sys.path])


def get_arg_dict(**kwargs):
    for i in sys.argv[1:]:
        k, v = i.split("=")
        if k not in kwargs:
            raise KeyError(f"Unknown argument {k}")
        kwargs[k] = v
    return kwargs


def get_tp_args():
    params = get_arg_dict(local_environ="False", stack_method="inject")
    return {
        "env": {"False": None, "True": local_environ}[params["local_environ"]],
        "stack_method": params["stack_method"],
    }


def print_(*args, flush=True, **kwargs):
    return print(f"[{os.getpid() == pid_on_init}]", *args, flush=flush, **kwargs)


def setup_verbose_logging():
    logging.addLevelName(logging.DEBUG, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
    logging.addLevelName(5, "\033[1;35mBYTECODE\033[1;0m")
    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.basicConfig(level=0, stream=sys.stderr)


def repr_object(o):
    """
    Represent an object for testing purposes.

    Parameters
    ----------
    o
        Object to represent.

    Returns
    -------
    result : str
        The representation.
    """
    if isinstance(o, (str, bytes, int, float, type)) or o is None:
        return repr(o)
    return "!" + str(type(o))


def print_stack_here(log=print_):
    frame = currentframe().f_back
    log("vstack", '[' + (', '.join(map(repr_object,
                                       get_value_stack_from_beacon(frame, id(print_stack_here))
                                       ))) + ']')
    log("bstack", '[' + (', '.join(f'{i[0]}/{i[2]}' for i in get_block_stack(frame))) + ']')
