from ..core import get_value_stack, get_block_stack

import logging
import os
import sys
from inspect import currentframe


def get_arg_dict(**kwargs):
    for i in sys.argv[1:]:
        k, v = i.split("=")
        if k not in kwargs:
            raise KeyError(f"Unknown argument {k}")
        kwargs[k] = v
    return kwargs


def get_tp_args():
    params = get_arg_dict(stack_method="inject")
    return {
        "stack_method": params["stack_method"],
    }


class printer:
    def __init__(self, pid):
        self.pid = pid

    def __call__(self, *args, flush=True, **kwargs):
        return print(f"[{os.getpid() == self.pid}]", *args, flush=flush, **kwargs)


print_ = printer(os.getpid())


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


def print_stack_here(log, *args):
    frame = currentframe().f_back
    log("vstack", *args, '[' + (', '.join(map(
        repr_object,
        get_value_stack(frame, until=print_stack_here),
    ))) + ']')
    log("bstack", *args, '[' + (', '.join(
        f'{i[0]}/{i[2]}'
        for i in get_block_stack(frame)
    )) + ']')
