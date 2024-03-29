import sys
import logging
import os
from inspect import currentframe
from socket import gethostname

from ..frame import FrameWrapper
from ..snapshot import predict_stack_size
from ..util import BYTECODE_LOG_LEVEL
from ..bytecode.opcodes import python_feature_block_stack


def get_tp_args():
    dtypes = {"dry_run": lambda x: {"True": True, "False": False}[x]}
    result = {}
    for i in sys.argv[1:]:
        k, v = i.split("=")
        v = dtypes.get(k, str)(v)
        result[k] = v
    return result


class printer:
    def __init__(self, pid):
        self.pid = pid

    def __call__(self, *args, flush=True, **kwargs):
        return print(f"[{os.getpid() == self.pid}]", *args, flush=flush, **kwargs)


print_ = printer(os.getpid())


def setup_verbose_logging(level=0):
    logging.addLevelName(logging.DEBUG, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
    logging.addLevelName(BYTECODE_LOG_LEVEL, "\033[1;35mBYTECODE\033[1;0m")
    logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.basicConfig(level=level, stream=sys.stderr)


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


def print_stack_here(log, *args, rtn=None):
    frame = currentframe().f_back
    wrapper = FrameWrapper(frame)
    log("vstack", *args, '[' + (', '.join(map(
        repr_object,
        wrapper.get_value_stack(
            predict_stack_size(frame),
        ),
    ))) + ']')
    if python_feature_block_stack:
        log("bstack", *args, '[' + (', '.join(
            f'{i[0]}/{i[2]}'
            for i in wrapper.get_block_stack()
        )) + ']')
    else:
        log("bstack --")
    return rtn


def hello():
    """
    Prints hello message.
    """
    print(f"hello from {gethostname()} / pid {os.getpid()}", flush=True)
