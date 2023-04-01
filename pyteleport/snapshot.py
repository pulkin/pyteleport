"""
Making python frame snapshots.

- `snapshot()`: make a snapshot;
"""
import dis
from collections import namedtuple
from types import FunctionType, BuiltinFunctionType
import logging

from .frame import get_value_stack, get_block_stack, snapshot_value_stack, get_value_stack_size, get_locals
from .minias import Bytecode
from .util import log_bytecode
from .opcodes import CALL_METHOD, CALL_FUNCTION, CALL_FUNCTION_KW, CALL_FUNCTION_EX, LOAD_CONST, YIELD_VALUE
from .primitives import NULL


class FrameStackException(ValueError):
    pass


class FrameSnapshot(namedtuple("FrameSnapshot", (
        "code", "pos", "lineno", "v_stack", "v_locals", "v_cells", "v_globals",
        "v_builtins", "block_stack", "tos_plus_one",
))):
    """A snapshot of python frame"""
    slots = ()

    def __str__(self):
        return f'File "{self.code.co_filename}", line {self.lineno}, in {self.code.co_name}'

    def __repr__(self):
        code = self.code
        contents = []
        for i in "v_stack", "v_locals", "v_cells", "v_globals", "v_builtins", "block_stack":
            v = getattr(self, i)
            if v is None or len(v) == 0:
                pass
            else:
                contents.append(f"    {i}: {len(v):d}")
        result = '\n'.join([
            f'  {str(self)}',
            *contents,
            f'    instruction: #{self.pos} {self.current_opcode_repr}',
        ])

        try:
            with open(code.co_filename, 'r') as f:
                result += f"\n    > {list(f)[self.lineno - 1].strip()}"
        except:
            pass

        return result

    @property
    def current_opcode(self):
        if self.pos == -1:  # beginning
            return None
        if 0 <= self.pos < len(self.code.co_code):
            return self.code.co_code[self.pos]
        raise ValueError(f"invalid {self.pos=}")

    @property
    def current_opcode_repr(self):
        if self.current_opcode is None:
            return "<head>"
        else:
            return dis.opname[self.current_opcode]


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
    logging.debug(f"  predicting stack size for {opcode}: {opcode.stack_size}")
    for i in str(code).split("\n"):
        log_bytecode(i)
    if opcode.stack_size is None:
        raise ValueError("Failed to predict stack size")
    return opcode.stack_size - 1  # the returned value is not there yet


def normalize_frames(topmost_frame):
    """
    Prepares a list of frames (top to bottom) to serialize.

    Parameters
    ----------
    topmost_frame : FrameObject
        The topmost frame to start from.

    Returns
    -------
    result : list
        A list of frames top to bottom.
    """
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
    result = FrameSnapshot(
        code=frame.f_code,
        pos=frame.f_lasti,
        lineno=frame.f_lineno,
        v_stack=None,
        v_locals=None,
        v_cells=None,
        v_globals=frame.f_globals,
        v_builtins=frame.f_builtins,
        block_stack=get_block_stack(frame),
        tos_plus_one=None,
    )
    # save value stack object ids
    vs_snapshot = snapshot_value_stack(frame)

    # TODO: revise this
    if result.current_opcode in (YIELD_VALUE, None, LOAD_CONST):  # TODO: LOAD_CONST stands for YIELD_FROM
        # generator frame (None = generator never yielded)
        stack_size = get_value_stack_size(frame)  # frame has the value stack size set
        vstack = get_value_stack(vs_snapshot, stack_size)
        called = None

    elif result.current_opcode is CALL_METHOD:
        # ordinary frame, stack size unknown
        #   use bytecode heuristics
        stack_size = predict_stack_size(frame)
        vstack = get_value_stack(vs_snapshot, stack_size + 2)
        if vstack[-2] is not NULL:
            called = vstack[-2]  # bound method
        else:
            called = vstack[-1]

    elif result.current_opcode in (CALL_FUNCTION, CALL_FUNCTION_KW, CALL_FUNCTION_EX):
        # same as above, TOS+1 is "guaranteed: to be callable
        stack_size = predict_stack_size(frame)
        vstack = get_value_stack(vs_snapshot, stack_size + 1)
        called = vstack[-1]

    else:
        logging.error(f"Failed to interpret {result.current_opcode_repr} (bytecode follows)")
        for i in str(Bytecode.disassemble(result.code)).split("\n"):
            log_bytecode(i)
        raise NotImplementedError(f"Cannot interpret {result.current_opcode_repr}")

    v_locals, v_cells, v_free = get_locals(frame)
    result = result._replace(v_stack=vstack[:stack_size], v_locals=v_locals,
                             v_cells=v_cells + v_free, tos_plus_one=called)
    for i in repr(result).split("\n"):
        logging.debug(i)
    return result


def check_stack_continuity(snapshots):
    """
    Checks stack continuity.

    Parameters
    ----------
    snapshots : list
        Snapshots collected.
    """
    do_raise = False
    message = []

    for i, (frame, upper) in enumerate(zip(snapshots[:-1], snapshots[1:])):
        fun = upper.tos_plus_one
        message.append(str(frame))

        if not isinstance(fun, (FunctionType, BuiltinFunctionType)):
            message.append(f'  TOS+1 in the frame above is an unknown object:\n'
                           f'    {repr(fun)}')
            do_raise = True

        elif isinstance(fun, BuiltinFunctionType):
            message.append(f'  TOS+1 in the frame above is a built-in function or method\n'
                           f'    {repr(fun)}')
            do_raise = True

        elif fun.__code__ is not frame.code:
            code = fun.__code__
            message.append(f'  (TOS+1).__code__ from the frame above does not match the code object of the frame below\n'
                           f'    below: "{code.co_filename}" in {fun.__name__}'
                           f'    above: "{frame}')
            do_raise = True

    if do_raise:
        message.append(str(upper))
        raise FrameStackException(
            f"Recorded frame stack does not match TOS+1 analysis\n"
            f"Snapshot traceback (most recent call last):\n" + "\n".join(message[::-1]))


def snapshot(topmost_frame, stack_method="predict"):
    """
    Snapshots the frame stack starting from the frame
    provided.

    Parameters
    ----------
    topmost_frame : FrameObject
        Topmost frame.
    stack_method : {"direct", "predict"}
        Method to use for the stack:
        * "predict": attempts to analyze the bytecode and to
          derive the stack size based on bytecode instruction
          sequences.
        * "direct": makes a snapshot of an inactive stack
          by reading FrameObject structure fields. Can only
          be used with generator frames.

    Returns
    -------
    result : list
        A list of frame snapshots: from inner to outer.
    """
    if stack_method is None:
        stack_method = "predict"
    assert stack_method in ("predict", "direct")

    # determine the frame stack
    frames = normalize_frames(topmost_frame)
    logging.debug(f"Snapshot traceback (most recent call last) stack_method={repr(stack_method)}:")

    result = []
    prev_builtins = None
    for frame in frames:
        # check builtins
        if prev_builtins is None:
            prev_builtins = frame.f_builtins
        else:
            assert prev_builtins is frame.f_builtins
        result.append(snapshot_frame(frame))
    logging.debug("  verifying frame stack continuity ...")
    check_stack_continuity(result)
    return result
