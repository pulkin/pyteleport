import sys
import dis
import inspect
from collections import namedtuple
from types import CodeType, FunctionType, GeneratorType, ModuleType, BuiltinFunctionType
import logging
import dill

from .frame import get_value_stack, get_block_stack, snapshot_value_stack, get_value_stack_size
from .minias import Bytecode
from .morph import morph_stack
from .util import exit, log_bytecode
from .bytecode import CALL_METHOD
from .primitives import NULL


class FrameStackException(ValueError):
    pass


def _repr_scope(_what):
    if isinstance(_what, ModuleType):
        return f"<{_what.__name__}>"
    elif isinstance(_what, FunctionType):
        return _what.__name__
    elif _what is None:
        return "(unknown)"
    else:
        return repr(_what)


class FrameSnapshot(namedtuple("FrameSnapshot", (
        "code", "pos", "lineno", "v_stack", "v_locals", "v_globals",
        "v_builtins", "block_stack", "tos_plus_one",
))):
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

        def _len(_name, _what):
            if _what is None or len(_what) == 0:
                return ''
            return f"\n    {_name}: ({len(_what)})"

        result = f'  File "{code.co_filename}", line {self.lineno}, in {_repr_scope(self.module)}\n' \
                 f'    instruction: #{self.pos} {dis.opname[self.current_opcode]}' \
                 f'{_len("locals", self.v_locals)}{_len("stack", self.v_stack)}' \
                 f'{_len("block_stack", self.block_stack)}'

        try:
            with open(code.co_filename, 'r') as f:
                result += f"\n\n    {list(f)[self.lineno - 1].strip()}"
        except:
            pass

        return result

    @property
    def current_opcode(self):
        return self.code.co_code[self.pos]

    @property
    def module(self):
        if "__name__" in self.v_globals:
            return sys.modules[self.v_globals["__name__"]]
        else:
            raise ValueError("Failed to determine the module globals belong to: __name__ is undefined")


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
    logging.debug("  predicting stack size ...")
    log_bytecode(f"  disassembly pos={opcode.pos}")
    for i in str(code).split("\n"):
        log_bytecode(i)
    if opcode.stack_size is None:
        raise ValueError("Stack size information is not available")
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
        v_locals=frame.f_locals.copy(),
        v_globals=frame.f_globals,
        v_builtins=frame.f_builtins,
        block_stack=get_block_stack(frame),
        tos_plus_one=None,
    )
    for i in str(result).split("\n"):
        logging.debug(i)
    return result


def check_stack_continuity(snapshots):
    """
    Checks stack continuity.

    Parameters
    ----------
    snapshots
        Snapshots collected.
    """
    for i, (frame, upper) in enumerate(zip(snapshots[:-1], snapshots[1:])):
        fun = upper.tos_plus_one
        message = None
        if not isinstance(fun, (FunctionType, BuiltinFunctionType)):
            message = f'  TOS+1 in the frame below is an unknown object:\n' + \
                      f'    {repr(fun)}\n'
        elif isinstance(fun, BuiltinFunctionType):
            message = f'  Built-in function or method\n' + \
                      f'    {fun}\n'
        elif fun.__code__ is not frame.code:
            code = fun.__code__
            message = f'  File "{code.co_filename}" in {_repr_scope(fun)}\n' + \
                      f'    (determined by analyzing value stack of the frame below)\n'
        if message is not None:
            raise FrameStackException(
                f"Frame stack is broken\nSnapshot traceback (most recent call last):\n" + \
                "\n".join(map(str, snapshots[:i + 1])) + \
                '\n  -----------------------\n' + \
                '  Frame stack breaks here\n' + \
                '  -----------------------\n' + \
                message + \
                "\n".join(map(str, snapshots[i + 1:]))
            )


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
        A list of frame snapshots.
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

        # save locals, globals, etc.
        fs = snapshot_frame(frame)
        # peek stands for capturing TOS+1 where, presumably,
        #    a callable object of the next stack frame is written
        peek = 1
        if fs.current_opcode == CALL_METHOD:
            peek = 2  # CALL_METHOD may accept callable at TOS+2

        if stack_method == "direct":
            stack_size = get_value_stack_size(frame)  # frame has the value stack size set
        elif stack_method == "predict":
            stack_size = predict_stack_size(frame)

        vstack = get_value_stack(
            snapshot_value_stack(frame),
            stack_size + peek,
        )
        for called in vstack[stack_size:]:
            if called is not NULL:
                break
        else:
            raise ValueError(f"Failed to find a callable in {vstack[stack_size]}")

        fs = fs._replace(v_stack=vstack[:stack_size], tos_plus_one=called)

        result.append(fs)
    logging.debug("  verifying frame stack continuity ...")
    check_stack_continuity(result)
    return result


def snapshot_to_exit(topmost_frame, finalize, stack_method=None):
    """
    Snapshots the stack starting from the frame provided,
    returns it to the `finalize` method and exits the
    interpreter.

    Parameters
    ----------
    topmost_frame : FrameObject
        Topmost frame.
    finalize : Callable
        The function to return the frame snapshot to.
    stack_method : {str, None}
        Method to use for the stack:
        * `predict`: attempts to analyze the bytecode and to
          derive the stack size based on bytecode instruction
          sequences.
        * `direct`: makes a snapshot of an inactive stack
          by reading FrameObject structure fields. Can only
          be used with generator frames.
    """
    result = snapshot(topmost_frame, stack_method=stack_method)
    finalize(result)
    exit()


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
    morph_data = morph_stack(snapshot(obj.gi_frame, stack_method="direct"), root=False, flags=0x20)
    pickler.save_reduce(unpickle_generator, morph_data, obj=obj)


def dump(file, stack_method=None, **kwargs):
    """
    Serialize the runtime into a file and exit.

    Parameters
    ----------
    file : File
        The file to write to.
    stack_method : str
        Stack collection method.
    kwargs
        Arguments to `dill.dump`.
    """
    def serializer(stack_data):
        root_code, root_scope = morph_stack(stack_data)
        # TODO: the scope probably needs to be fixed
        dill.dump(FunctionType(root_code, {}), file, **kwargs)
        file.close()
    return snapshot_to_exit(
        inspect.currentframe().f_back,
        finalize=serializer,
        stack_method=stack_method,
    )


load = dill.load
