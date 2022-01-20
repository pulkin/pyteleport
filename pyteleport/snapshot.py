import inspect
from collections import namedtuple
from types import CodeType, FunctionType, GeneratorType, ModuleType
import logging
import dill

from .frame import get_value_stack, get_block_stack
from .minias import Bytecode
from .morph import morph_stack
from .inject import prepare_patch_chain, chain_patches
from .util import exit, log_bytecode


class FrameSnapshot(namedtuple("FrameSnapshot", (
        "scope", "code", "pos", "lineno", "v_stack", "v_locals", "v_globals",
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
            return f" {_name}: ({len(_what)})"

        def _repr_scope(_what):
            if isinstance(_what, ModuleType):
                return f"<{_what.__name__}>"
            elif isinstance(_what, FunctionType):
                return _what.__name__
            else:
                return repr(_what)

        result = f'  File "{code.co_filename}", line {self.lineno}, in {_repr_scope(self.scope)}\n' \
                 f'  [instr: #{self.pos}{_len("locals", self.v_locals)}{_len("stack", self.v_stack)}' \
                 f'{_len("block_stack", self.block_stack)}]'

        try:
            with open(code.co_filename, 'r') as f:
                result += f"\n    {list(f)[self.lineno - 1].strip()}"
        except:
            pass

        return result


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
        scope=inspect.getmodule(frame),
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
    for frame, upper in zip(snapshots[:-1], snapshots[1:]):
        if not isinstance(upper.tos_plus_one, FunctionType):
            raise RuntimeError(f"{upper} TOS+1={upper.tos_plus_one} is not a function type")
        if upper.tos_plus_one.__code__ is not frame.code:
            raise RuntimeError(f"{upper} TOS+1={upper.tos_plus_one} does not match code of the next frame {frame.code}")


def snapshot(topmost_frame, stack_method="direct"):
    """
    Snapshots the frame stack starting from the frame
    provided.

    Parameters
    ----------
    topmost_frame : FrameObject
        Topmost frame.
    stack_method : {None, "direct", "predict"}
        Method to use for the stack:
        * "direct": makes a snapshot of an inactive stack
          by reading FrameObject structure fields. Can only
          be used with generator frames.
        * "predict": attempts to analyze the bytecode and to
          derive the stack size based on bytecode instruction
          sequences.
        * `None`: no value stack collected.

    Returns
    -------
    result : list
        A list of frame snapshots.
    """
    assert stack_method in (None, "direct", "predict")

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
        if stack_method == "direct":
            vstack = get_value_stack(frame, depth=-2)  # -2 = capture 1 additional item
        elif stack_method == "predict":
            vstack = get_value_stack(frame, depth=predict_stack_size(frame) + 1)  # capture 1 additional item
        else:
            vstack = None
        if vstack is not None:
            fs = fs._replace(v_stack=vstack[:-1], tos_plus_one=vstack[-1])
        result.append(fs)
    if stack_method is not None:
        check_stack_continuity(result)
    return result


def snapshot_to_exit(topmost_frame, finalize, stack_method="direct"):
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
    stack_method : {None, "inject", "direct", "predict"}
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
        * `None`: no value stack collected.
    """
    assert stack_method in (None, "inject", "direct", "predict")
    result = snapshot(
        topmost_frame,
        stack_method=stack_method if stack_method != "inject" else None,
    )
    frames = normalize_frames(topmost_frame)

    def _finalize():
        finalize(result)
        exit()

    if stack_method == "inject":  # prepare patchers
        chain = prepare_patch_chain(frames, result)
        chain.append(_finalize)
        logging.info("Collecting frames through code injection")
        return chain_patches(chain)()

    else:
        _finalize()


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
    return snapshot_to_exit(
        inspect.currentframe().f_back,
        finalize=serializer,
        stack_method=stack_method,
    )


load = dill.load
