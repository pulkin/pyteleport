import inspect
import dis
import ctypes
from collections import namedtuple
import functools
from itertools import groupby
from types import CodeType, FunctionType
import logging
import struct
from importlib._bootstrap_external import _code_to_timestamp_pyc

import subprocess
import base64
from shlex import quote
import dill
import sys

from .mem_view import Mem
from .py import (
    JX,
    ptr_frame_stack_bottom,
    ptr_frame_stack_top,
    ptr_frame_block_stack_bottom,
    ptr_frame_block_stack_top,
    put_NULL,
    put_EXCEPT_HANDLER,
    disassemble,
)
from .minias import _dis, long2bytes

locals().update(dis.opmap)
EXCEPT_HANDLER = 257


class partial(functools.partial):
    def __str__(self):
        return f"partial({self.func}, positional={len(self.args)}, kw=[{', '.join(self.keywords)}])"
    __repr__ = __str__


def _overlapping(s1, l1, s2, l2):
    e1 = s1 + l1
    e2 = s2 + l2
    return s1 < e2 and s2 < e1


class CodePatcher(dict):
    """Collects and applies patches to bytecodes."""
    def __init__(self, code):
        self._code = code

    def __str__(self):
        return f"CodePatcher(code={self._code})"

    def _diff(self):
        _new = list(self._code.co_code)
        for pos, patch in self.items():
            _new[pos:pos + len(patch)] = patch
        return _dis(self._code, alt=_new)

    def commit(self):
        logging.debug(f"Commit patch to <{self._code.co_name}>")
        for i in self._diff():
            logging.log(5, ''.join(i))
        code = self._code.co_code
        code_view = Mem.view(code)
        for pos, patch in self.items():
            assert len(patch) <= len(code), f"len(patch) = {len(patch)} > len(code) = {len(code)}"
            assert 0 <= pos <= len(code) - len(patch), f"Index {pos:d} out of range [0, {len(code) - len(patch)}]"
            code_view[pos:pos + len(patch)] = patch
        self.clear()

    @property
    def last_opcode(self):
        return self._code.co_code[-2]

    def __setitem__(self, pos, patch):
        patch = bytes(patch)
        code = self._code.co_code
        assert len(patch) <= len(code), f"len(patch) = {len(patch)} > len(code) = {len(code)}"
        assert 0 <= pos <= len(code) - len(patch), f"Index {pos:d} out of range [0, {len(code) - len(patch)}]"
        for _pos, _other in self.items():
            if _overlapping(pos, len(patch), _pos, len(_other)):
                raise ValueError("Patches overlap")
        super().__setitem__(pos, patch)

    def patch(self, patch, pos):
        self[pos] = patch

    def __len__(self):
        return len(self._code.co_code)


class FramePatcher(CodePatcher):
    """Collects and applies patches to bytecodes."""
    def __init__(self, frame):
        self._frame = frame
        super().__init__(frame.f_code)

    def __str__(self):
        return f"FramePatcher(frame={self._frame})"

    @property
    def pos(self):
        return self._frame.f_lasti

    def _diff(self):
        result_ = super()._diff()
        result = []
        for i, l in enumerate(result_):
            if 2 * i == self.pos:
                if l[0].startswith('\033'):
                    result.append(('\033[92m', *l[1:]))
                else:
                    result.append(('\033[92m', *l, '\033[0m'))
            else:
                result.append(l)
        return result

    @property
    def current_opcode(self):
        return self._code.co_code[self.pos]

    def patch_current(self, patch, pos):
        return self.patch(patch, pos + self.pos)


def expand_long(c):
    """Expands opcode arguments if they do not fit byte"""
    result = []
    for opcode, val in zip(c[::2], c[1::2]):
        if not val:
            result.extend([opcode, val])
        else:
            bts = long2bytes(val)
            for b in bts[:-1]:
                result.extend([EXTENDED_ARG, b])
            result.extend([opcode, bts[-1]])
    return bytes(result)


class NULL:
    """Represents NULL"""
    def __new__(cls):
        return cls

    def __str__(self):
        return "<NULL>"
    __repr__ = __str__


def get_value_stack_from_beacon(frame, beacon, expand=0, null=NULL()):
    """
    Collects frame stack using beacon as
    an indicator of stack top.

    Parameters
    ----------
    frame : FrameObject
        Frame to process.
    beacon : int
        Value on top of the stack.
    expand : int
    null
        The NULL object replacement.

    Returns
    -------
    stack : list
        Stack contents.
    """
    logging.debug(f"collecting stack for {frame} with beacon 0x{beacon:016x}")
    stack_bot = ptr_frame_stack_bottom(frame)
    stack_view = Mem(stack_bot, (frame.f_code.co_stacksize + expand) * 8)
    logging.debug(f"  contents:\n{stack_view}")
    stack_view = stack_view[:]
    result = []
    for i in range(0, len(stack_view), 8):
        obj_ref = int.from_bytes(stack_view[i:i + 8], "little")
        logging.debug(f"  object at 0x{obj_ref:016x} ...")
        if obj_ref == beacon:
            logging.debug(f"    <beacon>")
            return result
        if obj_ref == 0:
            result.append(null)
            logging.debug(f"    (null)")
        else:
            result.append(ctypes.cast(obj_ref, ctypes.py_object).value)
            logging.debug(f"    {repr(result[-1])}")
    raise RuntimeError("Failed to determine stack top")


def get_value_stack(frame):
    """
    Collects frame stack for generator objects.

    Parameters
    ----------
    frame : FrameObject
        Frame to process.

    Returns
    -------
    stack : list
        Stack contents.
    """
    stack_bot = ptr_frame_stack_bottom(frame)
    stack_top = ptr_frame_stack_top(frame)
    data = Mem(stack_bot, stack_top - stack_bot)[:]
    result = []
    for i in range(0, len(data), 8):
        obj_ref = int.from_bytes(data[i:i + 8], "little")
        result.append(ctypes.cast(obj_ref, ctypes.py_object).value)
    return result


block_stack_item = namedtuple('block_stack_item', ('type', 'handler', 'level'))


def get_block_stack(frame):
    """
    Collects block stack.

    Parameters
    ----------
    frame : FrameObject
        Frame to process.

    Returns
    -------
    stack : list
        Block stack contents.
    """
    fr = ptr_frame_block_stack_bottom(frame)
    to = ptr_frame_block_stack_top(frame)
    size = to - fr
    size4 = size // 4
    result = struct.unpack("i" * size4, Mem(fr, size)[:])
    result = tuple(block_stack_item(*x) for x in zip(result[::3], result[1::3], result[2::3]))
    return result


class FrameSnapshot(namedtuple("FrameSnapshot", ("code", "pos", "v_stack", "v_locals", "v_globals", "v_builtins", "block_stack"))):
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
        return f'FrameSnapshot {code.co_name} at "{code.co_filename}"+{code.co_firstlineno} @{self.pos:d} {" ".join(contents)}'


def p_jump_to(pos, patcher, f_next, jx=JX):
    """
    Patch: jump to position.

    Parameters
    ----------
    pos : int
        Position to set.
    patcher : FramePatcher
    f_next : Callable
    jx : int
        Jump address multiplier.

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    if patcher.pos == pos - 2:
        if f_next is not None:
            return f_next()  # already at the top: execute next
    else:
        logging.debug(f"jump_to {pos:d}: patching ...")
        if patcher.pos != pos - 2:
            patcher.patch_current(expand_long([JUMP_ABSOLUTE, pos // jx]), 2)
        patcher.patch([CALL_FUNCTION, 0], pos)  # call next
        patcher.commit()
        logging.debug(f"jump_to {pos:d}: ⏎ {f_next}")
        return f_next


def p_set_bytecode(bytecode, post, patcher, f_next):
    """
    Patch: set the bytecode contents.

    Parameters
    ----------
    bytecode : bytearray
        Bytecode to overwrite.
    post : Callable
        Call this before returning.
    patcher : FramePatcher
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    logging.debug(f"set_bytecode: patching ...")
    patcher.patch(bytecode, 0)  # re-write the bytecode from scratch
    patcher.commit()
    if post is not None:
        post()
    logging.debug(f"set_bytecode: ⏎ {f_next}")
    return f_next


def p_place_beacon(beacon, patcher, f_next):
    """
    Patch: places the beacon.

    Parameters
    ----------
    beacon
        Beacon to place.
    patcher : FramePatcher
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    beacon
        The beacon object.
    """
    logging.debug(f"place_beacon {beacon}: patching ...")
    patcher.patch_current([
        UNPACK_SEQUENCE, 2,
        CALL_FUNCTION, 0,  # calls f_next
        CALL_FUNCTION, 0,  # calls what f_next returns
    ], 2)
    patcher.commit()
    logging.debug(f"place_beacon {beacon}: ⏎ ({f_next}, {beacon})")
    return f_next, beacon


def p_exit_block_stack(block_stack, patcher, f_next):
    """
    Patch: exits the block stack.

    Parameters
    ----------
    block_stack
        State of the block stack.
    patcher : FramePatcher
    f_next : Callable

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    logging.debug(f"exit block stack {len(block_stack):d} times")
    patcher.patch_current(
        [POP_BLOCK, 0] * len(block_stack) + [CALL_FUNCTION, 0], 2
    )
    patcher.commit()
    logging.debug(f"exit block stack: ⏎ {f_next}")
    return f_next


def snapshot(frame, finalize, method="inject"):
    """
    Snapshot the stack starting from the given frame.

    Parameters
    ----------
    frame : FrameObject
        Top of the stack frame.
    finalize : Callable
        Where to return the result.
    method : {"inject", "direct"}
        Method to use for the stack:
        * `inject`: makes a snapshot of an active stack by
          patching stack frames and running bytecode snippets
          inside. The stack is destroyed and the result is
          returned into `finalize` function (required).
        * `direct`: makes a snapshot of an inactive stack
          by reading FrameObject structure fields. Can only
          be used with generator frames.

    Returns
    -------
    rtn : object
        Depending on the method, this is either the snapshot
        itself or an object that has to be returned to the
        subject frame to initiate invasive frame collection.
    """
    assert method in {"inject", "direct"}
    if method == "inject" and finalize is None:
        raise ValueError("For method='inject' finalize has to set")
    # determine the frame to start with
    logging.debug(f"Start frame serialization; mode: {'active' if finalize is not None else 'inactive'}")
    if frame is None:
        logging.info("  no frame specified")
        frame = 1
    if isinstance(frame, int):
        logging.info(f"  taking frame #{frame:d}")
        _frame = inspect.currentframe()
        for i in range(frame):
            _frame = _frame.f_back
        frame = _frame

    logging.info(f"  frame: {frame}")

    result = []
    if method == "inject":  # prepare to recieve data from patched frames
        beacon = object()  # beacon object

        notify_current = 0
        def notify(frame, f_next):
            """A callback to save stack items"""
            nonlocal notify_current, beacon
            logging.debug(f"Identify/collect object stack ...")
            result[notify_current] = result[notify_current]._replace(
                v_stack=get_value_stack_from_beacon(frame, id(beacon), expand=1))  # this might corrupt memory
            logging.info(f"  received {len(result[notify_current].v_stack):d} items")
            notify_current += 1
            return f_next

        chain = []  # holds a chain of patches and callbacks

    prev_globals = None
    prev_builtins = None

    while frame is not None:  # iterate over frame stack
        logging.info(f"Frame: {frame}")

        # check globals and builtins
        if prev_globals is None:
            prev_globals = frame.f_globals
        else:
            assert prev_globals is frame.f_globals
        if prev_builtins is None:
            prev_builtins = frame.f_builtins
        else:
            assert prev_builtins is frame.f_builtins

        # save locals, globals, etc.
        logging.info("  saving snapshot ...")
        fs = FrameSnapshot(
            code=frame.f_code,
            pos=frame.f_lasti,
            v_stack=None if method == "inject" else get_value_stack(frame),
            v_locals=frame.f_locals.copy(),
            v_globals=prev_globals,
            v_builtins=prev_builtins,
            block_stack=get_block_stack(frame),
        )
        logging.info(f"    code: {fs.code}")
        logging.info(f"    pos: {fs.pos}")
        logging.info(f"    stack: {len(fs.v_stack) if fs.v_stack is not None else 'none'}")
        logging.info(f"    locals: {len(fs.v_locals)}")
        logging.info(f"    globals: {len(fs.v_globals)}")
        logging.info(f"    builtins: {len(fs.v_builtins)}")
        logging.info(f"    block_stack:")
        if len(fs.block_stack):
            for i in fs.block_stack:
                logging.info(f"      {i}")
        else:
            logging.info("      (empty)")
        result.append(fs)

        if method == "inject":  # prepare patchers
            logging.info(f"  patching the bytecode ...")
            original_code = bytearray(frame.f_code.co_code)  # store the original bytecode
            rtn_pos = original_code[::2].index(RETURN_VALUE) * 2  # figure out where it returns
            # note that the bytearray is intentional to guarantee the copy
            patcher = FramePatcher(frame)

            p_jump_to(0, patcher, None)  # make room for patches immediately
            chain.append(partial(p_place_beacon, beacon, patcher))  # place the beacon
            chain.append(partial(notify, frame))  # collect value stack
            chain.append(partial(p_jump_to, 0, patcher))  # jump once more to make more room
            chain.append(partial(p_exit_block_stack, fs.block_stack, patcher))  # exit from "finally" statements
            chain.append(partial(p_jump_to, rtn_pos - 2, patcher))  # jump 1 opcode before return
            chain.append(partial(
                p_set_bytecode,
                original_code,
                None
                if frame.f_back is not None
                else partial(finalize, result),
                patcher
            ))  # restore the bytecode

        frame = frame.f_back  # next frame

    if method == "inject":  # chain patches
        prev = None
        for i in chain[::-1]:
            prev = partial(i, prev)
        logging.info("Ready to collect frames")
        return prev

    else:
        logging.info("Snapshot ready")
        return result


def unpickle_generator(code):
    """
    Unpickles the generator.

    Parameters
    ----------
    code : Codetype
        The morph code.

    Returns
    -------
    result
        The generator.
    """
    return FunctionType(code, globals())()


def _():
    yield None


@dill.register(type(_()))
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
    code = morph_stack(snapshot(obj.gi_frame, None, method="direct"), root=False, flags=0x20)
    pickler.save_reduce(
        unpickle_generator,
        (code,),
        obj=obj,
    )


def _iter_stack(value_stack, block_stack):
    """
    Iterates value and block stacks in the order they have to be placed.

    Parameters
    ----------
    value_stack : Iterable
        Value stack items.
    block_stack : Iterable
        Block stack items.

    Yields
    ------
    stack_item
        The next item to place.
    is_value : bool
        Indicates if the item belongs to the value stack.
    """
    v_stack_iter = enumerate(value_stack, start=1)
    cur_stack_level = 0
    for block_stack_item in block_stack:

        # check if items are coming in accending order
        if block_stack_item.level < cur_stack_level:
            raise ValueError(f"Illegal block_stack.level={block_stack_item.level}")

        # output stack items up to the block stack level
        while cur_stack_level < block_stack_item.level:
            try:
                cur_stack_level, stack_item = next(v_stack_iter)
            except StopIteration:
                raise StopIteration(f"Depleted value stack items ({cur_stack_level}) for block_stack.level={block_stack_item.level}")
            yield stack_item, True

        # output block stack item
        yield block_stack_item, False

    # output the rest of the value stack
    for _, stack_item in v_stack_iter:
        yield stack_item, True


def is_marshalable(o):
    """
    Determines if the object is marshalable.

    Parameters
    ----------
    o
        Object to test.

    Returns
    -------
    result : bool
        True if marshalable. False otherwise.
    """
    return isinstance(o, (str, bytes, int, float, complex))  # TODO: add lists, tuples and dicts


def morph_execpoint(p, nxt, pack=None, unpack=None, globals=False, locals=True, fake_return=True, flags=0):
    """
    Prepares a code object which morphs into the desired state
    and continues the execution afterwards.

    Parameters
    ----------
    p : execpoint
        The execution point to morph into.
    nxt : CodeType
        The code object which develops the stack further.
    pack : Callable, None
        A method turning objects into bytes (serializer)
        locally.
    unpack : tuple, None
        A 2-tuple `(module_name, method_name)` specifying
        the method that morph uses to unpack the data.
    globals : bool
        If True, unpacks globals.
    locals : bool
        If True, unpacks locals.
    fake_return : bool
        If set, fakes returning None by putting None on top
        of the stack. This will be ignored if nxt is not
        None.
    flags : int
        Code object flags.

    Returns
    -------
    result : CodeType
        The resulting morph.
    """
    assert pack is None and unpack is None or pack is not None and unpack is not None,\
        "Either both or none pack and unpack arguments have be specified"
    logging.info(f"Preparing a morph into execpoint {p} pack={pack is not None} ...")
    code = disassemble(p.code)
    code.pos = 0
    code.c("Header")
    code.nop(b'mrph')  # signature
    f_code = p.code
    new_stacksize = f_code.co_stacksize

    if pack:
        unpack_mod, unpack_method = unpack
        code.c(f"from {unpack_mod} import {unpack_method}")
        for i in range(len(code.co_varnames) + 1):
            unpack = f"{unpack_mod}_{unpack_method}{i:d}"
            if unpack not in code.co_varnames:
                break
        unpack = code.co_varnames(unpack)
        code.I(LOAD_CONST, 0)
        code.I(LOAD_CONST, (unpack_method,))
        code.I(IMPORT_NAME, unpack_mod)
        code.I(IMPORT_FROM, unpack_method)
        code.i(STORE_FAST, unpack)
        code.i(POP_TOP, 0)

        def _LOAD(_what):
            if is_marshalable(_what):
                code.I(LOAD_CONST, _what)
            else:
                code.i(LOAD_FAST, unpack)
                code.I(LOAD_CONST, pack(_what))
                code.i(CALL_FUNCTION, 1)
    else:
        def _LOAD(_what):
            code.I(LOAD_CONST, _what)

    scopes = []
    if locals:
        scopes.append((p.v_locals, STORE_FAST, "locals"))
    if globals:
        scopes.append((p.v_globals, STORE_GLOBAL, "globals"))
    for _dict, _STORE, log_name in scopes:
        logging.info(f"  {log_name} ...")
        if len(_dict) > 0:
            code.c(f"{log_name} = ...")
            klist, vlist = zip(*_dict.items())
            _LOAD(vlist)
            code.i(UNPACK_SEQUENCE, len(vlist))
            for k in klist:
                # k = v
                code.I(_STORE, k)
            new_stacksize = max(new_stacksize, len(vlist))

    # load block and value stacks
    code.c("*stack")
    stack_items = _iter_stack(p.v_stack, p.block_stack)
    for item, is_value in stack_items:
        if is_value:
            if item is NULL:
                put_NULL(code)
            else:
                _LOAD(item)
        else:
            if item.type == SETUP_FINALLY:
                code.i(SETUP_FINALLY, 0, jump_to=code.by_pos(item.handler * JX))
            elif item.type == EXCEPT_HANDLER:
                assert next(stack_items) == (NULL, True)  # traceback
                assert next(stack_items) == (NULL, True)  # value
                assert next(stack_items) == (None, True)  # type
                put_EXCEPT_HANDLER(code)
            else:
                raise NotImplementedError(f"Unknown block type={type} ({dis.opname.get(type, 'unknown opcode')})")

    if nxt is not None:
        # call nxt which is a code object
        code.c(f"nxt()")

        # load code object
        _LOAD(nxt)
        code.I(LOAD_CONST, None)  # function name
        code.i(MAKE_FUNCTION, 0)  # turn code object into a function
        code.i(CALL_FUNCTION, 0)  # call it
    elif fake_return:
        code.c(f"fake return None")
        code.I(LOAD_CONST, None)  # fake nxt returning None

    # now jump to the previously saved position
    code.c(f"goto saved pos")
    code.i(JUMP_ABSOLUTE, 0, jump_to=code.by_pos(p.pos + 2))

    code.c(f"---------------------")
    code.c(f"The original bytecode")
    code.c(f"---------------------")
    result = CodeType(
        0,
        0,
        0,
        len(code.co_varnames),
        new_stacksize + 1,
        flags,
        code.get_bytecode(),
        tuple(code.co_consts),
        tuple(code.co_names),
        tuple(code.co_varnames),
        f_code.co_filename,  # TODO: something smarter should be here
        f_code.co_name,
        f_code.co_firstlineno,  # TODO: this has to be fixed
        f_code.co_lnotab,
        )
    logging.info(f"resulting morph:\n{str(code)}")
    return result


def morph_stack(frame_data, root=True, **kwargs):
    """
    Morphs the stack.

    Parameters
    ----------
    frame_data : list
        States of all individual frames.
    root : bool
        Indicates if the stack contains a root
        frame where globals instead of locals
        need to be unpacked.
    kwargs
        Arguments to morph_execpoint.

    Returns
    -------
    result : CodeType
        The resulting morph for the root frame.
    """
    prev = None
    for i, frame in enumerate(frame_data):
        logging.info(f"Preparing morph #{i:d}")
        prev = morph_execpoint(frame, prev,
            globals=root and frame is frame_data[-1],
            locals=frame is not frame_data[-1] or not root,
            **kwargs)
    return prev


def dump(file, **kwargs):
    """
    Serialize the runtime into a file and exit.

    Parameters
    ----------
    file : File
        The file to write to.
    kwargs
        Arguments to `dill.dump`.
    """
    def serializer(stack_data):
        dill.dump(FunctionType(morph_stack(stack_data), globals()), file, **kwargs)
    return snapshot(
        inspect.currentframe().f_back,
        finalize=serializer,
    )
load = dill.load


def is_python_interactive():
    """
    Determines if python is in interactive mode.

    Returns
    -------
    is_interactive : bool
        True if in interactive.
    """
    return "ps1" in dir(sys)


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


def shell_teleport(*shell_args, python=None, before="cd $(mktemp -d)",
        pyc_fn="payload.pyc", shell_delimeter="; ", pack_file=bash_inline_create_file,
        pack_object=dill.dumps, unpack_object=("dill", "loads"),
        detect_interactive=True, _frame=None, **kwargs):
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
    shell_delimeter : str
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
    _frame
        The frame to collect.
    kwargs
        Other arguments to `subprocess.run`.
    """
    if python is None:
        python = sys.executable
    payload = []
    if not isinstance(before, (list, tuple)):
        payload.append(before)
    else:
        payload.extend(before)

    python_flags = []
    if is_python_interactive():
        python_flags.append("-i")

    def _teleport(stack_data):
        """Will be executed after the snapshot is done."""
        logging.info("Snapshot done, composing morph ...")
        code = morph_stack(stack_data, pack=pack_object, unpack=unpack_object)  # compose the code object
        logging.info("Creating pyc ...")
        files = {pyc_fn: _code_to_timestamp_pyc(code)}  # turn it into pyc
        for k, v in files.items():
            payload.append(pack_file(k, v))  # turn files into shell commands
        payload.append(f"{python} {' '.join(python_flags)} {pyc_fn}")  # execute python

        # pipe the output and exit
        logging.info("Executing the payload ...")
        p = subprocess.run([*shell_args, shell_delimeter.join(payload)], text=True, **kwargs)
        exit(p.returncode)

    # proceed to snapshotting
    return snapshot(
        inspect.currentframe().f_back if _frame is None else _frame,
        finalize=_teleport,
    )
bash_teleport = shell_teleport


def dummy_teleport(**kwargs):
    """A dummy teleport into another python process in current environment."""
    return bash_teleport("bash", "-c", _frame=inspect.currentframe().f_back, **kwargs)


