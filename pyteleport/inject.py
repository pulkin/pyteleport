from functools import partial, wraps
import logging

from .frame import get_value_stack, get_value_stack_size
from .minias import _dis, long2bytes, jump_multiplier
from .mem import _unsafe_write_bytes
from .bytecode import (
    EXTENDED_ARG,
    JUMP_ABSOLUTE,
    CALL_FUNCTION,
    UNPACK_SEQUENCE,
    POP_BLOCK,
    RETURN_VALUE,
)


def expand_long(c):
    """Expands opcode arguments if they do not fit one byte"""
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


def _overlapping(s1, l1, s2, l2):
    e1 = s1 + l1
    e2 = s2 + l2
    return s1 < e2 and s2 < e1


class CodePatcher(dict):
    """Collects and applies patches to bytecodes."""
    def __init__(self, code):
        super().__init__()
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
        for pos, patch in self.items():
            assert len(patch) <= len(code), f"len(patch) = {len(patch)} > len(code) = {len(code)}"
            assert 0 <= pos <= len(code) - len(patch), f"Index {pos:d} out of range [0, {len(code) - len(patch)}]"
            _unsafe_write_bytes(code, patch, pos)
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

    def patch(self, patch, pos, autowrap=False):
        if autowrap and pos + len(patch) >= len(self):
            free = len(self) - pos
            assert len(patch) + 2 <= len(self), f"len(patch) = {len(patch)} is too large for len(code) = {len(self)}"
            self[pos] = list(patch[:free - 2]) + [JUMP_ABSOLUTE, 0]
            self[0] = patch[free - 2:]
        else:
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

    def patch_current(self, patch, pos, **kwargs):
        return self.patch(patch, pos + self.pos, **kwargs)


def interactive_patcher(fun):
    """
    Decorates patchers.

    Parameters
    ----------
    fun : Callable
        The patching function.

    Returns
    -------
    Decorated patcher.
    """
    @wraps(fun)
    def _wrapped(*args, f_next=None, **kwargs):
        action = fun(*args, **kwargs)
        if action in (None, "return") or f_next is None:
            logging.debug(f"  ⏎ {f_next}")
            return f_next
        elif action == "inline":
            logging.debug(f"  ⏎ {f_next}()")
            return f_next()
        elif isinstance(action, tuple):
            logging.debug(f"  ⏎ {f_next}, *action={action}")
            return f_next, *action
        else:
            raise ValueError(f"Unknown action returned: {action}")
    return _wrapped


@interactive_patcher
def p_jump_to(pos, patcher):
    """
    Jump to bytecode position.

    Parameters
    ----------
    pos : int
        Position to set.
    patcher : FramePatcher
    """
    if patcher.pos == pos:
        return "inline"
    else:
        logging.debug(f"PATCH: jump_to {pos:d}")
        if patcher.pos != pos - 2:
            patcher.patch_current(expand_long([JUMP_ABSOLUTE, pos // jump_multiplier]), 2)
        patcher.patch([CALL_FUNCTION, 0], pos)  # call next
        patcher.commit()


@interactive_patcher
def p_set_bytecode(bytecode, patcher):
    """
    Patch: set the bytecode contents.

    Parameters
    ----------
    bytecode : bytearray
        Bytecode to overwrite.
    patcher : FramePatcher
    """
    logging.debug(f"PATCH: set_bytecode")
    patcher.patch(bytecode, 0)  # re-write the bytecode from scratch
    patcher.commit()
    return "inline"


@interactive_patcher
def p_place_beacon(beacon, patcher):
    """
    Patch: places the beacon.

    Parameters
    ----------
    beacon
        Beacon to place.
    patcher : FramePatcher

    Returns
    -------
    f_next : Callable
        Next function to call.
    beacon
        The beacon object.
    """
    logging.debug(f"PATCH: place_beacon {beacon}")
    patcher.patch_current([
        UNPACK_SEQUENCE, 2,
        CALL_FUNCTION, 0,  # calls f_next
        CALL_FUNCTION, 0,  # calls what f_next returns
    ], 2, autowrap=True)
    patcher.commit()
    return beacon,


@interactive_patcher
def p_exit_block_stack(block_stack, patcher):
    """
    Patch: exits the block stack.

    Parameters
    ----------
    block_stack
        State of the block stack.
    patcher : FramePatcher

    Returns
    -------
    f_next : Callable
        Next function to call.
    """
    logging.debug(f"PATCH: exit block stack x{len(block_stack):d}")
    patcher.patch_current([POP_BLOCK, 0] * len(block_stack) + [CALL_FUNCTION, 0], 2, autowrap=True)
    patcher.commit()


def prepare_patch_chain(frames, snapshots):
    """
    Prepares patches to restore the value stack using
    a beacon object.

    Parameters
    ----------
    frames : list
        Stack to work with.
    snapshots : list
        A list of FrameSnapshots where to write
        snapshots to.

    Returns
    -------
    patches : list
        A list of patches.
    """
    beacon = object()  # the beacon object
    notify_current = 0  # a variable that holds position in the stack

    def notify(_frame, f_next):
        """A callback to save stack items"""
        nonlocal notify_current, beacon
        logging.debug(f"Identify/collect object stack ...")
        snapshot = snapshots[notify_current]
        vstack = get_value_stack(snapshot.v_stack, get_value_stack_size(_frame, beacon) + 1)
        snapshots[notify_current] = snapshot._replace(v_stack=vstack[:-1], tos_plus_one=vstack[-1])
        logging.info(f"  received {len(snapshots[notify_current].v_stack):d} items")
        notify_current += 1
        return f_next

    chain = []  # holds a chain of patches and callbacks
    for frame, snapshot_data in zip(frames, snapshots):
        original_code = bytearray(frame.f_code.co_code)  # store the original bytecode
        rtn_pos = original_code[::2].index(RETURN_VALUE) * 2  # figure out where it returns

        # note that the bytearray is intentional to guarantee the copy
        patcher = FramePatcher(frame)

        chain.append(partial(p_place_beacon, beacon, patcher))  # place the beacon
        chain.append(partial(notify, frame))  # collect value stack
        chain.append(partial(p_exit_block_stack, snapshot_data.block_stack, patcher))  # exit from "finally" statements
        chain.append(partial(p_jump_to, rtn_pos - 2, patcher))  # jump 1 opcode before return
        chain.append(partial(p_set_bytecode, original_code, patcher))  # restore the bytecode
    return chain


def chain_patches(patches):
    """
    Chains multiple patches into a single function.

    Parameters
    ----------
    patches : list
        A list of patch functions to chain.

    Returns
    -------
    result : Callable
        The resulting function to call.
    """
    prev = patches[-1]
    for i in patches[-2::-1]:
        prev = partial(i, f_next=prev)
    return prev
