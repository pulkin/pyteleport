"""
`dill`-related tools.

- `pickle_generator`: pickles generator objects;
- `unpickle_generator`: unpickles generator objects;

Note that generator picklers are registered to `dill` on
importing this module.
"""
from types import GeneratorType, FunctionType
import dill

from .snapshot import snapshot
from .morph import morph_stack


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
    frame = obj.gi_frame
    frame_snapshot = snapshot(frame, stack_method="direct")
    morph_code = morph_stack(frame_snapshot, root=False, flags=0x20)
    morph_fun = FunctionType(
        morph_code,
        frame_snapshot[-1].v_globals,
        f"morph_into:{morph_code.co_name}",
        (),  # no arguments
        tuple(frame_snapshot[-1].c_free),
    )
    pickler.save_reduce(unpickle_generator, (morph_fun,), obj=obj)


def unpickle_generator(morph_fun):
    """
    Restores a generator.

    Parameters
    ----------
    morph_fun : FunctionType
        Generator (morph) function.

    Returns
    -------
    result
        The generator.
    """
    return morph_fun()


def register_generator():
    """
    Registers generator pickler.
    """
    dill.register(GeneratorType)(pickle_generator)
