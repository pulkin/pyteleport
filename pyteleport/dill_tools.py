from types import GeneratorType, FunctionType
import dill
from dill import dumps

from .snapshot import snapshot
from .morph import morph_stack


def portable_loads(data: bytes):
    """
    A portable self-contained version of loads
    that does not use globals.

    Parameters
    ----------
    data
        Object data.

    Returns
    -------
    The object itself.
    """
    from dill import loads
    return loads(data)


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
    morph_fun = morph_stack(snapshot(obj.gi_frame, stack_method="direct"), root=False, flags=0x20)
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
