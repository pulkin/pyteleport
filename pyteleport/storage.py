"""
Object storage tools.
"""
from dill import dumps as dill_dumps
from collections import namedtuple


storage_protocol = namedtuple("storage_protocol", ("save_to_code", "load_from_code", "load_on_startup"))


def portable_dill_loads(data: bytes):
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


in_code_storage_protocol = storage_protocol(save_to_code=dill_dumps, load_from_code=portable_dill_loads,
                                            load_on_startup=None)
