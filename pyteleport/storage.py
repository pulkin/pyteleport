"""
Object storage tools.
"""
from dill import dumps as dill_dumps


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


class LocalStorage(dict):
    def __init__(self, loads=None, dumps=None):
        assert (loads is None and dumps is None) or (loads is not None and dumps is not None),\
            "Both loads=... and dumps=... have to be specified"
        if loads is None:
            loads = portable_dill_loads
            dumps = dill_dumps
        self.loads = loads
        self.dumps = dumps

    def store(self, obj):
        handle = id(obj)
        self[handle] = obj
        return handle

    def __str__(self):
        return f"{self.__class__.__name__}({len(self):d} items)"
