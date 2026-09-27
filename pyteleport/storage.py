"""
Object storage tools.
"""
import dill
from dill import dumps as dill_dumps
from collections import namedtuple


transmission_engine = namedtuple("transmission_engine", ("save_to_code", "load_from_code", "on_startup"))


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


in_code_transmission_engine = transmission_engine(save_to_code=dill_dumps,
                                                  load_from_code=portable_dill_loads,
                                                  on_startup=None)


def _beacon(x: int) -> bytes:
    return f"expect object transmission {x:08x}\n".encode()


def stream_storage_out(object_storage, conn):
    """
    Streams the storage data over stdio.

    Parameters
    ----------
    object_storage
        Storage to stream.
    conn
        Socket connection.
    """
    with conn.makefile('rwb') as fd:
        dill.dump(object_storage, fd)


def portable_stream_storage_in(handle):
    """
    Receives storage data stream over stdio.

    Parameters
    ----------
    handle
        Handle (object id) (not) used for communication.

    Returns
    -------
    The resulting storage.
    """
    import sys
    import socket
    import dill
    port = int(sys.argv[1])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', port))
    with s.makefile('rwb') as fd:
        result = dill.load(fd)
    return result


socket_transmission_engine = transmission_engine(save_to_code=lambda object_storage: id(object_storage),
                                                 load_from_code=portable_stream_storage_in,
                                                 on_startup=stream_storage_out)
