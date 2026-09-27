"""Encode and decode CPython 3.11+ zero-cost exception tables.

Exception-table offsets are byte offsets into ``CodeType.co_code``. CPython
stores them as unsigned six-bit varints in units of two-byte code words. The
first byte of every table entry additionally has bit 7 set as an entry marker.

This format and :func:`dis._parse_exception_table` are private CPython details.
Callers must therefore guard use of exception tables by the running Python
version rather than treating this module as a portable bytecode format.
"""

from types import CodeType
from typing import Iterable, Mapping, Union

from .opcodes import python_feature_exceptiontable
from .primitives import ExceptionCodeBlock

if python_feature_exceptiontable:
    from dis import _parse_exception_table


def _pack_varint(value: int, first: bool = False) -> bytes:
    """Encode one non-negative integer in CPython's six-bit varint format.

    Parameters
    ----------
    value
        Integer to encode.
    first
        Set bit 7 on the first output byte. CPython uses this to mark the
        beginning of each exception-table entry.

    Returns
    -------
    bytes
        The encoded integer.

    Raises
    ------
    ValueError
        If ``value`` is negative.
    """
    if value < 0:
        raise ValueError("exception table values must be non-negative")

    chunks = [value & 0x3F]
    value >>= 6
    while value:
        chunks.append(value & 0x3F)
        value >>= 6

    result = bytearray()
    for chunk in reversed(chunks):
        result.append(chunk | 0x40)
    result[-1] &= 0x3F
    if first:
        result[0] |= 0x80
    return bytes(result)


def pack_exception_table(handlers: Iterable[ExceptionCodeBlock]) -> bytes:
    """
    Pack exception handlers from Python 3.11 into bytes.

    Parameters
    ----------
    handlers
        Exception handlers to pack.

    Returns
    -------
    Packed data suitable for ``CodeType``'s ``exceptiontable`` argument.
    """
    result = bytearray()
    for start, end, target, depth, lasti in sorted(
        (entry.start, entry.end, entry.target, entry.depth, entry.lasti)
        for entry in handlers
    ):
        result.extend(_pack_varint(start // 2, first=True))
        result.extend(_pack_varint((end - start) // 2))
        result.extend(_pack_varint(target // 2))
        result.extend(_pack_varint((depth << 1) | int(lasti)))
    return bytes(result)


def unpack_exception_table(code: CodeType) -> list[ExceptionCodeBlock]:
    """
    Decode a CPython 3.11+ exception table.

    Parameters
    ----------
    code
        Code object whose ``co_exceptiontable`` to decode.

    Returns
    -------
    A list of exception table entries.
    """
    if not python_feature_exceptiontable:
        raise RuntimeError("exception tables are not supported by this Python version")

    return [
        ExceptionCodeBlock(
            start=entry.start,
            end=entry.end,
            target=entry.target,
            depth=entry.depth,
            lasti=entry.lasti,
        )
        for entry in _parse_exception_table(code)
    ]
