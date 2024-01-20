from typing import Optional


def truncate(
        s: str,
        size: Optional[int],
        suffix: str = "...",
        left: str = "",
        right: str = "",
) -> str:
    """
    Truncates a string.

    Parameters
    ----------
    s
        String to truncate.
    size
        Maximal resulting string size.
    suffix
        The suffix to use when indicating truncation.
    left
        An optional left bracket.
    right
        An optional right bracket.

    Returns
    -------
    The truncated string.
    """
    if size is None:
        size = float("+inf")
    size -= len(left) + len(right)
    assert size > 0
    if len(s) <= size:
        return f"{left}{s}{right}"
    else:
        assert len(suffix) < size
        return f"{left}{s[:size - len(suffix)]}{suffix}{right}"


def str_truncated(o, size: Optional[int]) -> str:
    """
    Turn object into a string up to the specified length.

    Parameters
    ----------
    o
        Object to represent.
    size
        Maximal resulting string size.

    Returns
    -------
    The string representation of an object.
    """
    return truncate(str(o), size)


def repr_truncated(o, size: Optional[int]) -> str:
    """
    Truncated representation (`repr`).

    Parameters
    ----------
    o
        Object to represent.
    size
        Maximal resulting string size.

    Returns
    -------
    The string representation of an object.
    """
    o_repr = repr(o)
    if size is not None and len(o_repr) > size:
        return truncate(
            type(o).__name__,
            size,
            left="<",
            right=" instance>",
        )
    return o_repr


def int_diff(base: int, added: int, removed: int, limit: int) -> str:
    if base + max(0, added - removed) <= limit:
        changed = min(added, removed)
        return "." * (base - removed) + "*" * changed + "-" * (removed - changed) + "+" * (added - changed)
    else:
        return f"{base}->{base - removed + added}"
