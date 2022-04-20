def truncate(s: str, target: int, suffix: str = "...") -> str:
    """
    Truncates a string.

    Parameters
    ----------
    s
        String to truncate.
    target
        Maximal resulting string size.
    suffix
        The suffix to use when indicating truncation.

    Returns
    -------
    The truncated string.
    """
    if len(s) <= target:
        return s
    else:
        assert len(suffix) < target
        return s[:target - len(suffix)] + suffix


def repr_truncated(o, target: int) -> str:
    """
    Truncated representation (`repr`).

    Parameters
    ----------
    o
        Object to represent.
    target
        Maximal resulting string size.

    Returns
    -------
    The string representation of an object.
    """
    o_repr = repr(o)
    if len(o_repr) > target:
        o_repr = f"<{truncate(type(o).__name__, target - 11)} instance>"
    return o_repr


def text_table(table, column_spec, delimiter: str = " ") -> str:
    """
    Renders a text table.

    Parameters
    ----------
    table
        Table cell data as a nested list.
    column_spec
        Column specification (size and alignment).
    delimiter
        A string used as a vertical delimiter.

    Returns
    -------
    The resulting table.
    """
    result = []
    for line in table:
        result_line = []
        cs_iter = iter(column_spec)
        for cell, (size, align) in zip(line, cs_iter):
            if isinstance(cell, tuple):
                cell, x = cell
                for i in range(x - 1):
                    _size, align = next(cs_iter)
                    size += _size + len(delimiter)

            if cell is None:
                cell = " " * size
            else:
                cell = truncate(cell, size)
            result_line.append({"left": str.ljust, "right": str.rjust}[align](cell, size))
        result.append(delimiter.join(result_line).rstrip())  # trailing spaces
    return "\n".join(result)
