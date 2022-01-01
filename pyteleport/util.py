def lookup_nested(lookup, data, _inner=False):
    """
    Maps nested tuple to lookup values.

    Parameters
    ----------
    lookup : dict
        Lookup dictionary.
    data : tuple
        The data to map.
    _inner : bool
        Indicates a recursive call.

    Returns
    -------
    result : tuple
        The resulting tuple.
    """
    if not _inner:
        lookup = lookup.copy()
    if data in lookup:
        return lookup[data]
    if not isinstance(data, tuple):
        raise KeyError(f"{data} not found")
    result = tuple(lookup_nested(lookup, i, _inner=True) for i in data)
    lookup[data] = result
    return result
