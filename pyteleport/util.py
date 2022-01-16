import sys


def is_python_interactive():
    """
    Determines if python is in interactive mode.

    Returns
    -------
    is_interactive : bool
        True if in interactive.
    """
    return "ps1" in dir(sys)


def unique_name(prefix: str, collection) -> str:
    """
    Prepares a unique name that is not
    in the collection (yet).

    Parameters
    ----------
    prefix
        The prefix to use.
    collection
        Name collection.

    Returns
    -------
    A unique name.
    """
    if prefix not in collection:
        return prefix
    for i in range(len(collection) + 1):
        candidate = f"{prefix}{i:d}"
        if candidate not in collection:
            return candidate
