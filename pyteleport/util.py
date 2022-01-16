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
