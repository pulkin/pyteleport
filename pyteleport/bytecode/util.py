from collections.abc import Iterator, Iterable
from typing import TypeVar


class IndexStorage(list):
    """
    Collects objects and assigns indices.
    """
    def store(self, x) -> int:
        """
        Store an object and return its index.

        Parameters
        ----------
        x
            Object to store.

        Returns
        -------
        The index of the object in this collection.
        """
        try:
            return self.index(x)
        except ValueError:
            self.append(x)
            return len(self) - 1
    __call__ = store

    def copy(self):
        return self.__class__(self)


class NameStorage(IndexStorage):
    """
    Collects names and assigns indices.
    """
    def store(self, s: str, derive_unique: bool = False) -> int:
        """
        Store a name and return its index.

        Parameters
        ----------
        s
            Name to store.
        derive_unique
            If True, derives a different name
            in case the provided name is
            already present.

        Returns
        -------
        The index of the name in this collection.
        """
        if derive_unique:
            s = unique_name(s, self)
        return super().store(s)


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


class Cell:
    def __init__(self):
        self.value = None


T = TypeVar("T")


def log_iter(source: Iterable[T], cell: Cell) -> Iterator[T]:
    """
    Saves yielded values into the provided cell.

    Parameters
    ----------
    source
        The source iterator.
    cell
        The cell to save to.

    Yields
    ------
    Values from the source iterator.
    """
    for v in source:
        cell.value = v
        yield v
