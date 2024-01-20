from typing import Optional, Iterable


class Token:
    """
    A token interface represents  the bytecode during assembly.
    """
    def update_sequentially(self, prev: Optional["Token"]):
        """
        Updates this token after the previous token update.

        Parameters
        ----------
        prev
            The previous token in the sequence.
        """
        raise NotImplementedError

    @property
    def earlier_references_to_here(self) -> list["Token"]:
        """
        Earlier references to this token.
        """
        raise NotImplementedError

    def update_jump(self, reference: "Token") -> bool:
        """
        Updates this token after the reference token update.

        Parameters
        ----------
        reference
            The reference that was updated.

        Returns
        -------
        True to notify the size change; False otherwise.
        """
        raise NotImplementedError


class LookBackSequence:
    """
    A sequence with a possibility to look back.

    Parameters
    ----------
    source
        An iterable to source tokens from.
    """
    def __init__(self, source: Iterable[Token]):
        self.source = source
        self.next = {}
        self.current = None

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Optional[Token], Token]:
        """Outputs previous and current tokens in a sequence."""
        k = self.current
        try:
            self.current = self.next[k]
        except KeyError:
            nxt = next(self.source)
            if nxt in self.next or k is nxt:
                raise RuntimeError(f"non-unique token received: {nxt}")
            self.current = self.next[k] = nxt
        return k, self.current

    def restart_from_earlier(self, from_token: Optional[Token]):
        """
        Restarts iteration from one of the tokens yielded earlier.

        Parameters
        ----------
        from_token
            A token to restart from.
        """
        assert from_token in self.next
        self.current = from_token

    def reset(self):
        """Resets iteration from the very beginning."""
        self.restart_from_earlier(None)


def assemble(source: LookBackSequence):
    """
    Assembles the sequence.

    A very simple logic here: update tokens sequentially and rewind the sequence
    whenever the jump update requires to do so.

    Modifies sequence tokens in-place.

    Parameters
    ----------
    source
        The sequence of tokens with the possibility to look back.
    """
    for previous, token in source:
        token.update_sequentially(previous)

        # Update references
        stale = None
        for referencing_token in token.earlier_references_to_here:
            if referencing_token.update_jump(token) and stale is None:
                stale = referencing_token

        # If any reference indicates stale sequence state, restart from it
        if stale is not None:
            source.restart_from_earlier(stale)
