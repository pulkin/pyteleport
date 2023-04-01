import inspect

from .snapshot import snapshot_frame


class DummyContextManager:
    def __enter__(self, *args):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


dummy_context_manager = DummyContextManager()


def _contextlib_run_surrogate(self, fun):
    with self:
        return fun()


def _snapshot_this_frame():
    return snapshot_frame(inspect.currentframe().f_back)


_contextlib_run_surrogate_snapshot = _contextlib_run_surrogate(
    dummy_context_manager,
    _snapshot_this_frame,
)._replace(
    v_stack=(),
    v_cells=(),
    block_stack=(),
)


def construct_contextlib_run_surrogate(context, tos_plus_one):
    lookup = {
        dummy_context_manager: context,
        _snapshot_this_frame: None,
    }
    return _contextlib_run_surrogate_snapshot._replace(
        v_locals=tuple(lookup[i] for i in _contextlib_run_surrogate_snapshot.v_locals),
        tos_plus_one=tos_plus_one,
    )
