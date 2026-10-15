"""
`dill`-related tools.

- `pickle_generator`: pickles generator objects;
- `unpickle_generator`: unpickles generator objects;
- `register_module_dicts`: preserves runtime changes to module namespaces;

Picklers are registered to `dill` when importing `pyteleport`.
"""
from types import GeneratorType, ModuleType
import dill
import sys
from dill._dill import _import_module, _save_with_postproc, _setitems, save_module_dict, save_module
from dill._shims import Reduce

from .snapshot import snapshot
from .morph import morph_stack
from .primitives import NULL


_module_dicts = {}


def pickle_generator(pickler, obj):
    """
    Pickles generators.

    Parameters
    ----------
    pickler
        The pickler.
    obj
        The generator.
    """
    frame = obj.gi_frame
    frame_snapshot, = snapshot(reference_frame=frame)
    morph_fun = morph_stack([frame_snapshot], flags=0x20, tos=None if frame_snapshot.f_lasti is not None else NULL)
    pickler.save_reduce(unpickle_generator, (morph_fun,), obj=obj)


def unpickle_generator(morph_fun):
    """
    Restores a generator.

    Parameters
    ----------
    morph_fun : FunctionType
        Generator (morph) function.

    Returns
    -------
    result
        The generator.
    """
    return morph_fun()


def register_generator():
    """
    Registers generator pickler.
    """
    dill.register(GeneratorType)(pickle_generator)


def module_state(obj):
    """Save added/replaced bindings; an import supplies unchanged values."""
    # The receiving __main__ is the service entry point, not an import of
    # the original script. Its entire namespace must travel with the payload.
    original = {} if obj.__name__ == '__main__' else _module_dicts[id(obj)]
    return {
        name: value for name, value in vars(obj).items()
        if name not in ('__builtins__', '__loader__')
        and not (name in original and original[name] is value)
    }


def pickle_module(pickler, obj):
    """Import a module and restore its current namespace, including patches."""
    if obj.__name__ == 'dill' or obj.__name__.startswith('dill.'):
        save_module(pickler, obj)
    elif obj.__name__ != '__main__' and id(obj) not in _module_dicts:
        pickler.save_reduce(_import_module, (obj.__name__,), obj=obj)
    else:
        state = module_state(obj)
        pickler.save_reduce(_import_module, (obj.__name__,), obj=obj, state=state)


def pickle_module_dict(pickler, obj):
    """Restore module dictionaries in place so globals retain their identity."""
    name = obj.get('__name__')
    module = sys.modules.get(name) if isinstance(name, str) else None
    # Serializing dill's own dispatch tables would recursively serialize the
    # serialization machinery. __main__ needs state even without a snapshot.
    if (module is not None and getattr(module, '__dict__', None) is obj
            and (name == '__main__' or id(module) in _module_dicts)
            and name != 'dill' and not name.startswith('dill.')):
        update = (_setitems, (obj, module_state(module)))
        postproc_list = [update]
        if pickler._postproc:
            # A function may be waiting for this dictionary before it can be
            # memoized. Populate globals after the outermost pending object
            # exists, so namespace cycles never fall back to import references.
            next(iter(pickler._postproc.values())).append(update)
            postproc_list = []
        _save_with_postproc(
            pickler, (getattr, (Reduce(_import_module, (name,)), '__dict__')),
            obj=obj, postproc_list=postproc_list,
        )
    else:
        save_module_dict(pickler, obj)


def register_module_dicts():
    """Preserve added/replaced globals relative to importing pyteleport.

    Capture loaded namespaces once, before user patches. Keeping definitions
    importable avoids serializing interpreter internals such as import hooks.
    Use the payload's pickler so patches share its memo, unlike dump_module's
    separate session pickler. Dill's post-processing preserves globals identity
    and cycles even when a module dictionary is encountered first.

    This is a shallow binding snapshot: deletions and in-place mutations of
    startup objects are not captured. Modules without a startup snapshot are
    saved entirely by import reference, except __main__, whose namespace is
    saved in full because the receiving interpreter runs a different script.
    """
    global _module_dicts
    if not _module_dicts:
        _module_dicts = {id(module): vars(module).copy() for module in tuple(sys.modules.values())
                        if isinstance(module, ModuleType) and module.__name__ != '__main__'}
    dill.register(ModuleType)(pickle_module)
    dill.register(dict)(pickle_module_dict)
