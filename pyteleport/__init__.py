from .teleport import tp_shell, tp_bash, tp_dummy, dump
from .dill_tools import register_generator, register_module_dicts
register_generator()
register_module_dicts()
del register_generator, register_module_dicts
