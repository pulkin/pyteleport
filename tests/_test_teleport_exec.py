"""
[True] hello
[True] vstack []
[True] bstack {empty_bstack}
[{dry_run}] vstack []
[{dry_run}] bstack {empty_bstack}
[{dry_run}] exec_globals['a']=56 exec_locals['b']=55
[{dry_run}] world
"""
from pyteleport import tp_dummy
from tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


code = compile("""
global a

print_stack_here(print_)
tp_dummy(**get_tp_args())
print_stack_here(print_)

b += 42
a = b + 1
""", "<teleport-exec>", "exec")


setup_verbose_logging()
print_("hello")
exec_globals = {
    "print_stack_here": print_stack_here,
    "print_": print_,
    "tp_dummy": tp_dummy,
    "get_tp_args": get_tp_args,
}
exec_locals = {"b": 13}
exec(code, exec_globals, exec_locals)
print_(f"{exec_globals['a']=} {exec_locals['b']=}")
print_("world")
