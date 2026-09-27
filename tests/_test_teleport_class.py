"""
[True] hello
[True] vstack [!<class 'tuple_iterator'>]
[True] bstack {empty_bstack}
[{dry_run}] world
[{dry_run}] vstack [!<class 'tuple_iterator'>]
[{dry_run}] bstack {empty_bstack}
"""
from pyteleport import tp_dummy
from tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


class SomeClass:
    def __init__(self):
        self.messages = "hello", "world"

    def teleport(self):
        for m in self.messages:
            print_(m)
            print_stack_here(print_)
            if m is self.messages[0]:
                tp_dummy(**get_tp_args())


setup_verbose_logging()
instance = SomeClass()
instance.teleport()
