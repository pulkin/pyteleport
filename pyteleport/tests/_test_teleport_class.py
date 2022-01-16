"""
[True] hello
[True] vstack [!<class 'tuple_iterator'>]
[True] bstack []
[False] world
[False] vstack [!<class 'tuple_iterator'>]
[False] bstack []
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


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
