"""
[True] hello
[True] vstack []
[True] bstack []
[{dry_run}] world
[{dry_run}] vstack []
[{dry_run}] bstack []
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args, print_stack_here

import asyncio


setup_verbose_logging()


async def async_fn():
    print_("hello")
    print_stack_here(print_)
    tp_dummy(**get_tp_args())
    print_("world")
    print_stack_here(print_)


asyncio.run(async_fn())
