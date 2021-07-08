![build](https://github.com/pulkin/pyteleport/actions/workflows/test.yml/badge.svg)

pyteleport
==========

A proof-of-concept serialization, transmission and restoring python (stack) states.

How it works
------------

~~Given the code quality inside the repo, it is pure magic.~~

TBD

Example
-------

Say, you have a nested function call.

```python
def a():
    def b():
        def c():
            result = "hello"
            # teleport to another machine
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())

assert a() == 87.5
```

You would like to pause the execution somewhere inside `c()`, transmit the state and
resume your python process elsewhere. This is how you do it.

```python
from flow_control import dummy_teleport

def a():
    def b():
        def c():
            result = "hello"
            dummy_teleport()
            # execution will be paused here
            # dummy_teleport() will save the state of the execution,
            # create another python process and resume the code there
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())

assert a() == 87.5
```

The output of this snippet is exactly the same as before, but the first part of it
(before `dummy_teleport`) was running within the main process while the second part
of the code was running within a *subprocess*.

Known limitations
-----------------

This is a proof of concept.
It currently works only within specific conditions and with specific cPython versions.

This does not work with:
- non-python stacks (i.e. when native code invokes python);
- active generators in stack;
- for, try, if, and all other subframes;

More information to be added.

History
-------

8 July 2021 21:32 CEST a python state was first teleported into another process on the same machine in Amsterdam.

License
-------

[LICENSE.md](LICENSE.md)

