![build](https://github.com/pulkin/pyteleport/actions/workflows/test.yml/badge.svg)

pyteleport
==========

A proof-of-concept serialization, transmission and restoring python runtime.

How it works
------------

* You invoke `teleport` in your python script.
* `pyteleport` collects the runtime state: globals, locals, stack.
* `pyteleport` dumps the runtime into a specially designed "morph" bytecode
  which resumes from where `teleport` was invoked.
* The bytecode is transmitted to the target environment and passed to a
  python interpreter there.
* The local python runtime is terminated and simply pipes stdio from the
  target environment.

Example
-------

```python
from flow_control import bash_teleport
from socket import gethostname
from os import getpid

def log(*args):
    print(f"[{gethostname()}/{getpid()}]", *args)

log("hi")
bash_teleport("ssh", "cartesius",
    "conda activate py39;",
    other_fn=("mem_view.py", "flow_control.py"))
log("bye")
```

outputs

```
[stealth/4258] hi
[int1.bullx/17980] bye
```

Note that the two outputs were produced by different processes on different machines! This is what
`bash_teleport` does: it transmits the runtime from one `python` process to another.

Also works from within a stack:

```python
def a():
    def b():
        def c():
            result = "hello"
            bash_teleport("ssh", "cartesius",
                "conda activate py39;",
                other_fn=("mem_view.py", "flow_control.py"))
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())
assert a() == 87.5
```

Known limitations
-----------------

This is a proof of concept.
It currently works only within specific conditions and with specific cPython versions.

This does not work with:
- non-python stacks (i.e. when native code invokes python);
- active generators in stack;
- for, try, if, with, and all other subframes;

More information to be added.

History
-------

* 8 July 2021 21:32 CEST a python runtime was first teleported to another process on the same machine
* 11 July 2021 20:46 CEST a python runtime was first teleported to another machine

License
-------

[LICENSE.md](LICENSE.md)

