![build](https://github.com/pulkin/pyteleport/actions/workflows/test.yml/badge.svg)

pyteleport
==========

A proof-of-concept serialization, transmission and restoring python runtime.

Install
-------

```
pip install pyteleport
```

Example
-------

```python
from socket import gethostname
from os import getpid
def log(*args):
    print(f"[{gethostname()}/{getpid()}]", *args)

from pyteleport import bash_teleport

log("hi")
bash_teleport("ssh", "cartesius", "conda activate py39;")
log("bye")
```

output:

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
            bash_teleport(...)
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())
assert a() == 87.5
```

How it works
------------

* You invoke `teleport` in your python script.
* `pyteleport` collects the runtime state: globals, locals, stack.
* `pyteleport` dumps the runtime into a specially designed "morph" bytecode
  which resumes from where `teleport` was invoked.
* The bytecode is transmitted to the target environment and passed to a
  python interpreter there.
* The remote python runs the bytecode which restores the runtime state.
  The python program casually resumes from where it was interrupted.
* The local python runtime is terminated and simply pipes stdio from the
  target environment.

Known limitations
-----------------

This is a proof of concept.
The package assumes cPython v3.8 or 3.9.

What is working / not working:

- [x] MWE: snapshot, serialize, transmit, restore
- [x] serialize generators
- [ ] threads (currently ignored)
- [ ] `for`, `try`, `with`
  - [x] simplest `for` and `try`
  - [ ] `finally`
- [ ] `async` (never tested but likely needs minimal changes)
- [ ] `yield from` (never tested)
- [ ] non-python stack (won't fix)
- [ ] forking to remote (non-destructive teleport, needs investigating)
- [ ] back-teleport (needs API development)
- [ ] nested teleport (needs minimal changes)
- [ ] cross-fork communications (need API development)
- [ ] REPL integration (needs investigating)
- [ ] more python versions (maybe)
- [ ] cross-version (needs investigating)

The list is not final.

History
-------

* 11 July 2021 20:46 CEST a python runtime was first teleported to another machine

License
-------

[LICENSE.md](LICENSE.md)

