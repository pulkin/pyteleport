[![build](https://github.com/pulkin/pyteleport/actions/workflows/test.yml/badge.svg)](https://github.com/pulkin/pyteleport/actions)
[![pypi](https://img.shields.io/pypi/v/pyteleport)](https://pypi.org/project/pyteleport/)

# ![icon](resources/icon-full.svg)

A proof-of-concept serialization, transmission and restoring python runtime.

About
-----

`pyteleport` is capable of making snapshots of python runtime from
(almost) aribtrary state, including locals, globals and stack.
It then transforms snapshots into specially designed bytecode that
resumes execution from the snapshot state.
The bytecode can be run remotely: this way `pyteleport` teleports
python runtime.

Install
-------

```
pip install pyteleport
```

Example
-------

![term cast 0](resources/cast0.gif)

`hello` implementation:

```python
from socket import gethostname
from os import getpid
def hello():
    print(f"hello from {gethostname()} / pid {getpid()}")
```

Note that the two outputs were produced by different processes on different machines! This is what
`tp_bash` does: it transmits the runtime from one python process to another (remotely).

Also works from within a stack:

```python
def a():
    def b():
        def c():
            result = "hello"
            tp_bash(...)
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
The package works with cPython v3.8, 3.9, or 3.10.

What is implemented:

- [x] MWE: snapshot, serialize, transmit, restore
- [x] serialize generators
- [ ] threads (currently ignored)
- [x] block stack: `for`,`try`, `with`
- [ ] `async` (never tested but likely needs minimal changes)
- [ ] `yield from` (never tested)
- [ ] forking to remote (non-destructive teleport, needs investigating)
- [ ] back-teleport (needs API development)
- [ ] nested teleport (needs minimal changes)
- [ ] cross-fork communications (need API development)
- [x] REPL integration
- [ ] cross-version (needs investigating)

Won't fix:

- non-python stack (not possible)

License
-------

[LICENSE.md](LICENSE.md)

