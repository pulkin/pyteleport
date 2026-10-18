[![build](https://github.com/pulkin/pyteleport/actions/workflows/test.yml/badge.svg)](https://github.com/pulkin/pyteleport/actions)
[![pypi](https://img.shields.io/pypi/v/pyteleport)](https://pypi.org/project/pyteleport/)

# ![icon](resources/icon-full.svg)

A proof-of-concept serialization, transmission and restoring python runtime.

About
-----

`pyteleport` is capable of making snapshots of python runtime from
(almost) arbitrary state, including locals, globals and stack.
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

![A python REPL cast showcasing the teleport](resources/cast0.gif)

Teleport inside REPL:

```python
>>> result = "something important"
>>> from pyteleport import tp_bash
>>> tp_bash("ssh", "remote")
>>> print(result)  # REPL is now running on a remote machine with all data available
```

Also works from anywhere in the stack:

```python
def a():
    def b():
        def c():
            result = "hello"              # runs locally
            tp_bash(...)                  # teleport here
            return result + " world"      # runs remotely
        return len(c()) + float("3.5")    # c() runs locally, then remotely; the rest runs remotely
    return 5 * (3 + b())                  # b() runs locally, then remotely; sum and product run remotely
assert a() == 87.5                        # a() runs locally, then remotely; == and assert run remotely
```

API
---

### `pyteleport.tp_bash(*args)`, `pyteleport.tp_shell(*args)`

A PoC transfer of python runtime state: just specify how to reach the remote
shell and `pyteleport` will do the rest for you.

### `pyteleport.dump("filename.pyc")`

Save the current runtime state as a `snapshot.pyc` file.
You may then resume or replay your python program from the snapshot state
by simply running `python snapshot.pyc`.

How it works
------------

* You invoke `teleport` in your python script.
* `pyteleport` collects the runtime state: globals, locals, stack.
* `pyteleport` dumps the runtime into a specially designed "morph" bytecode.
* The bytecode is transmitted to the target environment and passed to a
  python interpreter there.
* The remote python runs the bytecode which (1) develops the runtime state
  into the snapshot and (2) resumes running.
* The local python runtime pipes the stdio of the remote process or resumes
  independently (fork mode).

Known limitations
-----------------

This is a proof of concept.
The package works with cPython 3.9-3.14.

Current limitations:

- [ ] no thread support
- [ ] no `async` support (needs further investigation regarding non-python stack)
- [ ] more generally, no native code support
- [ ] back-teleport and nested teleport (never tried)

License
-------

[LICENSE.md](LICENSE.md)

Useful information
------------------

- [Experimenting with AWS](doc/aws.md)
