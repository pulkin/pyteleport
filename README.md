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

API
---

TBD, see docstrings for the moment

How it works
------------

* You invoke `teleport` in your python script.
* `pyteleport` collects the runtime state: globals, locals, stack.
* `pyteleport` dumps the runtime into a specially designed "morph" bytecode
  which resumes from a state resembling current runtime state.
* The bytecode is transmitted to the target environment and passed to a
  python interpreter there.
* The remote python runs the bytecode which restores the runtime state.
  The python program casually resumes from where it was interrupted.
* The local python runtime is terminated and simply pipes stdio from the
  target environment.

Known limitations
-----------------

This is a proof of concept.
The package works with cPython ~~v3.8~~, 3.9, or 3.10.

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
