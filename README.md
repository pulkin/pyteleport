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

```python
from pyteleport import tp_shell

def tp_shell(
    *shell_args, python="python", before="cd $(mktemp -d)",
    pyc_fn="payload.pyc", shell_delimiter="; ",
    pack_file=bash_inline_create_file,
    pack_object=dumps, unpack_object=portable_loads,
    detect_interactive=True, files=None, stack_method=None,
    _frame=None, **kwargs):
    """
    Teleport into another shell.

    Parameters
    ----------
    shell_args
        Arguments to start a new shell.
    python : str
        Python executable in the shell.
    before : str, list
        Shell commands to be run before running python.
    pyc_fn : str
        Temporary filename to save the bytecode to.
    shell_delimiter : str
        Shell delimiter to chain multiple commands.
    pack_file : Callable
        A function `f(name, contents)` turning a file
        into a shell-friendly assembly.
    pack_object : Callable, None
        A method (serializer) turning objects into bytes
        locally.
    unpack_object : Callable, None
        A method (deserializer) turning bytes into objects
        remotely. It does not have to rely on globals.
    detect_interactive : bool
        If True, attempts to detect the interactive mode
        and to open an interactive session remotely while
        piping stdio into this python process.
    files : list
        A list of files to teleport alongside.
    stack_method
        Stack collection method.
    _frame
        The frame to collect.
    kwargs
        Other arguments to `subprocess.run`.
    """
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
The package works with cPython ~~v3.8~~, 3.9, or 3.10.

What is implemented:

- [x] MWE: snapshot, serialize, transmit, restore
- [x] serialize generators
- [x] `yield from`
- [ ] threads (currently ignored)
- [x] block stack: `for`,`try`, `with`
- [ ] `async` (non-python stack; needs further investigation)
- [ ] forking to remote (possible with bytecode sstack prediction)
- [ ] back-teleport (needs API development)
- [ ] nested teleport (needs minimal changes)
- [ ] cross-fork communications (need API development)
- [x] REPL integration
- [ ] detecting non-python stack (peek into past value stack?)

Won't fix:

- non-python stack (not possible)
- cross-version (too fragile)

License
-------

[LICENSE.md](LICENSE.md)

