pyteleport
==========

A proof-of-concept serialization and restoring python (stack) states.

Example
-------

Say, you have a nested function call.

```python
def a():
    def b():
        def c():
            result = "hello"
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())

assert a() == 87.5
```

One would like to stop the execution somewhere inside `c()` and to resume it afterwards (somewhat similar to `yield` statement).
Here is how you achieve it.

```python
from flow_control import Serializer

def a():
    def b():
        def c():
            result = "hello"
            Serializer().inject(None, to=-1)
            # execution will be halted here
            # inject() will save the state of the execution
            # and will return the state in the outermost frame
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())

serializer = a()  # instead of the actual result, returns the serialized state
morph_a = serializer.compose_morph()  # morph_a() resembles the function a()
# unlike a(), morph_a() will immediately rush into the saved stack state
# and will resume from where Serializer.inject was invoked
assert morph_a() == 87.5
```

Known limitations
-----------------

This is a proof of concept.
It currently works only within specific conditions under `cPython 3.9`.
More information to be added.

License
-------

[LICENSE.md](LICENSE.md)

