import subprocess
import sys

import pytest


@pytest.mark.parametrize('first', ['function', 'dictionary', 'module'])
def test_main_namespace_round_trip(first):
    sender = '''
import pyteleport
import dill
import sys

shared = []
shared.append(shared)
alias = shared
namespace = globals()
main = sys.modules["__main__"]

def read_state():
    return shared, namespace, main

objects = {"function": read_state, "dictionary": namespace, "module": main}
sys.stdout.buffer.write(dill.dumps((objects[sys.argv[1]], read_state, namespace, main, shared)))
'''
    receiver = '''
import dill
import sys

first, restored_function, namespace, main, shared = dill.loads(sys.stdin.buffer.read())
assert main is sys.modules["__main__"]
assert namespace is main.__dict__ is restored_function.__globals__
assert first is {"function": restored_function, "dictionary": namespace, "module": main}[sys.argv[1]]
assert namespace["read_state"] is restored_function
assert namespace["shared"] is namespace["alias"] is shared
assert namespace["namespace"] is namespace
assert namespace["main"] is main
assert shared[0] is shared
assert restored_function() == (shared, namespace, main)
'''
    data = subprocess.check_output([sys.executable, '-c', sender, first])
    subprocess.run([sys.executable, '-c', receiver, first], input=data, check=True)


@pytest.mark.parametrize('dictionary_first', [False, True])
def test_patched_module_round_trip(dictionary_first):
    # Separate interpreters ensure imports cannot hide lost module state.
    sender = '''
import pyteleport
import dill
import pathlib
import sys

shared = []
shared.append(shared)
pathlib.secret = shared
pathlib.alias = shared
pathlib.namespace = pathlib.__dict__
pathlib.module = pathlib
pathlib.Path = lambda: "patched"
sys.teleport_secret = "builtin patch"
payload = (pathlib.__dict__, pathlib) if sys.argv[1] == "True" else (pathlib, pathlib.__dict__)
sys.stdout.buffer.write(dill.dumps((payload, sys, shared)))
'''
    receiver = '''
import dill
import pathlib
import sys

payload, restored_sys, shared = dill.loads(sys.stdin.buffer.read())
module, namespace = payload[::-1] if sys.argv[1] == "True" else payload
assert module is pathlib
assert namespace is pathlib.__dict__
assert namespace["secret"] is pathlib.secret is pathlib.alias is shared
assert pathlib.namespace is namespace
assert pathlib.module is module
assert pathlib.secret[0] is pathlib.secret
assert pathlib.Path() == "patched"
assert restored_sys is sys
assert sys.teleport_secret == "builtin patch"
'''
    data = subprocess.check_output([sys.executable, '-c', sender, str(dictionary_first)])
    subprocess.run([sys.executable, '-c', receiver, str(dictionary_first)], input=data, check=True)
