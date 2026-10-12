import subprocess
import sys

import pytest


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
