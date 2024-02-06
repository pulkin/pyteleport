import subprocess
from subprocess import check_output, Popen, PIPE
import sys
from pathlib import Path
import ast
import re

import pytest

from ..bytecode.opcodes import python_feature_block_stack


def run_test(name, interactive=False, dry_run=False, timeout=2):
    if interactive:
        if dry_run:
            raise NotImplementedError(f"{dry_run=} not implemented for {interactive=}")
        with open(name, 'r') as fl:
            process = Popen([sys.executable], stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf-8')
            stdout, stderr = process.communicate(fl.read(), timeout=2)
            print(stderr, file=sys.stderr, flush=True)
            assert process.returncode == 0
            return stdout
    else:
        return check_output([sys.executable, name, f"{dry_run=}"], stderr=PIPE, text=True, env={"PYTHONPATH": "."}, timeout=timeout)


test_cases = list(map(lambda x: x.name, Path(__file__).parent.glob("_test_*.py")))


@pytest.mark.parametrize("test", test_cases)
@pytest.mark.parametrize("interactive,dry_run", [(False, False), (True, False), (False, True)])
def test_external(test, interactive, dry_run):
    if test == "_test_teleport_multi.py" and dry_run is True:
        pytest.skip(f"_test_teleport_multi has no dummy test")
    if test == "_test_teleport_ec2.py":
        pytest.skip(f"_test_teleport_ec2 requires ec2 setup")
    if test == "_test_teleport_ssh.py":
        pytest.skip(f"_test_teleport_ssh.py needs an ssh setup")
    test = Path(__file__).parent / test

    with open(test, 'r') as f:
        module_text = f.read()
        module = ast.parse(module_text)
        docstring = ast.get_docstring(module).format(dry_run=dry_run)
        if not python_feature_block_stack:
            docstring = re.sub(r"^\[(True|False)] bstack .*$", r"[\1] bstack --", docstring, flags=re.MULTILINE)

    try:
        assert run_test(test, interactive=interactive, dry_run=dry_run).rstrip() == docstring
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"The remote python process exited with code {e.returncode}\n"
                           f"--- stdout ---\n{e.stdout}\n"
                           f"--- stderr ---\n{e.stderr}")
