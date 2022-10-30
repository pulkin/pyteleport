from subprocess import check_output, Popen, PIPE
import sys
from pathlib import Path
import ast

import pytest


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


test_cases = list(map(lambda x: x.name, (Path(__file__).parent / "tests").glob("_test_*.py")))


@pytest.mark.parametrize("test", test_cases)
@pytest.mark.parametrize("interactive,dry_run", [(False, False), (True, False), (False, True)])
def test_external(test, interactive, dry_run):
    if test == "_test_teleport_multi.py" and dry_run is True:
        pytest.skip(f"_test_teleport_multi has no dummy test")
    test = Path(__file__).parent / "tests" / test

    with open(test, 'r') as f:
        module_text = f.read()
        module = ast.parse(module_text)
        docstring = ast.get_docstring(module).format(dry_run=dry_run)

    assert run_test(test, interactive=interactive, dry_run=dry_run).rstrip() == eval(f'f"""{docstring}"""')
