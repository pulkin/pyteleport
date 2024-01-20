import dis
import random
from pathlib import Path
from random import choice
from dataclasses import dataclass

import pytest

from ...tests.test_scripts import test_cases
from ..minias import disassemble


def _test_back_forth(source_code: str):
    code_obj = compile(source_code, "", "exec")
    my_code = disassemble(code_obj, keep_nop=True).assemble(
        consts=code_obj.co_consts,  # ensures const/name order is preserved
        names=code_obj.co_names,  # same
        varnames=code_obj.co_varnames,  # same
    )
    # ensure order is preserved
    assert tuple(my_code.names) == code_obj.co_names
    assert tuple(my_code.varnames) == code_obj.co_varnames

    ref = code_obj.co_code
    test = bytes(my_code)

    def _present(_i):
        return tuple(enumerate(zip((
            dis.opname[_j]
            for _j in _i[::2]
        ), _i[1::2])))

    assert _present(test) == _present(ref)


@pytest.mark.parametrize("code", [
    "a = 'hello'",
    "a = b",
    "if a: pass",
    "for i in something: 2 * i",
    "def f(): pass",
    "a, b = 3, 4",
    "try: something()\nexcept Exception as e: caught(e)\nelse: otherwise()\nfinally: finalize()",
    "class A(B): pass",
])
def test_oneliners(code: str):
    _test_back_forth(code)


@pytest.mark.parametrize("name", test_cases)
def test_script(name):
    with open(Path(__file__).parent.parent.parent / "tests" / name, "r") as f:
        _test_back_forth(f.read())


@pytest.mark.parametrize("size", [1, 10, 100, 200])
def test_random_branching(size: int):
    @dataclass
    class Tree:
        children: list["Tree"] = None

        def __post_init__(self):
            if self.children is None:
                self.children = []

        def lines(self):
            yield "if something:"
            for c in self.children:
                for l in c.lines():
                    yield "  " + l
            if len(self.children) == 0:
                yield "  pass"

    random.seed(0)
    all_nodes = [Tree()]
    for _ in range(size - 1):
        new = Tree()
        choice(all_nodes).children.append(new)
        all_nodes.append(new)

    source_code = "\n".join(all_nodes[0].lines())
    _test_back_forth(source_code)
