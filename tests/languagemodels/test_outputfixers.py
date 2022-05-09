#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

import pytest

from pynguin.languagemodels.outputfixers import rewrite_tests

src_1 = """
def test_Try_map():
    Try.of(lambda x: x + 1, 1).map(lambda x: x * 2) == Try(3, True)
"""
src_1_res = """def test_Try_map():
    var_0 = lambda x: x + 1
    var_1 = 1
    var_2 = Try.of(var_0, var_1)
    var_3 = lambda x: x * 2
    var_4 = var_2.map(var_3)
    var_5 = 3
    var_6 = True
    var_7 = Try(var_5, var_6)
    var_8 = var_4 == var_7
"""

src_2 = """
def test_Try_on_success():
    def success_callback(x):
        print(x)

    Try.of(lambda x: x + 1, 1).on_success(success_callback)
"""
src_2_res = """def test_Try_on_success():

    def success_callback(x):
        print(x)
    var_0 = lambda x: x + 1
    var_1 = 1
    var_2 = Try.of(var_0, var_1)
    var_3 = var_2.on_success(success_callback)
"""

src_3 = """
def test_Queue_full():
    q = Queue(10)
    assert q.full() == False
    q.enqueue(1)
    assert q.full() == False
    q.enqueue(2)
    assert q.full() == True
"""

src_3_res = """def test_Queue_full():
    var_0 = 10
    q = Queue(var_0)
    var_1 = q.full()
    assert var_1 == False
    var_2 = 1
    var_3 = q.enqueue(var_2)
    var_4 = q.full()
    assert var_4 == False
    var_5 = 2
    var_6 = q.enqueue(var_5)
    var_7 = q.full()
    assert var_7 == True
"""

src_4 = """
def test_HTTPieHTTPSAdapter_init_poolmanager():
    adapter = HTTPieHTTPSAdapter(verify=True)
    assert adapter.pool_manager.connection_pool_kw['ssl_context'] is not None
"""

src_4_res = """def test_HTTPieHTTPSAdapter_init_poolmanager():
    var_0 = True
    adapter = HTTPieHTTPSAdapter(verify=var_0)
    assert adapter.pool_manager.connection_pool_kw['ssl_context'] is not None
"""

src_5 = """
def test_HTTPieCertificate_to_raw_cert():
    cert = HTTPieCertificate(cert_file='cert', key_file='key')
    assert cert.to_raw_cert() == ('cert', 'key')
"""

src_5_res = """def test_HTTPieCertificate_to_raw_cert():
    var_0 = 'cert'
    var_1 = 'key'
    cert = HTTPieCertificate(cert_file=var_0, key_file=var_1)
    var_2 = cert.to_raw_cert()
    assert var_2 == ('cert', 'key')
"""

src_6 = """
def test__KVSection_parse():
    section = _KVSection("Parameters", "parameters")
    text = \"\"\"
    a : int
        a parameter
    b : int
        another parameter
    \"\"\"
    expected = [
        DocstringMeta(
            args=["a", "b"],
            description="a parameter\\n\\nanother parameter",
        )
    ]
    assert list(section.parse(text)) == expected
"""

src_6_res = (
    "def test__KVSection_parse():\n"
    "    var_0 = 'Parameters'\n"
    "    var_1 = 'parameters'\n"
    "    section = _KVSection(var_0, var_1)\n"
    "    text = '\\n    a : int\\n        a parameter\\n    b : int\\n        "
    "another parameter\\n    '\n"
    "    var_2 = 'a'\n"
    "    var_3 = 'b'\n"
    "    var_4 = [var_2, var_3]\n"
    "    var_5 = 'a parameter\\n\\nanother parameter'\n"
    "    var_6 = DocstringMeta(args=var_4, description=var_5)\n"
    "    expected = [var_6]\n"
    "    var_7 = section.parse(text)\n"
    "    var_8 = list(var_7)\n"
    "    assert var_8 == expected\n"
)

src_7 = """
# Unit test for method add_section of class NumpydocParser
def test_NumpydocParser_add_section():
    \"\"\"Unit test for method add_section of class NumpydocParser\"\"\"
    ndp = module_0.NumpydocParser()
    ndp.add_section(module_0.Section("Example", "examples"))
    assert ndp.sections["Example"] == module_0.Section("Example", "examples")
"""

src_7_res = """def test_NumpydocParser_add_section():
    var_0 = 'Unit test for method add_section of class NumpydocParser'
    ndp = module_0.NumpydocParser()
    var_1 = 'Example'
    var_2 = 'examples'
    var_3 = module_0.Section(var_1, var_2)
    var_4 = ndp.add_section(var_3)
    var_5 = module_0.Section(var_1, var_2)
    assert ndp.sections['Example'] == var_5
"""
src_8 = """
def test_is_pretty_print_string():
    assert is_pretty_print_string(get_wrapped_value(1), "I am the number with value 1")
"""

src_8_res = """def test_is_pretty_print_string():
    var_0 = 1
    var_1 = get_wrapped_value(var_0)
    var_2 = 'I am the number with value 1'
    var_3 = is_pretty_print_string(var_1, var_2)
    assert var_3
"""

src_9 = """def test_YieldFromTransformer_visit():
    from ..utils.snippet import load_ast
    from ..utils.unittest_base import BaseUnitTest
    from ..transformers.yield_from_transformer import YieldFromTransformer

    class TestYieldFromTransformer(BaseUnitTest):
        def test_yield_from_transformer(self):
            ast_tree = load_ast('yield_from_transformer.py')
            YieldFromTransformer().visit(ast_tree)
            self.assertTrue(ast_tree is not None)

    TestYieldFromTransformer().test_yield_from_transformer()
"""

src_9_res = """def test_YieldFromTransformer_visit():
    from ..utils.snippet import load_ast
    from ..utils.unittest_base import BaseUnitTest
    from ..transformers.yield_from_transformer import YieldFromTransformer

    class TestYieldFromTransformer(BaseUnitTest):

        def test_yield_from_transformer(self):
            var_0 = 'yield_from_transformer.py'
            ast_tree = load_ast(var_0)
            var_1 = YieldFromTransformer()
            var_2 = var_1.visit(ast_tree)
            var_3 = None
            var_4 = ast_tree is not var_3
            var_5 = self.assertTrue(var_4)
    var_0 = TestYieldFromTransformer()
    var_1 = var_0.test_yield_from_transformer()
"""

src_10 = """def test_yield_from():
    def gen():
        yield 1
        yield 2
        yield 3
    def func():
        yield from gen()
    assert list(func()) == [1, 2, 3]"""

src_10_res = """def test_yield_from():

    def gen():
        yield 1
        yield 2
        yield 3

    def func():
        yield from gen()
    var_0 = func()
    var_1 = list(var_0)
    assert var_1 == [1, 2, 3]
"""

src_11 = """def test_foo():
    (x := 1)
    y : int
    y : int = 3
    y += x
    x -= y
"""
src_11_res = """def test_foo():
    x = 1
    y = 3
    y = y + x
    x = x - y
"""

src_12 = """def test_foo():
    y = (x := 1)
"""
src_12_res = """def test_foo():
    x = 1
    y = x
"""

src_13 = """def test_Uniqueness():
    u = Uniqueness([1, 2, 3, 2, 1, 0, -1, 1, 2, -1, 2, 0, False, True])
    assert u._set == {0, 1, 2, 3, -1, False, True}
"""

src_13_res = """def test_Uniqueness():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 0
    var_4 = -1
    var_5 = -1
    var_6 = False
    var_7 = True
    var_8 = [var_0, var_1, var_2, var_1, var_0, var_3, var_4, var_0, var_1, var_5, var_1, var_3, var_6, var_7]
    u = Uniqueness(var_8)
    assert u._set == {0, 1, 2, 3, -1, False, True}
"""


@pytest.mark.parametrize(
    "original_src,result_src",
    [
        (src_1, src_1_res),
        (src_2, src_2_res),
        (src_3, src_3_res),
        (src_4, src_4_res),
        (src_5, src_5_res),
        (src_6, src_6_res),
        (src_7, src_7_res),
        (src_8, src_8_res),
        (src_9, src_9_res),
        (src_10, src_10_res),
        (src_11, src_11_res),
        (src_12, src_12_res),
        (src_13, src_13_res),
    ],
)
def test_rewrite_tests(original_src: str, result_src: str):
    result = rewrite_tests(original_src)
    assert result == result_src, (
        f"Incorrect rewriting. Expected:\n{result_src}" f"\ngot: \n{result}"
    )
