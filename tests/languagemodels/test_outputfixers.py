#  This file is part of CodaMOSA.
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#

import pytest

from pynguin.languagemodels.outputfixers import rewrite_tests

src_1 = """
def test_Try_map():
    Try.of(lambda x: x + 1, 1).map(lambda x: x * 2) == Try(3, True)
"""
src_1_res = """def test_Try_map():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = Try.of(var_1, var_0)
    var_3 = 2
    var_4 = lambda x: x * var_3
    var_5 = var_2.map(var_4)
    var_6 = 3
    var_7 = True
    var_8 = Try(var_6, var_7)
    var_9 = var_5 == var_8
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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = Try.of(var_1, var_0)
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

src_14 = """def test_foo():
    int_0 = 0
    int_1 = 1
    list_0 = [var_0, var_1]
    dict_0 = {var_0: var_1, var_1: var_0}
    var_0 = foo(*list_0, **dict_0)
"""

src_14_res = """def test_foo():
    int_0 = 0
    int_1 = 1
    list_0 = [var_0, var_1]
    dict_0 = {var_0: var_1, var_1: var_0}
    var_0 = foo(*list_0, **dict_0)
"""

src_15 = """def test_module_qualification():
    ast_0 = typed_ast._ast3.parse("stuff")
    chained_result = ast_0.fix_missing_locations().more_things()
"""

src_15_res = """def test_module_qualification():
    var_0 = 'stuff'
    ast_0 = typed_ast._ast3.parse(var_0)
    var_1 = ast_0.fix_missing_locations()
    chained_result = var_1.more_things()
"""

src_16 = """def test_foo():
    var_1 = foo(1, 3)
"""
src_16_res = """def test_foo():
    var_0 = 1
    var_2 = 3
    var_1 = foo(var_0, var_2)
"""

src_17 = """def test_foo():
    x = foo.lst[4]
    lst_0 = [4, 4, 4]
    y = lst_0[0:1:1]
    z = lst_0[0:1, 1]
"""

src_17_res = """def test_foo():
    var_0 = 4
    x = foo.lst[var_0]
    lst_0 = [var_0, var_0, var_0]
    var_1 = 0
    var_2 = 1
    y = lst_0[var_1:var_2:var_2]
    z = lst_0[var_1:var_2, var_2]
"""


src_18 = """def test_foo():
    (x, y) = (0, 1)
"""

src_18_res = """def test_foo():
    var_0 = 0
    var_1 = 1
    (x, y) = (var_0, var_1)
"""


src_19 = """def test_nested():
    x = foo(1)
    def test_nestee(z):
        y = bar(3, z)
"""

src_19_res = """def test_nested():
    var_0 = 1
    x = foo(var_0)

    def test_nestee(z):
        var_0 = 3
        y = bar(var_0, z)
"""

src_20 = """def test_stuff():

    class MyTest:
        x : int = 3

        def test_nestee(self, z):
            y = bar(3, z)
"""

src_20_res = """def test_stuff():

    class MyTest:
        x: int = 3

        def test_nestee(self, z):
            var_0 = 3
            y = bar(var_0, z)
"""

src_21 = """def test_stuff():

    class MyTest:
        x: int = 3
"""

src_21_res = """def test_stuff():

    class MyTest:
        x: int = 3
"""


src_22 = """def test_foo():
    y = (x := 1)

def test_bar():
    y = (x := 1)

pytest.main()
"""
src_22_res = """def test_foo():
    x = 1
    y = x

def test_bar():
    x = 1
    y = x
"""

src_23 = """def test_foo():
   y = baz(0)
   z = bar(\"\"\"my test starts here
"""

src_23_res = """def test_foo():
    var_0 = 0
    y = baz(var_0)
"""

# Everything we don't support!!!
src_24 = """def test_foo():
    import boo
    from baz import booboo
    await foo(3)

    async def main():
        booboo(5, 8)
    async for i in range(3):
        booboo(5, 8)
    async with open('bax') as f:
        foo(f)
"""

src_25 = """def test_foo():
    j = 5
    for i in range(3,5):
        y = foo(1)
    while j > 3:
        j = j - 1
    try:
        my_stuff = foo(1)
    except:
        pass
    if j == 3:
       foo(1)
    elif j == 2:
       foo(1)
    else:
       foo(1)
    with open('file') as f:
        x = foo(f)
"""

src_25_res = """def test_foo():
    j = 5
    for i in range(3, 5):
        var_0 = 1
        y = foo(var_0)
    while j > 3:
        var_0 = 1
        j = j - var_0
    try:
        var_0 = 1
        my_stuff = foo(var_0)
    except:
        pass
    if j == 3:
        var_0 = 1
        var_1 = foo(var_0)
    elif j == 2:
        var_0 = 1
        var_1 = foo(var_0)
    else:
        var_0 = 1
        var_1 = foo(var_0)
    with open('file') as f:
        x = foo(f)
"""

src_26 = """def test_foo():
    x = -1
    z = -(1 + 3)
"""

src_26_res = """def test_foo():
    x = -1
    var_0 = 1
    var_1 = 3
    var_2 = var_0 + var_1
    z = -var_2
"""

# Note the semantics change slightly in case maybe_side_effect
# has side-effects. Oh well.
src_27 = """def test_bound_vars():
    lst_0 = [obj for objs in foo.bar() for obj in objs]
    set_0 = {i + 3 for i in lst_0}
    dict_0 = {i: i + maybe_side_effect() for i in lst_0}
    lambda_0 = lambda x: x + 3
    lst_1 = [lambda_0(i) for i in lst_0]
    lst_2 = [foo.baz(i, 3) for i in lst_0]
    lambda_1 = lambda *args, **varargs: args + varargs
"""
src_27_res = """def test_bound_vars():
    var_0 = foo.bar()
    lst_0 = [obj for objs in var_0 for obj in objs]
    var_1 = 3
    set_0 = {i + var_1 for i in lst_0}
    var_2 = maybe_side_effect()
    dict_0 = {i: i + var_2 for i in lst_0}
    lambda_0 = lambda x: x + var_1
    lst_1 = [lambda_0(i) for i in lst_0]
    lst_2 = [foo.baz(i, var_1) for i in lst_0]
    lambda_1 = lambda *args, **varargs: args + varargs
"""

src_28 = """def test_nested_attrs():
    x = bar().baz(a)
    y = my_thing[4].list.boo(x)
    a = fs.bar[4].foo()
"""

src_28_res = """def test_nested_attrs():
    var_0 = bar()
    x = var_0.baz(a)
    var_1 = 4
    var_2 = my_thing[var_1]
    y = var_2.list.boo(x)
    var_3 = fs.bar[var_1]
    a = var_3.foo()
"""

src_29 = """def test_choice():
    Choice()(items=['a', 'b', 'c'])
"""

src_29_res = """def test_choice():
    var_0 = Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0(items=var_4)
"""

src_30 = """def test_generator_expr():
    lst_0 = [i for i in range(10)]
    has_gt_3 = any(i > 3 for i in lst_0)
"""

src_30_res = """def test_generator_expr():
    var_0 = 10
    var_1 = range(var_0)
    lst_0 = [i for i in var_1]
    var_2 = 3
    var_3 = (i > var_2 for i in lst_0)
    has_gt_3 = any(var_3)
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
        (src_14, src_14_res),
        (src_15, src_15_res),
        (src_16, src_16_res),
        (src_17, src_17_res),
        (src_18, src_18_res),
        (src_19, src_19_res),
        (src_20, src_20_res),
        (src_21, src_21_res),
        (src_22, src_22_res),
        (src_23, src_23_res),
        (src_24, src_24),
        (src_25, src_25_res),
        (src_26, src_26_res),
        (src_27, src_27_res),
        (src_28, src_28_res),
        (src_29, src_29_res),
        (src_30, src_30_res),
    ],
)
def test_rewrite_tests(original_src: str, result_src: str):
    result_dict = rewrite_tests(original_src)
    result = "\n".join(list(result_dict.values()))
    assert result == result_src, (
        f"Incorrect rewriting. Expected:\n{result_src}" f"\ngot: \n{result}"
    )
