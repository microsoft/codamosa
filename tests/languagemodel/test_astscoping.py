#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

import pytest

import ast
from pynguin.languagemodels.astscoping import operate_on_free_variables


all_free = "var_0 + var_1"
all_free_res = "MOO + MOO"

nested_lambda = "lambda x: lambda y: x + y"
nested_lambda_res = "lambda x: lambda y: x + y"

list_comprehension = '[i + 1 for i in var_0]'
list_comprehension_res = '[i + 1 for i in MOO]'

nested_list_comprehension = '[obj for objs in var_0 for obj in objs]'
nested_list_comprehension_res = '[obj for objs in var_0 for obj in objs]'

lambda_in_list_comprehension = '[lambda x: x + i for i in var_0]'
lambda_in_list_comprehension_res = '[lambda x: x + i for i in MOO]'

set_comprehension = '{i + var_1 for i in var_0}'
set_comprehension_res = '{i + MOO for i in MOO}'

dict_comprehension = '{i: i + var_1 for i in var_0}'
dict_comprehension_res = '{i: i + MOO for i in MOO}'


fn_call_on_bound = '{foo(i) for i in var_0}'
fn_call_on_bound_res = '{foo(i) for i in MOO}'


@pytest.mark.parametrize(
    "input_str,output_str",
    [
        (all_free, all_free_res),
        (nested_lambda, nested_lambda_res),
        (list_comprehension, list_comprehension_res),
        (nested_list_comprehension, nested_list_comprehension_res),
        (lambda_in_list_comprehension, lambda_in_list_comprehension_res),
        (set_comprehension, set_comprehension_res),
        (dict_comprehension, dict_comprehension_res),
        (fn_call_on_bound, fn_call_on_bound_res)
    ],
)
def test_moo_operate(input_str, output_str):
    mod = ast.parse(input_str)
    assert mod is not None
    expr = mod.body[0]
    assert hasattr(expr, 'value')
    expr = expr.value
    result_ast = operate_on_free_variables(expr, lambda x: ast.Name('MOO'))
    result_str = ast.unparse(result_ast)
    assert result_str == output_str
