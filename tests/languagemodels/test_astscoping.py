#  This file is part of CodaMOSA.
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#

import ast

import pytest

from pynguin.languagemodels.astscoping import operate_on_free_variables

all_free = "var_0 + var_1"
all_free_res = "MOO + MOO"

nested_lambda = "lambda x: lambda y: x + y"
nested_lambda_res = "lambda x: lambda y: x + y"

list_comprehension = "[i + 1 for i in var_0]"
list_comprehension_res = "[i + 1 for i in MOO]"

nested_list_comprehension = "[obj for objs in var_0 for obj in objs]"
nested_list_comprehension_res = "[obj for objs in MOO for obj in objs]"

lambda_in_list_comprehension = "[lambda x: x + i for i in var_0]"
lambda_in_list_comprehension_res = "[lambda x: x + i for i in MOO]"

set_comprehension = "{i + var_1 for i in var_0}"
set_comprehension_res = "{i + MOO for i in MOO}"

dict_comprehension = "{i: i + var_1 for i in var_0}"
dict_comprehension_res = "{i: i + MOO for i in MOO}"


fn_call_on_bound = "{foo(i) for i in var_0}"
fn_call_on_bound_res = "{foo(i) for i in MOO}"

free_var_name_collision = "[x for x in var_0] + x"
free_var_name_collision_res = "[x for x in MOO] + MOO"

setcomp_name_collision = "{x for x in var_0} + x"
setcomp_name_collision_res = "{x for x in MOO} + MOO"

dictcomp_name_collision = "{x: x for x in var_0} + x"
dictcomp_name_collision_res = "{x: x for x in MOO} + MOO"

lambda_name_collision = "(lambda x: x) + x"
lambda_name_collision_res = "(lambda x: x) + MOO"

# Why would anyone ever do this?
nested_list_name_collision = "[xs for x in xs for xs in x]"
nested_list_name_collision_res = "[xs for x in MOO for xs in x]"


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
        (fn_call_on_bound, fn_call_on_bound_res),
        (free_var_name_collision, free_var_name_collision_res),
        (setcomp_name_collision, setcomp_name_collision_res),
        (dictcomp_name_collision, dictcomp_name_collision_res),
        (lambda_name_collision, lambda_name_collision_res),
        (nested_list_name_collision, nested_list_name_collision_res),
    ],
)
def test_moo_operate(input_str, output_str):
    mod = ast.parse(input_str)
    assert mod is not None
    expr = mod.body[0]
    assert hasattr(expr, "value")
    expr = expr.value
    result_ast = operate_on_free_variables(expr, lambda x: ast.Name("MOO"))
    result_str = ast.unparse(result_ast)
    assert result_str == output_str
