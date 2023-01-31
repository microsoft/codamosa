#  This file is part of CodaMOSA
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#
"""Visits a module containing a test function and adds a placeholder for mutation."""

import ast
import logging
from typing import Optional, Tuple

import pynguin.utils.randomness as randomness
from pynguin.languagemodels.outputfixers import fixup_imports

logger = logging.getLogger()


class _RandomMutationVisitor(ast.NodeVisitor):
    """
    A node visitor that randomly chooses a test function to mutate, and mutates it with
    a placeholder expression. It chooses a random call expression to mutate, and either
    replaces the whole call with a placeholder expression, or one of its arguments with
    a placeholder expression. Or it mutates the RHS of an assignment statement.
    """

    def __init__(self) -> None:
        super().__init__()
        self._placeholder_elem: ast.Name = ast.Name(id="??")

    def visit_Module(self, node: ast.Module) -> Optional[Tuple[str, str]]:
        """Visits a module and adds a placeholder element to one of the
        test functions defined in the module.

        Args:
            node: the module to visit

        Returns:
            a tuple containing (1) the name of the function that was altered and
            (2) the altered function as an ast node; or None, if no suitable
            test function was found

        """
        fn_defs = [
            stmt
            for stmt in node.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_")
        ]
        if len(fn_defs) == 0:
            print("No test functions found in the module.")
            return None
        fn_def_to_mutate = randomness.choice(fn_defs)
        return self.visit(fn_def_to_mutate)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Tuple[str, str]:
        """Visits a single function and adds a placeholder element to be
        filled, either:
           (1) replacing an entire function call
           (2) replacing a string literal
           (3) adding a placeholder to the end of the function.

        Args:
            node: the function to add a placeholder to

        Returns:
            a tuple containing (1) the name of the function that was altered and
            (2) the altered function as string.
        """

        mutateable_assigns_idx = [
            i
            for i, stmt in enumerate(node.body)
            if isinstance(stmt, ast.Assign)
            and (
                isinstance(stmt.value, ast.Call)
                or (
                    isinstance(stmt.value, ast.Constant)
                    and (isinstance(stmt.value.value, str) or stmt.value is None)
                )
            )
        ]
        if mutateable_assigns_idx == [] or randomness.next_float() < 0.3:
            new_body = node.body + [ast.Expr(self._placeholder_elem)]
        else:
            stmt_idx_to_mutate = randomness.choice(mutateable_assigns_idx)
            stmt_to_mutate: ast.Assign = node.body[stmt_idx_to_mutate]  # type: ignore
            new_stmt = ast.Assign(
                targets=stmt_to_mutate.targets, value=self._placeholder_elem
            )
            new_body = (
                node.body[:stmt_idx_to_mutate]
                + [new_stmt]
                + node.body[stmt_idx_to_mutate + 1 :]
            )

        new_function = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=new_body,
            decorator_list=node.decorator_list,
            returns=node.returns,
        )
        new_function = ast.fix_missing_locations(new_function)

        return node.name, ast.unparse(new_function)


def add_placeholder(node: ast.Module) -> str:
    """Adds the placeholder ?? somewhere to a test function in `node`, and replaces
    the pynguin module_x. qualifications with the natural names, removing the
    qualfications entirely for functions from the module under test.

    Args:
        node: parsed module containing a test function to mutate

    Returns:
        the test function with placeholder added an imports fixed up, as string.
    """

    # First, add the placeholder
    visitor = _RandomMutationVisitor()
    if (res := visitor.visit(node)) is None:
        logger.error("Could not find any functions to mutate.")
        return ast.unparse(node)
    function_name_mutated, mutated = res

    # Then, fixup the imports
    mutated = fixup_imports(mutated, node)

    return mutated
