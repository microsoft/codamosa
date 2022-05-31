#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

import ast
from typing import Any, Set, List, Dict, Callable

import pynguin.testcase.variablereference as vr


def operate_on_free_variables(node: ast.expr, operation: Callable[[ast.expr], Any]) -> ast.expr:
    """Visits `node` and applies an operation on all the free variables in `node`,
    replacing any `ast.Name` node n that is a free variable with the result of
    operation(n)

    Args:
        node: the node to visit
        operation: the operation to apply on free variables

    Returns:
        a, possibly strange, ast.expr
    """
    return node


def _replace_with_var_refs(node: ast.AST, ref_dict):
    return node

class VariableRefAST:
    """This class stores an AST, but where name nodes that belong to
    a vr.VariableReference are replaced with that reference.
    """
    def __init__(self, node: ast.AST, ref_dict: Dict[str, vr.VariableReference]):
        self._node = _replace_with_var_refs(node, ref_dict)

    def dump(self) -> str:
        """Dumps self._node to a string

        Returns:
            the dumped representation of the inner node"""
        return self._node.dump()

    def structural_equal(self, second: 'VariableRefAST', memo: dict[vr.VariableReference, vr.VariableReference]) -> bool:
        """Compares whether the two AST nodes are equal w.r.t. memo..."""
        return False

    def clone_node(self, memo: dict[vr.VariableReference, vr.VariableReference]) -> 'VariableRefAST':
        """Clone the node as an ast, doing any replacement given in memo.
        """
        return self

    def get_all_var_refs(self) -> Set[vr.VariableReference]:
        """Returns all the variable references that are used in node

        """
        return set()

    def mutate_var_ref(self, var_refs: list[vr.VariableReference]) -> bool:
        """Mutate one of the variable references in `self._node` so that it
        points to some other variable reference in var_refs."""
        all_refs = self.get_all_var_refs().union(var_refs)
        return False

    def replace_var_ref(self, old_var : vr.VariableReference, new_var : vr.VariableReference) -> 'VariableRefAST':
        """Replace occurrences of old_var with the new_var.
        """
        return self.clone_node({old_var: new_var})



