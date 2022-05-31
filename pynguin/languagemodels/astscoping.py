#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

import ast
from typing import Any, Set, List, Dict, Callable

import pynguin.testcase.variablereference as vr

class FreeVariableOperator(ast.NodeTransformer):

    def __init__(self,  operation: Callable[[ast.expr], Any]):
        self._bound_variables = set()
        self._operator = operation

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self._bound_variables:
            return self._operator(node)
        else:
            return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        new_args = [self.visit(arg) for arg in node.args]
        new_kwargs = [ast.keyword(arg=kwarg.arg, value=self.visit(kwarg.value)) for kwarg in node.keywords]
        return ast.Call(func=node.func, args=new_args, keywords=new_kwargs)

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        bound_variables_before = set(self._bound_variables)
        all_args: ast.arguments = node.args
        for arg in all_args.args + all_args.kwonlyargs:
            arg_name = arg.arg
            self._bound_variables.add(arg_name)
        if all_args.kwarg is not None:
            self._bound_variables.add(all_args.kwarg.arg)
        if all_args.vararg is not None:
            self._bound_variables.add(all_args.vararg.arg)
        new_body = self.visit(node.body)
        self._bound_variables = bound_variables_before
        return ast.Lambda(args=node.args, body=new_body)

    def get_comprehension_bound_vars(self, node: ast.comprehension) -> List[str]:
        return [elem.id for elem in ast.walk(node.target) if isinstance(elem, ast.Name)]

    def _visit_generators_common(self, generators: List[ast.comprehension]):
        new_generators = []
        for comp in generators:
            self._bound_variables.update(self.get_comprehension_bound_vars(comp))
            new_generators.append(ast.comprehension(target=comp.target, iter=self.visit(comp.iter),
                                                    ifs=[self.visit(iff) for iff in comp.ifs],
                                                    is_async=comp.is_async))
        return new_generators

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        bound_variables_before = set(self._bound_variables)
        new_generators = self._visit_generators_common(node.generators)
        new_elt = self.visit(node.elt)
        ret_val = ast.ListComp(elt=new_elt, generators=new_generators)
        self._bound_variables = bound_variables_before
        return ret_val

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        bound_variables_before = set(self._bound_variables)
        new_generators = self._visit_generators_common(node.generators)
        new_elt = self.visit(node.elt)
        ret_val = ast.SetComp(elt=new_elt, generators=new_generators)
        self._bound_variables = bound_variables_before
        return ret_val

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        bound_variables_before = set(self._bound_variables)
        new_generators = self._visit_generators_common(node.generators)
        new_key = self.visit(node.key)
        new_value = self.visit(node.value)
        ret_val = ast.DictComp(key=new_key, value=new_value, generators=new_generators)
        self._bound_variables = bound_variables_before
        return ret_val



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
    transformed_node = FreeVariableOperator(operation).visit(node)
    return transformed_node


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



