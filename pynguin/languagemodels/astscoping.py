#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

import ast
from typing import Any, Callable, Dict, List, Set

import pynguin.testcase.variablereference as vr
from pynguin.utils import randomness


class VariableReferenceVisitor:
    """A class which visits an ast and returns a copied ast, with  an operation
    applied to all the instances of vr.VariableReferences.
    """

    def __init__(self, copy: bool, operation: Callable[[vr.VariableReference], Any]):
        """Initializes the visitor with the given operation.

        Args:
            copy: whether or not to return a copy of the visited tree
            operation: operation to apply to any VariableReference encountered
                during visiting.
        """
        self._copy = copy
        self._operator = operation

    def visit(self, node):
        """Visits everything, copying the node `node`, except that `self._operator`
        is applied to any children that are VariableReferences.

        Args:
            node: the ast.AST node to visit

        Returns:
            a copy of the node, with `self._operator` applied to all VariableReferences
        """

        fields_to_assign = {}
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    elif isinstance(value, vr.VariableReference):
                        value = self._operator(value)
                    new_values.append(value)
                fields_to_assign[field] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    pass
                else:
                    fields_to_assign[field] = new_node
            elif isinstance(old_value, vr.VariableReference):
                new_node = self._operator(old_value)
                if new_node is None:
                    pass
                else:
                    fields_to_assign[field] = new_node
            elif self._copy:
                fields_to_assign[field] = old_value
        if self._copy:
            return node.__class__(**fields_to_assign)
        else:
            return None


class FreeVariableOperator(ast.NodeTransformer):
    def __init__(self, operation: Callable[[ast.Name], Any]):
        self._bound_variables: Set[str] = set()
        self._operator = operation

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self._bound_variables:
            return self._operator(node)
        else:
            return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        new_args = [self.visit(arg) for arg in node.args]
        new_kwargs = [
            ast.keyword(arg=kwarg.arg, value=self.visit(kwarg.value))
            for kwarg in node.keywords
        ]
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
            new_generators.append(
                ast.comprehension(
                    target=comp.target,
                    iter=self.visit(comp.iter),
                    ifs=[self.visit(iff) for iff in comp.ifs],
                    is_async=comp.is_async,
                )
            )
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


def operate_on_variable_references(
    node: ast.AST, operation: Callable[[vr.VariableReference], Any]
) -> None:
    """Visits `node` and applies an operation on all the VariableReferences in `node`.

    Args:
        node: the node to visit
        operation: the operation to apply on variable references
    """
    _ = VariableReferenceVisitor(False, operation).visit(node)


def copy_and_operate_on_variable_references(
    node: ast.AST, operation: Callable[[vr.VariableReference], Any]
) -> ast.AST:
    """Visits `node` and applies an operation on all the VariableReferences in `node`,
    replacing any VariableReference v that is a free variable with the result of
    operation(v)

    Args:
        node: the node to visit
        operation: the operation to apply on variable references

    Returns:
        a, possibly strange, ast.AST
    """
    return VariableReferenceVisitor(True, operation).visit(node)


def operate_on_free_variables(
    node: ast.AST, operation: Callable[[ast.Name], Any]
) -> ast.AST:
    """Visits `node` and applies an operation on all the free variables in `node`,
    replacing any `ast.Name` node n that is a free variable with the result of
    operation(n)

    Args:
        node: the node to visit
        operation: the operation to apply on free variables

    Returns:
        a, possibly strange, ast.AST
    """
    transformed_node = VariableReferenceVisitor(True, lambda x: x).visit(node)
    transformed_node = FreeVariableOperator(operation).visit(transformed_node)
    return transformed_node


def _replace_with_var_refs(node: ast.AST, ref_dict: Dict[str, vr.VariableReference]):
    """Returns a new ast with all non-bound variables (ast.Name nodes) replaced
    with the corresponding vr.VariableReference in ref_dict.

    Args:
        node: the ast to analyze
        ref_dict: the mapping of free variables to VariableReferences

    Returns:
        a new ast with all free variables replaced with a vr.VariableReference

    Raises:
        a ValueError if there is a non-bound variable whose name is not in ref_dict.
    """

    def replacer(name_node: ast.Name):
        if name_node.id not in ref_dict:
            raise ValueError(
                f"The Name node with name: {ast.unparse} is an unresolved reference"
            )
        else:
            return ref_dict[name_node.id]

    return operate_on_free_variables(node, replacer)


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

    def structural_hash(self):
        """Structural hash for self, using structural_hash() for variable references

        Returns:
            a hash of this object
        """

        def value_hash(current_hash, value: Any) -> int:
            if isinstance(value, ast.AST):
                current_hash += hash_ast_helper(current_hash, value)
            elif isinstance(value, vr.VariableReference):
                current_hash += 17 * value.structural_hash()
            else:
                current_hash += 17 * hash(value)
            return current_hash

        def hash_ast_helper(current_hash: int, node: ast.AST):
            field_dict = dict(ast.iter_fields(node))
            for field, value in field_dict.items():
                current_hash += 17 * hash(field)
                if isinstance(value, list):
                    for elem in value:
                        current_hash = value_hash(current_hash, elem)
                else:
                    current_hash = value_hash(current_hash, value)
            return current_hash

        return hash_ast_helper(31, self._node)

    def structural_eq(
        self,
        other: "VariableRefAST",
        memo: Dict[vr.VariableReference, vr.VariableReference],
    ) -> bool:
        """Compares whether the two AST nodes are equal w.r.t. memo...

        Args:
            other: the VarRefAST to compare to
            memo: the varref mapping

        Returns:
            whether second is struturally equal to self w.r.t. memo
        """

        def value_equal_helper(first: Any, second: Any):
            if type(first) != type(second):
                return False
            elif isinstance(first, ast.AST):
                return equal_helper_ast(first, second)
            elif isinstance(first, vr.VariableReference):
                return first.structural_eq(second, memo)
            else:
                return first == second

        def equal_helper_ast(first: ast.AST, second: ast.AST):
            if type(first) != type(second):
                return False
            first_fields = dict(ast.iter_fields(first))
            second_fields = dict(ast.iter_fields(second))
            if set(first_fields.keys()) != set(second_fields.keys()):
                return False
            for field in first_fields.keys():
                first_value = first_fields[field]
                second_value = second_fields[field]
                if isinstance(first_value, list) and isinstance(second_value, list):
                    if len(first_value) != len(second_value):
                        return False
                    for i in range(len(first_value)):
                        first_elem = first_value[i]
                        second_elem = second_value[i]
                        if not value_equal_helper(first_elem, second_elem):
                            return False
                else:
                    if not value_equal_helper(first_value, second_value):
                        return False
            return True

        return equal_helper_ast(self._node, other._node)

    def clone(
        self, memo: Dict[vr.VariableReference, vr.VariableReference]
    ) -> "VariableRefAST":
        """Clone the node as an ast, doing any replacement given in memo.

        Args:
            memo: the vr.VariableReference replacements to do.

        Returns:
            a clone of this AST

        Raises:
            ValueError: if there is a missing mapping in memo
        """

        def replace_var_ref(v: vr.VariableReference):
            return v.clone(memo)

        cloned = copy_and_operate_on_variable_references(self._node, replace_var_ref)

        # There should be no effect from re-visiting cloned, since all its free
        # variables have been replaced by VariableReferences, and thus will not
        # be visited.
        try:
            ret_val = VariableRefAST(cloned, {})
            return ret_val
        except ValueError:
            # This should never happen because cloned is created by operating on
            # self._node, which was convereted to a weird AST already.
            raise ValueError(
                "clone was called on a VariableRefAST which was incorrectly converted"
            )

    def count_var_refs(self) -> int:
        """Count the number of variable references in self._node.

        Returns:
            the number of variable references in self._node, including dupes
        """
        num_refs = 0

        def count_var_refs(v: vr.VariableReference):
            nonlocal num_refs
            num_refs += 1

        operate_on_variable_references(self._node, count_var_refs)

        return num_refs

    def get_all_var_refs(self) -> Set[vr.VariableReference]:
        """Returns all the variable references that are used in node

        Returns:
            all the variable references that appear in self._node
        """
        var_refs = set()

        def store_var_ref(v: vr.VariableReference):
            var_refs.add(v)

        operate_on_variable_references(self._node, store_var_ref)

        return var_refs

    def mutate_var_ref(self, var_refs: Set[vr.VariableReference]) -> bool:
        """Mutate one of the variable references in `self._node` so that it
        points to some other variable reference in var_refs.

        Args:
            var_refs: the variable references we can choose from

        Returns:
            true if self._node was successfully mutated.
        """
        num_var_refs = self.count_var_refs()
        if num_var_refs == 0:
            return False

        at_least_one_mutated = False

        # We're going to try to mutate the variable references in this order.
        mutation_order = list(range(num_var_refs))
        randomness.shuffle(mutation_order)

        for num_to_mutate in mutation_order:
            # If we've managed to mutate one, break out of the loop
            if at_least_one_mutated:
                break
            else:
                vr_idx = 0
                # This visitor tries to mutate the `num_to_mutate`th VariableReference
                # visited.

                def mutate_ref(v: vr.VariableReference):
                    nonlocal vr_idx, num_to_mutate, at_least_one_mutated
                    if vr_idx == num_to_mutate:
                        vr_idx += 1
                        candidate_refs = list(var_refs.difference({v}))
                        if len(candidate_refs) == 0:
                            return v
                        else:
                            replacer = randomness.choice(candidate_refs)
                            at_least_one_mutated = True
                            return replacer
                    else:
                        vr_idx += 1
                        return v

                self._node = copy_and_operate_on_variable_references(
                    self._node, mutate_ref
                )

        return at_least_one_mutated

    def replace_var_ref(
        self, old_var: vr.VariableReference, new_var: vr.VariableReference
    ) -> "VariableRefAST":
        """Replace occurrences of old_var with the new_var.

        Args:
            old_var: the variable to replace
            new_var: the variable to replace it with

        Returns:
            a copy of this object, with new_var instead of old_var
        """
        return self.clone({old_var: new_var})

    def get_normal_ast(
        self, vr_replacer: Callable[[vr.VariableReference], ast.Name | ast.Attribute]
    ) -> ast.AST:
        """Gets a normal ast out of the stored AST in self._node, which has variable
        references in places of names.

        Args:
            vr_replacer: the function that replaces vr.VariableReferences with ast.ASTs

        Returns:
            an AST with all VariableReferences replaced by ast.Names or ast.Attributes,
            as mandated by vr_replacer.
        """
        ret_val = copy_and_operate_on_variable_references(self._node, vr_replacer)
        return ret_val
