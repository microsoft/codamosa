#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides a strategy implementation that utilises stub files."""
from __future__ import annotations

import ast
import inspect
import os
import sys
from pydoc import locate
from typing import Callable

from pynguin.typeinference.strategy import InferredSignature, TypeInferenceStrategy
from pynguin.typeinference.typehintsstrategy import TypeHintsInferenceStrategy


# pylint: disable=too-few-public-methods
class StubInferenceStrategy(TypeInferenceStrategy):
    """Provides a strategy that utilises stub files to infer variable types."""

    _cache: dict[str, ast.Module] = {}

    def __init__(self, pyi_dir: str | os.PathLike) -> None:
        self._pyi_dir = pyi_dir

    def infer_type_info(self, method: Callable) -> InferredSignature:
        assert self._pyi_dir

        module = sys.modules[method.__module__]
        name = module.__name__.replace(".", "/")
        pyi_src = name + ".pyi"
        pyi_ast = self._read_stub(pyi_src)
        parameter_types, return_type = self._get_parameter_annotations(method, pyi_ast)
        if parameter_types:
            return InferredSignature(
                signature=inspect.signature(method),
                parameters=parameter_types if parameter_types else {},
                return_type=return_type if return_type else None,
            )
        return TypeHintsInferenceStrategy().infer_type_info(method)

    def _read_stub(self, pyi_src: str) -> ast.Module | None:
        path = os.path.join(self._pyi_dir, pyi_src)
        if path in self._cache:
            return self._cache[path]

        try:
            with open(path, encoding="utf-8") as pyi_file:
                pyi_content = pyi_file.read()
                pyi_ast = ast.parse(pyi_content)
                self._cache[path] = pyi_ast
        except OSError:
            return None
        return pyi_ast

    @staticmethod
    def _get_parameter_annotations(
        method: Callable, pyi_ast: ast.Module | None
    ) -> tuple[dict[str, type | None], type | None]:
        if not pyi_ast:
            return {}, None

        param_types_matches: dict[str, type | None] = {}
        return_type: type | None = None
        if inspect.isfunction(method):
            function_listener = _FunctionListener(pyi_ast, method.__name__)
            function_listener.visit(pyi_ast)
            param_types_matches = function_listener.param_type_matches
            return_type = function_listener.return_type
        elif inspect.isclass(method):
            class_listener = _ClassListener(pyi_ast, method.__name__, "__init__")
            class_listener.visit(pyi_ast)
            param_types_matches = class_listener.param_type_matches
        return param_types_matches, return_type


class _FunctionListener(ast.NodeVisitor):
    def __init__(self, pyi_ast: ast.Module | None, name: str) -> None:
        self._pyi_ast = pyi_ast
        self._name = name
        self._param_type_matches: dict[str, type | None] = {}
        self._return_type: type | None = None

    @property
    def param_type_matches(self) -> dict[str, type | None]:
        """Provides the matched parameter types.

        Returns:
            The matched parameter types
        """
        if "self" in self._param_type_matches:
            del self._param_type_matches["self"]
        return self._param_type_matches

    @property
    def return_type(self) -> type | None:
        """Provides the return type.

        Returns:
            Teh inferred return type
        """
        return self._return_type

    # pylint: disable=invalid-name, missing-function-docstring
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._name == node.name:
            self._param_type_matches.clear()
            for name, field in ast.iter_fields(node):
                if name == "args":
                    for arg in field.args:
                        if arg.annotation and hasattr(arg.annotation, "id"):
                            self._param_type_matches[arg.arg] = self._locate(
                                arg.annotation.id
                            )
                        elif arg.annotation and hasattr(arg.annotation, "attr"):
                            self._param_type_matches[arg.arg] = self._locate(
                                f"{arg.annotation.value.id}.{arg.annotation.attr}"
                            )
                        else:
                            self._param_type_matches[arg.arg] = None
            if node.returns and hasattr(node.returns, "id"):
                self._return_type = self._locate(node.returns.id)  # type: ignore

    def _locate(self, identifier: str | None) -> type | None:
        assert self._pyi_ast
        if not identifier:
            return None
        result = locate(identifier)
        if result:
            return result  # type: ignore
        visitor = _ImportVisitor(identifier)
        visitor.generic_visit(self._pyi_ast)
        return locate(visitor.full_qualified_name)  # type: ignore


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, name: str) -> None:
        self._name = name
        self._full_qualified_name: str | None = None

    @property
    def full_qualified_name(self) -> str | None:
        """Provides the inferred fully-qualified name.

        Returns:
            The fully-qualified name
        """
        return self._full_qualified_name

    # pylint: disable=invalid-name, missing-function-docstring
    def visit_ImportFrom(self, statement: ast.ImportFrom) -> None:
        if statement.names:
            for alias in statement.names:
                if alias.name == self._name:
                    self._full_qualified_name = f"{statement.module}.{alias.name}"


class _ClassListener(ast.NodeVisitor):
    def __init__(
        self, pyi_ast: ast.Module | None, class_name: str, function_name: str
    ) -> None:
        self._pyi_ast = pyi_ast
        self._class_name = class_name
        self._function_name = function_name
        self._param_type_matches: dict[str, type | None] = {}

    @property
    def param_type_matches(self) -> dict[str, type | None]:
        """Provides the matched parameter types.

        Returns:
            The matched parameter types
        """
        return self._param_type_matches

    # pylint: disable=invalid-name, missing-function-docstring
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._class_name == node.name:
            listener = _FunctionListener(self._pyi_ast, self._function_name)
            listener.generic_visit(node)
            self._param_type_matches = listener.param_type_matches
