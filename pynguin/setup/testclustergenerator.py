#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides capabilities to create a test cluster"""
from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import logging
import os
import re
import typing

from typing_inspect import get_args, is_union_type

import pynguin.configuration as config
from pynguin.setup.testcluster import (
    ExpandableTestCluster,
    FullTestCluster,
    TestCluster,
)
from pynguin.typeinference import typeinference
from pynguin.typeinference.nonstrategy import NoTypeInferenceStrategy
from pynguin.typeinference.strategy import TypeInferenceStrategy
from pynguin.typeinference.stubstrategy import StubInferenceStrategy
from pynguin.typeinference.typehintsstrategy import TypeHintsInferenceStrategy
from pynguin.utils.exceptions import ConfigurationException
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericEnum,
    GenericFunction,
    GenericMethod,
)
from pynguin.utils.type_utils import (
    class_in_module,
    class_not_in_module,
    function_in_module,
    function_not_in_module,
    get_class_that_defined_method,
    is_primitive_type,
    is_type_unknown,
)


def _retrieve_plain_imports(filename: typing.Optional[str]):
    if filename is not None and os.path.exists(filename):
        # pylint: disable=unspecified-encoding
        with open(filename) as open_file:
            lines = open_file.read().split("\n")
            import_lines = [line for line in lines if line.startswith("import")]
            import_re = re.compile(
                r"^import ((?:[a-zA-Z_][a-zA-Z_0-9]*)"
                r"(?:\.[a-zA-Z_][a-zA-Z_0-9]*)*)\s*(?:#.*)?$"
            )
            matches = [import_re.match(line) for line in import_lines]
            directly_imported_modules = [m.group(1) for m in matches if m is not None]
            return directly_imported_modules
    return []


@dataclasses.dataclass(eq=True, frozen=True)
class DependencyPair:
    """
    Represents a dependency for a type that still needs to be resolved.
    We also store the recursion level, so we can enforce a limit on it.
    The recursion level is excluded from hash/eq so we don't get duplicate
    dependencies at different recursion levels.
    """

    dependency_type: type = dataclasses.field(compare=True, hash=True)
    recursion_level: int = dataclasses.field(compare=False, hash=False)


class TestClusterGenerator:  # pylint: disable=too-few-public-methods
    """Generate a new test cluster"""

    _logger = logging.getLogger(__name__)

    def __init__(self, modules_name: str, make_expandable: bool = False):
        self._module_name = modules_name
        self._analyzed_classes: set[type] = set()
        self._dependencies_to_solve: set[DependencyPair] = set()
        self._make_expandable_cluster = make_expandable
        if make_expandable:
            assert (
                config.configuration.seeding.allow_expandable_cluster
                or config.configuration.seeding.expand_cluster
            )
        if not self._make_expandable_cluster:
            self._test_cluster = FullTestCluster()
        else:
            self._test_cluster = ExpandableTestCluster()
        self._inference = typeinference.TypeInference(
            strategies=self._initialise_type_inference_strategies()
        )
        self._enable_conditional_typing_imports()

    @staticmethod
    def _enable_conditional_typing_imports():
        """Enable imports that are conditional on the typing.TYPE_CHECKING variable.
        See https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING"""
        typing.TYPE_CHECKING = True

    @staticmethod
    def _initialise_type_inference_strategies() -> list[TypeInferenceStrategy]:
        strategy = config.configuration.type_inference.type_inference_strategy
        if strategy == config.TypeInferenceStrategy.NONE:
            return [NoTypeInferenceStrategy()]
        if strategy == config.TypeInferenceStrategy.STUB_FILES:
            if config.configuration.type_inference.stub_dir == "":
                raise ConfigurationException(
                    "Missing configuration value `stub_dir' for StubInferenceStrategy"
                )
            return [StubInferenceStrategy(config.configuration.type_inference.stub_dir)]
        if strategy == config.TypeInferenceStrategy.TYPE_HINTS:
            return [TypeHintsInferenceStrategy()]
        raise ConfigurationException("Invalid type-inference strategy")

    def generate_cluster(self) -> TestCluster:
        """Generate new test cluster from the configured module.

        Returns:
            The new test cluster
        """
        self._logger.debug("Generating test cluster")
        self._logger.debug("Analyzing module %s", self._module_name)
        module = importlib.import_module(self._module_name)

        if self._make_expandable_cluster:
            for module_name, module_obj in inspect.getmembers(module, inspect.ismodule):
                # Add module aliases so we get all the names right in the first place
                if module_name != module_obj.__name__:
                    self._test_cluster.add_module_alias(  # type: ignore
                        module_obj.__name__, module_name
                    )

        self.add_classes_and_functions(self._module_name, module, True, True, 1)
        self._resolve_dependencies_recursive()

        # If we're making an expandable cluster, create the backup set of GAOs.
        if self._make_expandable_cluster:
            # If we're making the whole expandable cluster from the start, then we
            # never go into backup mode --- just add all the generators, modifiers,
            # etc. to the test cluster
            self._test_cluster.set_backup_mode(  # type: ignore
                not config.configuration.seeding.expand_cluster
            )
            self.add_classes_and_functions(self._module_name, module, False, False, 1)

            # Retrieve functions and classes in imported modules too.
            # First, aliased modules:
            for module_name, module_obj in inspect.getmembers(module, inspect.ismodule):
                if module_name != module_obj.__name__:
                    self.add_classes_and_functions(
                        module_obj.__name__, module_obj, False, True, 2
                    )

            # Modules that aren't aliased are tricky if they're qualified,
            # because only the parent module shows up in inspection.
            # So retrieve the modules directly.
            plain_imported_modules = _retrieve_plain_imports(module.__file__)
            for module_name in plain_imported_modules:
                module_obj = importlib.import_module(module_name)
                self.add_classes_and_functions(module_name, module_obj, False, True, 2)

            self._resolve_dependencies_recursive()
            self._test_cluster.set_backup_mode(False)  # type: ignore
        return self._test_cluster

    # pylint: disable=too-many-arguments
    def add_classes_and_functions(
        self,
        module_name: str,
        module_obj,
        under_test: bool,
        from_module: bool,
        recursion_lvl: int,
    ):
        """Adds all the classes and functions from the module_obj with name module_name
        to the test cluster,if from_module is True, only keep the ones defined in this
        module; if from_module is false, only keep the ones imported by but not defined
         in this module.

        Args:
            module_name: the name of the module
            module_obj: the module object
            under_test: should functions/methods/constructed be added to generic
                        accessibles under test?
            from_module: if True, filter to classes/functions defined in the module;
                    if False, filter to classes/functions defined out of the module.
            recursion_lvl: the level of recursion (1 if this is the module under test;
                       2 if it's already an importedmodule)
        """
        module_check = class_in_module if from_module else class_not_in_module
        function_check = function_in_module if from_module else function_not_in_module

        for _, klass in inspect.getmembers(module_obj, module_check(module_name)):
            self._add_dependency(klass, recursion_lvl, under_test)

        for function_name, funktion in inspect.getmembers(
            module_obj, function_check(module_name)
        ):

            generic_function = GenericFunction(
                funktion, self._inference.infer_type_info(funktion)[0], function_name
            )
            if self._is_protected(
                function_name
            ) or self._discard_accessible_with_missing_type_hints(generic_function):
                self._logger.debug("Skip function %s", function_name)
                continue

            self._logger.debug("Analyzing function %s", function_name)
            self._test_cluster.add_generator(generic_function)
            if under_test:
                self._test_cluster.add_accessible_object_under_test(generic_function)
            self._add_callable_dependencies(generic_function, recursion_lvl)

    def _add_callable_dependencies(
        self, call: GenericCallableAccessibleObject, recursion_level: int
    ) -> None:
        """Add required dependencies.

        Args:
            call: The object whose parameter types should be added as dependencies.
            recursion_level: The current level of recursion of the search
        """
        self._logger.debug("Find dependencies for %s", call)

        if recursion_level > config.configuration.type_inference.max_cluster_recursion:
            self._logger.debug("Reached recursion limit. No more dependencies added.")
            return

        for param_name, type_ in call.inferred_signature.parameters.items():
            self._logger.debug("Resolving '%s' (%s)", param_name, type_)
            types = {type_}
            if is_union_type(type_):
                types = set(get_args(type_))

            for elem in types:
                if is_primitive_type(elem):
                    self._logger.debug("Not following primitive argument.")
                    continue
                if inspect.isclass(elem):
                    assert elem
                    self._logger.debug("Adding dependency for class %s", elem)
                    self._dependencies_to_solve.add(
                        DependencyPair(elem, recursion_level)
                    )
                else:
                    self._logger.debug("Found typing annotation %s, skipping", elem)
                    # TODO(fk) fully support typing annotations.

    def _add_dependency(self, klass: type, recursion_level: int, add_to_test: bool):
        """Add constructor/methods/attributes of the given type to the test cluster.

        Args:
            klass: The type of the dependency
            recursion_level: the current recursion level of the search
            add_to_test: whether the accessible objects are also added to objects
                under test.
        """
        assert inspect.isclass(klass), "Can only add dependencies for classes."
        if klass in self._analyzed_classes:
            self._logger.debug("Class %s already analyzed", klass)
            return
        self._analyzed_classes.add(klass)
        if klass == type(None):  # noqa: E721
            self._logger.debug("Class %s is NoneType, skipping", klass)
            return
        self._logger.debug("Analyzing class %s", klass)
        if issubclass(klass, enum.Enum):
            generic: GenericEnum | GenericConstructor = GenericEnum(klass)
        else:
            generic = generic_constructor = GenericConstructor(
                klass, self._inference.infer_type_info(klass.__init__)[0]
            )
            if self._discard_accessible_with_missing_type_hints(generic_constructor):
                return
            self._add_callable_dependencies(generic_constructor, recursion_level)

        self._test_cluster.add_generator(generic)
        if add_to_test:
            self._test_cluster.add_accessible_object_under_test(generic)

        for method_name, method in inspect.getmembers(klass, inspect.isfunction):
            # TODO(fk) why does inspect.ismethod not work here?!
            # TODO(fk) Instance methods of enums are only visible on elements of the
            # enum but not the class itself :|
            self._logger.debug("Analyzing method %s", method_name)

            generic_method = GenericMethod(
                klass, method, self._inference.infer_type_info(method)[0], method_name
            )

            if (
                self._is_constructor(method_name)
                or (
                    not self._is_method_defined_in_class(klass, method)
                    and not self._make_expandable_cluster
                )
                or self._is_protected(method_name)
                or self._discard_accessible_with_missing_type_hints(generic_method)
            ):
                # Skip methods that should not be added to the cluster here.
                # Constructors are handled elsewhere; inherited methods should not be
                # part of the cluster, only overridden methods; private methods should
                # neither be part of the cluster.
                continue

            if (
                not self._is_method_defined_in_class(klass, method)
                and self._make_expandable_cluster
            ):
                # If we're making an expandable cluster, keep track of methods not
                # directly defined in the object under test as modifiers of this class.
                #
                # If we're already in backup mode, no need to set/unset it
                if self._test_cluster.get_backup_mode():  # type: ignore
                    self._test_cluster.add_modifier(klass, generic_method)
                else:
                    # If we're fully expanding the cluster, don't go into backup mode.
                    self._test_cluster.set_backup_mode(  # type: ignore
                        not config.configuration.seeding.expand_cluster
                    )
                    self._test_cluster.add_modifier(klass, generic_method)
                    self._test_cluster.set_backup_mode(False)  # type: ignore
                # TODO(ANON): this doesn't keep track of callable dependencies...
                # in most cases if it's a class we're directly inheriting, we should
                # be importing that class and resolving the dependencies there.
                continue

            self._test_cluster.add_generator(generic_method)
            self._test_cluster.add_modifier(klass, generic_method)
            if add_to_test:
                self._test_cluster.add_accessible_object_under_test(generic_method)
            self._add_callable_dependencies(generic_method, recursion_level)
        # TODO(fk) how do we find attributes?

    @staticmethod
    def _is_constructor(method_name: str) -> bool:
        return method_name == "__init__"

    @staticmethod
    def _is_method_defined_in_class(class_: type, method: object) -> bool:
        return class_ == get_class_that_defined_method(method)

    @staticmethod
    def _is_protected(method_name: str) -> bool:
        return method_name.startswith("_") and not method_name.startswith("__")

    @staticmethod
    def _discard_accessible_with_missing_type_hints(
        accessible_object: GenericCallableAccessibleObject,
    ) -> bool:
        """Should we discard accessible objects that are not fully type hinted?

        Args:
            accessible_object: the object to check

        Returns:
            Whether or not the accessible should be discarded
        """
        if config.configuration.type_inference.guess_unknown_types:
            return False
        inf_sig = accessible_object.inferred_signature
        return any(
            is_type_unknown(type_) for param, type_ in inf_sig.parameters.items()
        )

    def _resolve_dependencies_recursive(self):
        """Resolve the currently open dependencies."""
        while self._dependencies_to_solve:
            to_solve = self._dependencies_to_solve.pop()
            self._add_dependency(
                to_solve.dependency_type, to_solve.recursion_level + 1, False
            )
