#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides a test cluster."""
from __future__ import annotations

import inspect
import json
import logging
import typing
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ordered_set import OrderedSet
from typing_inspect import get_args, is_union_type

from pynguin.instrumentation.instrumentation import CODE_OBJECT_ID_KEY
from pynguin.utils import randomness, type_utils
from pynguin.utils.exceptions import ConstructionFailedException
from pynguin.utils.generic.genericaccessibleobject import (
    GenericAccessibleObject,
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericMethod,
)
from pynguin.utils.type_utils import COLLECTIONS, PRIMITIVES

if typing.TYPE_CHECKING:  # Break circular dependencies at runtime.
    import pynguin.ga.computations as ff
    import pynguin.generation.algorithms.archive as arch
    from pynguin.testcase.execution import KnownData


class TestCluster(ABC):
    """A test cluster which contains all methods/constructors/functions
    and all required transitive dependencies.
    """

    @property
    @abstractmethod
    def accessible_objects_under_test(self) -> OrderedSet[GenericAccessibleObject]:
        """Provides all accessible objects that are under test.

        Returns:
            The set of all accessible objects under test
        """

    @property
    @abstractmethod
    def all_accessible_objects(self) -> OrderedSet[GenericAccessibleObject]:
        """Provides all accessible objects.

        Returns:
            The set of all accessible objects
        """

    @abstractmethod
    def num_accessible_objects_under_test(self) -> int:
        """Provide the number of accessible objects under test.

        This is useful to check if there even is something to test.

        Returns:
            The number of all accessibles under test
        """

    @abstractmethod
    def get_generators_for(self, for_type: type) -> OrderedSet[GenericAccessibleObject]:
        """Retrieve all known generators for the given type.

        Args:
            for_type: The type we want to have the generators for

        Returns:
            The set of all generators for that type
        """

    @abstractmethod
    def get_modifiers_for(self, for_type: type) -> OrderedSet[GenericAccessibleObject]:
        """Get all known modifiers of a type.

        This currently does not take inheritance into account.

        Args:
            for_type: The type

        Returns:
            The set of all accessibles that can modify the type
        """

    @property
    @abstractmethod
    def generators(self) -> dict[type, OrderedSet[GenericAccessibleObject]]:
        """Provides all available generators.

        Returns:
            A dictionary of types and their generating accessibles
        """

    @property
    @abstractmethod
    def modifiers(self) -> dict[type, OrderedSet[GenericAccessibleObject]]:
        """Provides all available modifiers.

        Returns:
            A dictionary of types and their modifying accessibles
        """

    @abstractmethod
    def get_random_accessible(self) -> GenericAccessibleObject | None:
        """Provide a random accessible of the unit under test.

        Returns:
            A random accessible
        """

    @abstractmethod
    def get_random_call_for(self, type_: type) -> GenericAccessibleObject:
        """Get a random modifier for the given type.

        Args:
            type_: The type

        Returns:
            A random modifier for that type

        Raises:
            ConstructionFailedException: if no modifiers for the type exist
        """

    @abstractmethod
    def get_all_generatable_types(self) -> list[type]:
        """Provides all types that can be generated, including primitives
        and collections.

        Returns:
            A list of all types that can be generated
        """

    @abstractmethod
    def select_concrete_type(self, select_from: type | None) -> type | None:
        """Select a concrete type from the given type.

        This is required e.g. when handling union types.
        Currently only unary types, Any and Union are handled.

        Args:
            select_from: An optional type

        Returns:
            An optional type
        """


class FullTestCluster(TestCluster):
    """A test cluster which contains all methods/constructors/functions
    and all required transitive dependencies.
    """

    def __init__(self):
        """Create new test cluster."""
        self._generators: dict[type, OrderedSet[GenericAccessibleObject]] = {}
        self._modifiers: dict[type, OrderedSet[GenericAccessibleObject]] = {}
        self._accessible_objects_under_test: OrderedSet[
            GenericAccessibleObject
        ] = OrderedSet()

    def add_generator(self, generator: GenericAccessibleObject) -> None:
        """Add the given accessible as a generator.

        It is only added if the type is known, not primitive and not NoneType.

        Args:
            generator: The accessible object
        """
        type_ = generator.generated_type()
        if (
            type_ is None
            or type_utils.is_none_type(type_)
            or type_utils.is_primitive_type(type_)
        ):
            return
        if type_ in self._generators:
            self._generators[type_].add(generator)
        else:
            self._generators[type_] = OrderedSet([generator])

    def add_accessible_object_under_test(self, obj: GenericAccessibleObject) -> None:
        """Add accessible object to the objects under test.

        Args:
            obj: The accessible object
        """
        self._accessible_objects_under_test.add(obj)

    def add_modifier(self, type_: type, obj: GenericAccessibleObject) -> None:
        """Add a modifier.

        A modified is something that can be used to modify the given type,
        e.g. a method.

        Args:
            type_: The type that can be modified
            obj: The accessible that can modify
        """
        if type_ in self._modifiers:
            self._modifiers[type_].add(obj)
        else:
            self._modifiers[type_] = OrderedSet([obj])

    @property
    def accessible_objects_under_test(self) -> OrderedSet[GenericAccessibleObject]:
        return self._accessible_objects_under_test

    @property
    def all_accessible_objects(self) -> OrderedSet[GenericAccessibleObject]:
        ret_set: OrderedSet[GenericAccessibleObject] = OrderedSet()
        ret_set = ret_set.union(self._accessible_objects_under_test)
        for vals in self._modifiers.values():
            ret_set = ret_set.union(vals)
        for vals in self._generators.values():
            ret_set = ret_set.union(vals)
        return ret_set

    def num_accessible_objects_under_test(self) -> int:
        return len(self._accessible_objects_under_test)

    def get_generators_for(self, for_type: type) -> OrderedSet[GenericAccessibleObject]:
        if for_type in self._generators:
            return self._generators[for_type]
        return OrderedSet()

    def get_modifiers_for(self, for_type: type) -> OrderedSet[GenericAccessibleObject]:
        if for_type in self._modifiers:
            return self._modifiers[for_type]
        return OrderedSet()

    @property
    def generators(self) -> dict[type, OrderedSet[GenericAccessibleObject]]:
        return self._generators

    @property
    def modifiers(self) -> dict[type, OrderedSet[GenericAccessibleObject]]:
        return self._modifiers

    def get_random_accessible(self) -> GenericAccessibleObject | None:
        if self.num_accessible_objects_under_test() == 0:
            return None
        return randomness.choice(self._accessible_objects_under_test)

    def get_random_call_for(self, type_: type) -> GenericAccessibleObject:
        accessible_objects = self.get_modifiers_for(type_)
        if len(accessible_objects) == 0:
            raise ConstructionFailedException("No modifiers for " + str(type_))
        return randomness.choice(accessible_objects)

    def get_all_generatable_types(self) -> list[type]:
        generatable = list(self._generators.keys())
        generatable.extend(PRIMITIVES)
        generatable.extend(COLLECTIONS)
        return generatable

    def select_concrete_type(self, select_from: type | None) -> type | None:
        if select_from == Any:  # pylint:disable=comparison-with-callable
            return randomness.choice(self.get_all_generatable_types())
        if is_union_type(select_from):
            possible_types = get_args(select_from)
            if possible_types is not None and len(possible_types) > 0:
                return randomness.choice(possible_types)
            return None
        return select_from


class ExpandableTestCluster(FullTestCluster):
    """A test cluster that keeps track of *all possible* method/constructors/functions
    in the module under test as well as *all* accessible modules under import.

    Initially, it behaves as a regular FullTestCluster, but if resolve_function_call
    resolves the function call with a
    """

    def __init__(self):
        """Create new test cluster."""
        super().__init__()
        self._backup_accessible_objects: OrderedSet[
            GenericAccessibleObject
        ] = OrderedSet()
        self._all_backups: OrderedSet[GenericAccessibleObject] = OrderedSet()
        self._backup_mode = False
        self._backup_dependency_map: Dict[GenericAccessibleObject, List[type]] = {}
        self._name_idx: Dict[str, List[GenericAccessibleObject]] = {}
        self._module_aliases: Dict[str, str] = {}

    @property
    def all_accessible_objects(self) -> OrderedSet[GenericAccessibleObject]:
        ret_set = super().all_accessible_objects
        return ret_set.union(self._all_backups)

    def set_backup_mode(self, mode: bool):
        """
        Put the test cluster in backup mode, that is, don't add anything to
        generators/modifiers/test cluster, just keep track of the GAOs so they
        can be retrieved later

        Args:
            mode: if True, turn on backup mode

        """
        self._backup_mode = mode

    def get_backup_mode(self) -> bool:
        """
        Returns whether we are currently in backup mode

        Returns:
            the current backup mode
        """
        return self._backup_mode

    def add_module_alias(self, orig_module_name: str, alias_in_file: str):
        """Keep track of module aliases

        Args:
            orig_module_name: the real name of the module
            alias_in_file: what the module is imported `as` in the file

        """
        self._module_aliases[orig_module_name] = alias_in_file

    def _add_to_index(self, func: GenericAccessibleObject):
        """Adds the function func to the index of names -> GAO mappings.

        Args:
            func: function to add
        """
        if func.is_constructor():
            generated_type = func.generated_type()
            assert generated_type is not None
            func_name = generated_type.__name__
            module_name = generated_type.__module__
            func_names = [func_name, module_name + "." + func_name]
            if module_name in self._module_aliases:
                qual_module_name = self._module_aliases[module_name]
                func_names.append(qual_module_name + "." + func_name)
        elif func.is_function():
            func_name = func.function_name  # type: ignore
            callable_ = func.callable  # type: ignore
            module_name = callable_.__module__
            func_names = [func_name, module_name + "." + func_name]
            if module_name in self._module_aliases:
                qual_module_name = self._module_aliases[module_name]
                func_names.append(qual_module_name + "." + func_name)
        elif func.is_method():
            assert func.owner is not None
            func_name = func.method_name  # type: ignore
            owner_name = func.owner.__name__
            func_names = [owner_name + "." + func_name]
        else:
            func_names = []

        for func_name in func_names:
            if func_name in self._name_idx:
                self._name_idx[func_name].append(func)
            else:
                self._name_idx[func_name] = [func]

    def promote_object(self, func: GenericAccessibleObject):
        """
        Promotes the object to go into generators/modifiers.

        Args:
            func: function to promote
        """
        # Otherwise add_generator and add_modifier will do nothing
        assert self._backup_mode is False
        if func in self._backup_accessible_objects:
            # To prevent recursion when adding dependencies, remove this
            # from backup objects.
            self._backup_accessible_objects.remove(func)

            # Add it as a generator if it can generate types
            self.add_generator(func)

            # Add it as a modifier if it is a method
            if func.is_method():
                modified_type = func.owner
                assert modified_type is not None
                self.add_modifier(modified_type, func)

            # Add dependencies... there is some repetition here with the work done in
            # testclustergenerator.py

            # Promote any types in the type signature to the test cluster
            signature = func.inferred_signature  # type: ignore
            for _, type_ in signature.parameters.items():
                types = {type_}
                if is_union_type(type_):
                    types = set(get_args(type_))
                for elem in types:
                    if inspect.isclass(elem):
                        assert elem
                        # The constructor should be available via the name of the class.
                        constructors = [
                            obj
                            for obj in self._backup_accessible_objects
                            if isinstance(obj, GenericConstructor)
                            and obj.generated_type() == elem
                        ]
                        for constructor in constructors:
                            self.promote_object(constructor)

            # Also retrieve all the methods for a constructor
            if func.is_constructor():
                type_under_test = func.owner
                assert type_under_test is not None
                methods = [
                    obj
                    for obj in self._backup_accessible_objects
                    if isinstance(obj, GenericMethod) and obj.owner == type_under_test
                ]
                for method in methods:
                    self.promote_object(method)

            # TODO: do we also add it to objects under test? No, we want to
            # add it as a generator, but of the correct type. I think that
            # might require dynamically observing types...
            # self.accessible_objects_under_test.add(func)

    def add_generator(self, generator: GenericAccessibleObject) -> None:
        """Add the given accessible as a generator, and keep track of its name.

        Args:
            generator: The accessible object
        """
        self._add_to_index(generator)
        if not self._backup_mode:
            super().add_generator(generator)
        else:
            self._backup_accessible_objects.add(generator)
            self._all_backups.add(generator)

    def add_accessible_object_under_test(self, obj: GenericAccessibleObject) -> None:
        """Add accessible object to the objects under test, and keep track of its name.

        Args:
            obj: The accessible object
        """
        self._add_to_index(obj)
        if not self._backup_mode:
            super().add_accessible_object_under_test(obj)
        else:
            self._backup_accessible_objects.add(obj)
            self._all_backups.add(obj)

    def add_modifier(self, type_: type, obj: GenericAccessibleObject) -> None:
        """Add a modifier.

        A modified is something that can be used to modify the given type,
        e.g. a method, and keep track of its name.

        Args:
            type_: The type that can be modified
            obj: The accessible that can modify
        """
        self._add_to_index(obj)
        if not self._backup_mode:
            super().add_modifier(type_, obj)
        else:
            self._backup_accessible_objects.add(obj)
            self._all_backups.add(obj)

    def was_added_in_backup(self, obj: GenericAccessibleObject):
        """Returns true if the object `obj` was added as a backup. For statistics
        tracking purposes.

        Args:
            obj: the object to check

        Returns:
            True if obj was added in backup mode
        """
        return obj in self._all_backups


class FilteredTestCluster(TestCluster):
    """A test cluster that wraps another test cluster.
    This test cluster forwards most methods to the wrapped delegate.

    This test cluster filters out accessible objects under test that are already
    fully covered, in order to focus the search on areas that are not yet fully covered.
    """

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        test_cluster: TestCluster,
        archive: arch.Archive,
        known_data: KnownData,
        targets: OrderedSet[ff.TestCaseFitnessFunction],
    ):
        self._delegate = test_cluster
        self._known_data = known_data
        self._code_object_id_to_accessible_object: dict[
            int, GenericCallableAccessibleObject
        ] = {
            json.loads(acc.callable.__code__.co_consts[0])[CODE_OBJECT_ID_KEY]: acc
            for acc in test_cluster.accessible_objects_under_test
            if isinstance(acc, GenericCallableAccessibleObject)
            and hasattr(acc.callable, "__code__")
        }
        # Checking for __code__ is necessary, because the __init__ of a class that does
        # not define __init__ points to some internal CPython stuff.

        self._accessible_to_targets: dict[
            GenericCallableAccessibleObject, OrderedSet
        ] = {
            acc: OrderedSet()
            for acc in self._code_object_id_to_accessible_object.values()
        }
        for target in targets:
            if (acc := self._get_accessible_object_for_target(target)) is not None:
                targets_for_acc = self._accessible_to_targets[acc]
                targets_for_acc.add(target)

        # Get informed by archive, when a target is covered.
        archive.add_on_target_covered(self._on_target_covered)

    def _get_accessible_object_for_target(
        self, target: ff.TestCaseFitnessFunction
    ) -> GenericCallableAccessibleObject | None:
        code_object_id: int | None = target.code_object_id
        while code_object_id is not None:
            if (
                acc := self._code_object_id_to_accessible_object.get(
                    code_object_id, None
                )
            ) is not None:
                return acc
            code_object_id = self._known_data.existing_code_objects[
                code_object_id
            ].parent_code_object_id
        return None

    def _on_target_covered(self, target: ff.TestCaseFitnessFunction) -> None:
        acc = self._get_accessible_object_for_target(target)
        if acc is not None:
            targets_for_acc = self._accessible_to_targets.get(acc)
            assert targets_for_acc is not None
            targets_for_acc.remove(target)
            if len(targets_for_acc) == 0:
                self._accessible_to_targets.pop(acc)
                self._logger.debug(
                    "Removed %s from test cluster because all "
                    "targets within it are covered",
                    acc,
                )

    @property
    def accessible_objects_under_test(self) -> OrderedSet[GenericAccessibleObject]:
        accessibles = self._accessible_to_targets.keys()
        if len(accessibles) == 0:
            # Should never happen, just in case everything is already covered?
            return self._delegate.accessible_objects_under_test
        return OrderedSet(accessibles)

    @property
    def all_accessible_objects(self) -> OrderedSet[GenericAccessibleObject]:
        # TODO(ANON): These are not filtered
        ret_set: OrderedSet[GenericAccessibleObject] = OrderedSet()
        ret_set = ret_set.union(self.accessible_objects_under_test)
        for vals in self.modifiers.values():
            ret_set = ret_set.union(vals)
        for vals in self.generators.values():
            ret_set = ret_set.union(vals)
        return ret_set

    def num_accessible_objects_under_test(self) -> int:
        return self._delegate.num_accessible_objects_under_test()

    def get_generators_for(self, for_type: type) -> OrderedSet[GenericAccessibleObject]:
        return self._delegate.get_generators_for(for_type)

    def get_modifiers_for(self, for_type: type) -> OrderedSet[GenericAccessibleObject]:
        return self._delegate.get_modifiers_for(for_type)

    @property
    def generators(self) -> dict[type, OrderedSet[GenericAccessibleObject]]:
        return self._delegate.generators

    @property
    def modifiers(self) -> dict[type, OrderedSet[GenericAccessibleObject]]:
        return self._delegate.modifiers

    def get_random_accessible(self) -> GenericAccessibleObject | None:
        accessibles = self._accessible_to_targets.keys()
        if len(accessibles) == 0:
            return self._delegate.get_random_accessible()
        return randomness.choice(OrderedSet(accessibles))

    def get_random_call_for(self, type_: type) -> GenericAccessibleObject:
        return self._delegate.get_random_call_for(type_)

    def get_all_generatable_types(self) -> list[type]:
        return self._delegate.get_all_generatable_types()

    def select_concrete_type(self, select_from: type | None) -> type | None:
        return self._delegate.select_concrete_type(select_from)
