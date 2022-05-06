#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
# This file tests the "expandable" test cluster
# TODO: no tests for name collisions yet.

from pynguin.setup.testcluster import ExpandableTestCluster
from pynguin.setup.testclustergenerator import TestClusterGenerator
from pynguin.utils.generic.genericaccessibleobject import (
    GenericConstructor,
    GenericFunction,
)

# Retrieve GenericAccessibleObjects for function/constructors *in the test module*


def test_can_retrieve_constructor_without_qualification():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor when given only its class name.
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_dependencies", True
    ).generate_cluster()
    gao = expandable_cluster.try_resolve_call("Test")
    assert isinstance(gao, GenericConstructor) and gao.owner.__name__ == "Test"


def test_can_retrieve_function_without_qualification():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a high-level function (i.e. not a method) when given only its name.
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_dependencies", True
    ).generate_cluster()
    gao = expandable_cluster.try_resolve_call("test_function")
    assert isinstance(gao, GenericFunction) and gao.function_name == "test_function"


# Retrieve GenericAccessibleObjects for constructors *imported as type hints*


def test_can_retrieve_typehint_constructor_imported_from():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, which is imported via a `from ... import ...`
    statement.
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.typehint_constructor_imports", True
    ).generate_cluster()

    # Check for `constructor`
    yat_noqualified = expandable_cluster.try_resolve_call("YetAnotherType")
    assert (
        yat_noqualified.is_constructor()
        and yat_noqualified.owner.__name__ == "YetAnotherType"
    ), f"Wrong object: {yat_noqualified}"

    # Check for qualified `module.constructor`
    yat_qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.complex_dependency.YetAnotherType"
    )
    assert (
        yat_qualified.is_constructor()
        and yat_qualified.owner.__name__ == "YetAnotherType"
    ), f"Wrong object: {yat_qualified}"


def test_can_retrieve_typehint_constructor_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, whose module is imported via ` import ... as ...`
    """

    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.typehint_constructor_imports", True
    ).generate_cluster()

    # Check for `constructor`
    aat_noqualified = expandable_cluster.try_resolve_call("AnotherArgumentType")
    assert (
        aat_noqualified.is_constructor()
        and aat_noqualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {aat_noqualified}"
    # Check for qualified `module.constructor`
    aat_qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.another_dependency.AnotherArgumentType"
    )
    assert (
        aat_qualified.is_constructor()
        and aat_qualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {aat_qualified}"
    # Check for qualified `as_module.constructor`
    aat_asqualified = expandable_cluster.try_resolve_call("ad.AnotherArgumentType")
    assert (
        aat_asqualified.is_constructor()
        and aat_asqualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {aat_asqualified}"


def test_can_retrieve_typehint_constructor_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, whose module is imported via ` import ...`
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.typehint_constructor_imports", True
    ).generate_cluster()

    # Check for `constructor`
    sat_noqualified = expandable_cluster.try_resolve_call("SomeArgumentType")
    assert (
        sat_noqualified.is_constructor()
        and sat_noqualified.owner.__name__ == "SomeArgumentType"
    ), f"Wrong object: {sat_noqualified}"

    # Check for qualified `module.constructor`
    sat_qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.dependency.SomeArgumentType"
    )
    assert (
        sat_qualified.is_constructor()
        and sat_qualified.owner.__name__ == "SomeArgumentType"
    ), f"Wrong object: {sat_qualified}"


# Retrieve GenericAccesibleObjects for functions and constructors accessible via
# imports, but which are not part of the original test cluster.


def test_can_retrieve_constructor_imported_from():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor which is imported via a `from ... import ...` statement.
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `constructor`
    noqualified = expandable_cluster.try_resolve_call("Test")
    assert (
        noqualified.is_constructor() and noqualified.owner.__name__ == "Test"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.constructor`
    qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.no_dependencies.Test"
    )
    assert (
        qualified.is_constructor() and qualified.owner.__name__ == "Test"
    ), f"Wrong object: {qualified}"


def test_can_retrieve_constructor_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor whose module is imported via ` import ... as ...`
    """

    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `constructor`
    noqualified = expandable_cluster.try_resolve_call("AnotherArgumentType")
    assert (
        noqualified.is_constructor()
        and noqualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.constructor`
    qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.another_dependency.AnotherArgumentType"
    )
    assert (
        qualified.is_constructor() and qualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {qualified}"

    # Check for qualified `as_module.constructor`
    asqualified = expandable_cluster.try_resolve_call("ad.AnotherArgumentType")
    assert (
        asqualified.is_constructor()
        and asqualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {asqualified}"


def test_can_retrieve_constructor_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor, whose module is imported via ` import ...`
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `constructor`
    noqualified = expandable_cluster.try_resolve_call("SomeArgumentType")
    assert (
        noqualified.is_constructor()
        and noqualified.owner.__name__ == "SomeArgumentType"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.constructor`
    qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.dependency.SomeArgumentType"
    )
    assert (
        qualified.is_constructor() and qualified.owner.__name__ == "SomeArgumentType"
    ), f"Wrong object: {qualified}"


def test_can_retrieve_function_imported_from():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function which is imported via a `from ... import ...` statement.
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `function`
    noqualified = expandable_cluster.try_resolve_call("test_function")
    assert (
        noqualified.is_function() and noqualified.function_name == "test_function"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.function`
    qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.no_dependencies.test_function"
    )
    assert (
        qualified.is_function() and qualified.function_name == "test_function"
    ), f"Wrong object: {qualified}"


def test_can_retrieve_function_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function whose module is imported via ` import ... as ...`
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `function`
    noqualified = expandable_cluster.try_resolve_call("a_function_to_call")
    assert (
        noqualified.is_function() and noqualified.function_name == "a_function_to_call"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.function`
    qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.another_dependency.a_function_to_call"
    )
    assert (
        qualified.is_function() and qualified.function_name == "a_function_to_call"
    ), f"Wrong object: {qualified}"

    # Check for qualified `as_module.function`
    asqualified = expandable_cluster.try_resolve_call("ad.a_function_to_call")
    assert (
        asqualified.is_function() and asqualified.function_name == "a_function_to_call"
    ), f"Wrong object: {asqualified}"


def test_can_retrieve_function_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function, whose module is imported via ` import ...`
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `function`
    noqualified = expandable_cluster.try_resolve_call("method_with_optional")
    assert (
        noqualified.is_function()
        and noqualified.function_name == "method_with_optional"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.function`
    qualified = expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.typing_parameters_legacy.method_with_optional"
    )
    assert (
        qualified.is_function() and qualified.function_name == "method_with_optional"
    ), f"Wrong object: {qualified}"


def test_can_retrieve_parent_method():
    """Checks whether `try_resolve_call` can retrieve a method call for a method defined in
    a parent class, if it's a special method (starting with __)
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.inheritance", True
    ).generate_cluster()
    assert len(expandable_cluster.accessible_objects_under_test) == 5
    assert "Bar" in [cls.__name__ for cls in expandable_cluster.modifiers.keys()]
    bar_key = [
        cls for cls in expandable_cluster.modifiers.keys() if cls.__name__ == "Bar"
    ][0]
    assert len(expandable_cluster.modifiers[bar_key]) == 1
    expandable_cluster.try_resolve_method_call(bar_key, "iterator")
    assert len(expandable_cluster.modifiers[bar_key]) == 2


def test_can_retrieve_parent_method_special():
    """Checks whether `try_resolve_call` can retrieve a method call for a method defined in
    a parent class, if it's a special method (starting with __)
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.overridden_inherited_methods", True
    ).generate_cluster()
    assert len(expandable_cluster.accessible_objects_under_test) == 5
    assert "Bar" in [cls.__name__ for cls in expandable_cluster.modifiers.keys()]
    bar_key = [
        cls for cls in expandable_cluster.modifiers.keys() if cls.__name__ == "Bar"
    ][0]
    assert len(expandable_cluster.modifiers[bar_key]) == 1
    expandable_cluster.try_resolve_method_call(bar_key, "__iter__")
    assert len(expandable_cluster.modifiers[bar_key]) == 2


# Test that retrieving generic accessible objects retrieves their dependencies, if known
def test_retrieve_gao_constructor_dependencies():
    """Checks that when `try_resolve_call` retrieves a constructor
    with callable dependencies, those callable dependencies are added to the test cluster.
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # At first it should only contain the function foo
    assert len(expandable_cluster.accessible_objects_under_test) == 1
    assert len(expandable_cluster.generators) == 0, f"{expandable_cluster.generators}"
    assert len(expandable_cluster.modifiers) == 0
    expandable_cluster.try_resolve_call("cd.SomeOtherType")
    assert len(expandable_cluster.generators) == 2
    assert len(expandable_cluster.modifiers) == 2
    generateable_types = [t.__name__ for t in expandable_cluster.generators.keys()]
    assert (
        "SomeOtherType" in generateable_types and "YetAnotherType" in generateable_types
    )


def test_retrieve_gao_function_dependencies():
    """Checks that when `try_resolve_call` retrieves a constructor
    with callable dependencies, those callable dependencies are added to the test cluster.
    """
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # At first it should only contain the function foo
    assert len(expandable_cluster.accessible_objects_under_test) == 1
    assert len(expandable_cluster.generators) == 0, f"{expandable_cluster.generators}"
    assert len(expandable_cluster.modifiers) == 0
    expandable_cluster.try_resolve_call(
        "tests.fixtures.cluster.typing_parameters_legacy.method_with_union"
    )
    assert len(expandable_cluster.generators) == 1
    assert len(expandable_cluster.modifiers) == 0
    generateable_types = [t.__name__ for t in expandable_cluster.generators.keys()]
    assert ["SomeArgumentType"] == generateable_types
