#  This file is part of CodaMOSA.
#
#  SPDX-FileCopyrightText: Microsoft.
#
#  SPDX-License-Identifier: MIT
#
# This file tests the "expandable" test cluster
# TODO: no tests for name collisions yet.

import pynguin.configuration as config
from pynguin.analyses.codedeserializer import deserialize_code_to_testcases
from pynguin.setup.testcluster import ExpandableTestCluster, TestCluster
from pynguin.setup.testclustergenerator import TestClusterGenerator
from pynguin.utils.generic.genericaccessibleobject import (
    GenericAccessibleObject,
    GenericConstructor,
    GenericFunction,
)

# Retrieve GenericAccessibleObjects for function/constructors *in the test module*


def try_resolve_call(cluster, fn_name) -> GenericAccessibleObject | None:
    config.configuration.seeding.allow_expandable_cluster = True
    testcase_seed = f"""def test_foo():
    bar = {fn_name}()
    """
    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, cluster)
    if len(testcases) == 1:
        if len(testcases[0].statements) == 1:
            return testcases[0].statements[0].accessible_object()
    else:
        return None


def test_can_retrieve_constructor_without_qualification():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor when given only its class name.
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_dependencies", True
    ).generate_cluster()
    gao = try_resolve_call(expandable_cluster, "Test")
    assert isinstance(gao, GenericConstructor) and gao.owner.__name__ == "Test"


def test_can_retrieve_function_without_qualification():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a high-level function (i.e. not a method) when given only its name.
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_dependencies", True
    ).generate_cluster()
    gao = try_resolve_call(expandable_cluster, "test_function")
    assert isinstance(gao, GenericFunction) and gao.function_name == "test_function"


# Retrieve GenericAccessibleObjects for constructors *imported as type hints*


def test_can_retrieve_typehint_constructor_imported_from():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, which is imported via a `from ... import ...`
    statement.
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.typehint_constructor_imports", True
    ).generate_cluster()

    # Check for `constructor`
    yat_noqualified = try_resolve_call(expandable_cluster, "YetAnotherType")
    assert (
        yat_noqualified.is_constructor()
        and yat_noqualified.owner.__name__ == "YetAnotherType"
    ), f"Wrong object: {yat_noqualified}"

    # Check for qualified `module.constructor`
    yat_qualified = try_resolve_call(
        expandable_cluster, "tests.fixtures.cluster.complex_dependency.YetAnotherType"
    )
    assert (
        yat_qualified.is_constructor()
        and yat_qualified.owner.__name__ == "YetAnotherType"
    ), f"Wrong object: {yat_qualified}"


def test_can_retrieve_typehint_constructor_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, whose module is imported via ` import ... as ...`
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.typehint_constructor_imports", True
    ).generate_cluster()

    # Check for `constructor`
    aat_noqualified = try_resolve_call(expandable_cluster, "AnotherArgumentType")
    assert (
        aat_noqualified.is_constructor()
        and aat_noqualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {aat_noqualified}"
    # Check for qualified `module.constructor`
    aat_qualified = try_resolve_call(
        expandable_cluster,
        "tests.fixtures.cluster.another_dependency.AnotherArgumentType",
    )
    assert (
        aat_qualified.is_constructor()
        and aat_qualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {aat_qualified}"
    # Check for qualified `as_module.constructor`
    aat_asqualified = try_resolve_call(expandable_cluster, "ad.AnotherArgumentType")
    assert (
        aat_asqualified.is_constructor()
        and aat_asqualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {aat_asqualified}"


def test_can_retrieve_typehint_constructor_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, whose module is imported via ` import ...`
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.typehint_constructor_imports", True
    ).generate_cluster()

    sat_noqualified = try_resolve_call(expandable_cluster, "SomeArgumentType")
    assert (
        sat_noqualified.is_constructor()
        and sat_noqualified.owner.__name__ == "SomeArgumentType"
    ), f"Wrong object: {sat_noqualified}"

    # Check for qualified `module.constructor`
    sat_qualified = try_resolve_call(
        expandable_cluster, "tests.fixtures.cluster.dependency.SomeArgumentType"
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
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `constructor`
    noqualified = try_resolve_call(expandable_cluster, "Test")
    assert (
        noqualified.is_constructor() and noqualified.owner.__name__ == "Test"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.constructor`
    qualified = try_resolve_call(
        expandable_cluster, "tests.fixtures.cluster.no_dependencies.Test"
    )
    assert (
        qualified.is_constructor() and qualified.owner.__name__ == "Test"
    ), f"Wrong object: {qualified}"


def test_can_retrieve_constructor_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor whose module is imported via ` import ... as ...`
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `constructor`
    noqualified = try_resolve_call(expandable_cluster, "AnotherArgumentType")
    assert (
        noqualified.is_constructor()
        and noqualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.constructor`
    qualified = try_resolve_call(
        expandable_cluster,
        "tests.fixtures.cluster.another_dependency.AnotherArgumentType",
    )
    assert (
        qualified.is_constructor() and qualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {qualified}"

    # Check for qualified `as_module.constructor`
    asqualified = try_resolve_call(expandable_cluster, "ad.AnotherArgumentType")
    assert (
        asqualified.is_constructor()
        and asqualified.owner.__name__ == "AnotherArgumentType"
    ), f"Wrong object: {asqualified}"


def test_can_retrieve_constructor_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor, whose module is imported via ` import ...`
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `constructor`
    noqualified = try_resolve_call(expandable_cluster, "SomeArgumentType")
    assert (
        noqualified.is_constructor()
        and noqualified.owner.__name__ == "SomeArgumentType"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.constructor`
    qualified = try_resolve_call(
        expandable_cluster, "tests.fixtures.cluster.dependency.SomeArgumentType"
    )
    assert (
        qualified.is_constructor() and qualified.owner.__name__ == "SomeArgumentType"
    ), f"Wrong object: {qualified}"


def test_can_retrieve_function_imported_from():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function which is imported via a `from ... import ...` statement.
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `function`
    noqualified = try_resolve_call(expandable_cluster, "test_function")
    assert (
        noqualified.is_function() and noqualified.function_name == "test_function"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.function`
    qualified = try_resolve_call(
        expandable_cluster, "tests.fixtures.cluster.no_dependencies.test_function"
    )
    assert (
        qualified.is_function() and qualified.function_name == "test_function"
    ), f"Wrong object: {qualified}"


def test_can_retrieve_function_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function whose module is imported via ` import ... as ...`
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `function`
    noqualified = try_resolve_call(expandable_cluster, "a_function_to_call")
    assert (
        noqualified.is_function() and noqualified.function_name == "a_function_to_call"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.function`
    qualified = try_resolve_call(
        expandable_cluster,
        "tests.fixtures.cluster.another_dependency.a_function_to_call",
    )
    assert (
        qualified.is_function() and qualified.function_name == "a_function_to_call"
    ), f"Wrong object: {qualified}"

    # Check for qualified `as_module.function`
    asqualified = try_resolve_call(expandable_cluster, "ad.a_function_to_call")
    assert (
        asqualified.is_function() and asqualified.function_name == "a_function_to_call"
    ), f"Wrong object: {asqualified}"


def test_can_retrieve_function_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function, whose module is imported via ` import ...`
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # Check for `function`
    noqualified = try_resolve_call(expandable_cluster, "method_with_optional")
    assert (
        noqualified.is_function()
        and noqualified.function_name == "method_with_optional"
    ), f"Wrong object: {noqualified}"

    # Check for qualified `module.function`
    qualified = try_resolve_call(
        expandable_cluster,
        "tests.fixtures.cluster.typing_parameters_legacy.method_with_optional",
    )
    assert (
        qualified.is_function() and qualified.function_name == "method_with_optional"
    ), f"Wrong object: {qualified}"


def test_can_retrieve_parent_method():
    """Checks whether `try_resolve_call` can retrieve a method call for a method defined in
    a parent class, if it's a special method (starting with __)
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.inheritance", True
    ).generate_cluster()
    assert len(expandable_cluster.accessible_objects_under_test) == 5
    assert "Bar" in [cls.__name__ for cls in expandable_cluster.modifiers.keys()]
    bar_key = [
        cls for cls in expandable_cluster.modifiers.keys() if cls.__name__ == "Bar"
    ][0]
    assert len(expandable_cluster.modifiers[bar_key]) == 1
    testcase_seed = """def test_foo():
    bar = Bar()
    var_0 = bar.iterator()
    """
    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, expandable_cluster)
    assert len(testcases) == 1
    assert len(testcases[0].statements) == 2
    assert len(expandable_cluster.modifiers[bar_key]) == 2


def test_can_retrieve_parent_method_special():
    """Checks whether `try_resolve_call` can retrieve a method call for a method defined in
    a parent class, if it's a special method (starting with __)
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.overridden_inherited_methods", True
    ).generate_cluster()
    assert len(expandable_cluster.accessible_objects_under_test) == 5
    assert "Bar" in [cls.__name__ for cls in expandable_cluster.modifiers.keys()]
    bar_key = [
        cls for cls in expandable_cluster.modifiers.keys() if cls.__name__ == "Bar"
    ][0]
    assert len(expandable_cluster.modifiers[bar_key]) == 1

    testcase_seed = """def test_foo():
    bar = Bar()
    var_0 = bar.__iter__()
    """
    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, expandable_cluster)
    assert len(testcases) == 1
    assert len(testcases[0].statements) == 2
    assert len(expandable_cluster.modifiers[bar_key]) == 2


# Test that retrieving generic accessible objects retrieves their dependencies, if known
def test_retrieve_gao_constructor_dependencies():
    """Checks that when `try_resolve_call` retrieves a constructor
    with callable dependencies, those callable dependencies are added to the test cluster.
    """
    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # At first it should only contain the function foo
    assert len(expandable_cluster.accessible_objects_under_test) == 1
    assert len(expandable_cluster.generators) == 0, f"{expandable_cluster.generators}"
    assert len(expandable_cluster.modifiers) == 0
    try_resolve_call(expandable_cluster, "cd.SomeOtherType")
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

    config.configuration.seeding.allow_expandable_cluster = True
    expandable_cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # At first it should only contain the function foo
    assert len(expandable_cluster.accessible_objects_under_test) == 1
    assert len(expandable_cluster.generators) == 0, f"{expandable_cluster.generators}"
    assert len(expandable_cluster.modifiers) == 0
    try_resolve_call(
        expandable_cluster,
        "tests.fixtures.cluster.typing_parameters_legacy.method_with_union",
    )
    assert len(expandable_cluster.generators) == 1
    assert len(expandable_cluster.modifiers) == 0
    generateable_types = [t.__name__ for t in expandable_cluster.generators.keys()]
    assert ["SomeArgumentType"] == generateable_types


def test_expand_full_cluster():
    """Tests that the configuration option to make the full expanded cluster expands
    the cluster from the start."""

    # expandable_cluster: TestCluster = TestClusterGenerator(
    #     "tests.fixtures.cluster.no_typehint_imports", True
    # ).generate_cluster()
    # assert len(expandable_cluster.accessible_objects_under_test) == 1
    # assert len(expandable_cluster.generators) == 0
    # assert len(expandable_cluster.modifiers) == 0
    #
    # assert not config.configuration.seeding.expand_cluster

    config.configuration.seeding.expand_cluster = True
    expandable_cluster: TestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.no_typehint_imports", True
    ).generate_cluster()
    # There are 5 imports in this file:
    # another depdendency: 1 function, 1 class with constructor
    #   + 1 generator
    # complex dependency: 2 classes with a modifier each
    #   + 2 generators
    #   + 2 modifiers
    # dependency: 1 class with no modifier
    #   + 1 generator
    # typing_parameters_legacy: 3 functions that output None
    # no_dependencies: Test class (1 modifier), test_function
    #   + 1 generator
    #   + 1 modifier
    assert len(expandable_cluster.accessible_objects_under_test) == 1
    assert len(expandable_cluster.generators) == 5, "\n".join(
        [
            str((t.__name__, len(gens)))
            for t, gens in expandable_cluster.generators.items()
        ]
    )
    assert len(expandable_cluster.modifiers) == 3
