#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
# This file tests the "expandable" test cluster
# TODO: no tests for name collisions yet.

import os
from typing import cast

import pytest
from ordered_set import OrderedSet

import pynguin.configuration as config
from pynguin.setup.testclustergenerator import TestClusterGenerator
from pynguin.setup.testcluster import  ExpandableTestCluster
from pynguin.typeinference.nonstrategy import NoTypeInferenceStrategy
from pynguin.typeinference.stubstrategy import StubInferenceStrategy
from pynguin.typeinference.typehintsstrategy import TypeHintsInferenceStrategy
from pynguin.utils.exceptions import ConfigurationException
from pynguin.utils.generic.genericaccessibleobject import (
    GenericAccessibleObject,
    GenericConstructor,
    GenericEnum,
    GenericMethod,
)


# Retrieve GenericAccessibleObjects for function/constructors *in the test module*

def test_can_retrieve_constructor_without_qualification():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor when given only its class name.
    """
    pass

def test_can_retrieve_function_without_qualification():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a high-level function (i.e. not a method) when given only its name.
    """
    pass

# Retrieve GenericAccessibleObjects for constructors *imported as type hints*

def test_can_retrieve_typehint_constructor_imported_from():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, which is imported via a `from ... import ...`
    statement.
    """
    # Check for qualified `module.constructor`
    # Check for `constructor`
    pass

def test_can_retrieve_typehint_constructor_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, whose module is imported via ` import ... as ...`
    """
    # Check for qualified `module.constructor`
    # Check for `constructor`
    # Check for qualified `as_module.constructor`
    pass

def test_can_retrieve_typehint_constructor_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor in a type hint, whose module is imported via ` import ...`
    """
    # Check for qualified `module.constructor`
    # Check for `constructor`
    pass

# Retrieve GenericAccesibleObjects for functions and constructors accessible via
# imports, but which are not part of the original test cluster.

def test_can_retrieve_constructor_imported_from():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor which is imported via a `from ... import ...` statement.
    """
    # Check for qualified `module.constructor`
    # Check for `constructor`
    pass

def test_can_retrieve_constructor_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor whose module is imported via ` import ... as ...`
    """
    # Check for qualified `module.constructor`
    # Check for `constructor`
    # Check for qualified `as_module.constructor`
    pass

def test_can_retrieve_constructor_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a constructor, whose module is imported via ` import ...`
    """
    # Check for qualified `module.constructor`
    # Check for `constructor`
    pass


def test_can_retrieve_function_imported_from():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function which is imported via a `from ... import ...` statement.
    """
    # Check for qualified `module.function`
    # Check for `function`
    pass

def test_can_retrieve_function_imported_as():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function whose module is imported via ` import ... as ...`
    """
    # Check for qualified `module.function`
    # Check for `function`
    # Check for qualified `as_module.function`
    pass

def test_can_retrieve_function_imported():
    """Checks whether `try_resolve_call` can retrieve the GenericAccessibleObject
    for a function, whose module is imported via ` import ...`
    """
    # Check for qualified `module.function`
    # Check for `function`
    pass

# Test that retrieving generic accessible objects retrieves their dependencies, if known
def test_retrieve_gao_dependencies():
    """Checks that when `try_resolve_call` retrieves a GenericAccessibleObject
    with callable dependencies, those callable dependencies are added to the test cluster.
    """
    # Check for qualified `module.constructor`
    # Check for `constructor`
    pass
