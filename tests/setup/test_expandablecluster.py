#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

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

def test_can_retrieve_constructor():
    cluster: ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.complex_dependencies", True
    ).generate_cluster()
    gao = cluster.try_resolve_call('SomeClass')
    assert gao is not None, f'{gao}'

def test_accessible_expand_with_types():
    cluster : ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.typeless_dependencies", True
    ).generate_cluster()
    assert isinstance(cluster, ExpandableTestCluster)
    assert len(cluster.accessible_objects_under_test) == 2
    class_names = [c.__name__ for c in cluster.get_all_generatable_types()]
    assert 'SomeClass' in class_names, f'{class_names}'
    assert 'SomeOtherType' not in class_names, f'SomeOtherType already in {class_names}'
    cluster.try_resolve_call("SomeOtherType")
    assert len(cluster.accessible_objects_under_test) == 6
    class_names = [c.__name__ for c in cluster.get_all_generatable_types()]
    assert 'SomeOtherType' in class_names, f'SomeOtherType not in {class_names}'
    assert 'YetAnotherType' in class_names, f'YetAnotherType not in {class_names}'

def test_accessible_expand_without_types_in_import():
    cluster : ExpandableTestCluster = TestClusterGenerator(
        "tests.fixtures.cluster.typeless_dependencies", True
    ).generate_cluster()
    assert isinstance(cluster, ExpandableTestCluster)
    assert len(cluster.accessible_objects_under_test) == 2
    class_names = [c.__name__ for c in cluster.get_all_generatable_types()]
    assert 'SomeClass' in class_names, f'{class_names}'
    assert 'MyOtherType' not in class_names, f'MyOtherType already in {class_names}'
    cluster.try_resolve_call("tests.fixtures.cluster.typeless_dependency.MyOtherType")
    assert len(cluster.accessible_objects_under_test) == 4
    class_names = [c.__name__ for c in cluster.get_all_generatable_types()]
    assert 'MyOtherType' in class_names, f'MyOtherType not in {class_names}'

