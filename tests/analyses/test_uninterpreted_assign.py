#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
import pytest

import pynguin.configuration as config
from pynguin.analyses.codedeserializer import deserialize_code_to_testcases
from pynguin.generation.export.exportprovider import ExportProvider
from pynguin.setup.testclustergenerator import TestClusterGenerator


def test_list_literal_uninterpreted_assign():
    """UninterpretedAssignment should allow"""

    testcase_seed = """def test_case_0():
    int_0 = 0
    int_1 = 1
    int_2 = (int_0, int_1)
    var_0 = list(int_2)"""

    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()

    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, test_cluster, True)
    content = ExportProvider.get_exporter().export_sequences_to_str(testcases)
    assert (
        content == testcase_seed
    ), f"=======\n{content}\n=== differs from ===\n{testcase_seed}"


def test_fail_to_parse_unbound_varref():
    """UninterpretedAssignment should allow"""

    testcase_seed = """def test_case_0():
    int_0 = 0
    int_2 = [x + int_0 for x in lst_0]"""

    config.configuration.seeding.include_partially_parsable = True

    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()

    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, test_cluster, True)
    assert len(testcases) == 1
    assert len(testcases[0].statements) == 1


@pytest.mark.parametrize(
    "testcase_seed",
    [
        (
            """    var_0 = lambda x: x + 2
    var_1 = module_0.positional_only(var_0)"""
        ),
        (
            """    int_0 = 0
    int_1 = [int_0, int_0, int_0]
    var_0 = [x for x in int_1]
    var_1 = module_0.positional_only(var_0)"""
        ),
        (
            """    int_0 = 0
    int_1 = [int_0, int_0, int_0]
    var_0 = {x for x in int_1}
    var_1 = module_0.positional_only(var_0)"""
        ),
        (
            """    int_0 = 0
    int_1 = [int_0, int_0, int_0]
    var_0 = {x: x + 1 for x in int_1}
    var_1 = module_0.positional_only(var_0)"""
        ),
        (
            """    int_0 = 0
    int_1 = 1
    var_0 = int_0 + int_1
    var_1 = module_0.positional_only(var_0)"""
        ),
        (
            """    int_0 = 2
    a_0 = module_0.A(int_0)
    var_0 = a_0.x
    var_1 = a_0.y
    var_2 = a_0.a"""
        ),
    ],
)
def test_uninterpreted_assign_roundtrip(testcase_seed):
    testcase_seed = (
        """import tests.fixtures.grammar.parameters as module_0

def test_case_0():
"""
        + testcase_seed
    )

    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()

    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, test_cluster, True)
    content = ExportProvider.get_exporter().export_sequences_to_str(testcases)
    assert (
        content == testcase_seed
    ), f"=======\n{content}\n=== differs from ===\n{testcase_seed}"


def test_uninterpreted_assign_expandable():
    testcase_seed = """import tests.fixtures.cluster.list_annot as module_0

def test_case_0():
    int_0 = 5
    var_0 = lambda x: x > int_0
    int_1 = 10
    var_1 = range(int_1)
    var_2 = module_0.foo(var_1)
    var_3 = list(var_2)"""

    config.configuration.seeding.include_partially_parsable = True
    config.configuration.seeding.allow_expandable_cluster = True

    test_cluster = TestClusterGenerator(
        "tests.fixtures.cluster.list_annot", True
    ).generate_cluster()

    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, test_cluster, True)
    content = ExportProvider.get_exporter().export_sequences_to_str(testcases)
    assert (
        content == testcase_seed
    ), f"=======\n{content}\n=== differs from ===\n{testcase_seed}"
