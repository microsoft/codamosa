#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
import pytest

from pynguin.analyses.codedeserializer import deserialize_code_to_testcases
from pynguin.generation.export.exportprovider import ExportProvider
from pynguin.setup.testclustergenerator import TestClusterGenerator
import pynguin.configuration as config

def test_list_literal_uninterpreted_assign():
    """UninterpretedAssignment should allow
    """

    testcase_seed = """def test_case_0():
    int_0 = 0
    int_1 = 1
    int_2 = (int_0, int_1)
    int_3 = list(int_2)"""

    config.configuration.seeding.uninterpreted_statements = True

    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()

    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, test_cluster)
    content = ExportProvider.get_exporter().export_sequences_to_str(testcases)
    assert (
        content == testcase_seed
    ), f"=======\n{content}\n=== differs from ===\n{testcase_seed}"


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
        """    int_0 = 0
    int_1 = 1
    var_0 = int_0 + int_1
    var_1 = module_0.positional_only(var_0)"""

    ],
)
def test_uninterpreted_assign_roundtrip(testcase_seed):
    testcase_seed = (
        """import tests.fixtures.grammar.parameters as module_0

def test_case_0():
"""
        + testcase_seed
    )

    config.configuration.seeding.uninterpreted_statements = True

    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()

    testcases, _, _ = deserialize_code_to_testcases(testcase_seed, test_cluster)
    content = ExportProvider.get_exporter().export_sequences_to_str(testcases)
    assert (
        content == testcase_seed
    ), f"=======\n{content}\n=== differs from ===\n{testcase_seed}"

