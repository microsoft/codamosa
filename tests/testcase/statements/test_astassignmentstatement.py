#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#


from pynguin.analyses.codedeserializer import deserialize_code_to_testcases
import pynguin.testcase.testcase as tc
import pytest
import pynguin.configuration as config
from pynguin.generation.export.exportprovider import ExportProvider

from pynguin.setup.testclustergenerator import TestClusterGenerator


# End-to-end test of the ast assign statement's mutation operation
def test_mutate_ast_assign_tc():
    config.configuration.seeding.uninterpreted_statements = True
    config.configuration.seeding.include_partially_parsable = True

    test_case_str="""def test_case_0():
    int_0 = 0
    int_1 = 1
    var_0 = lambda x: x + int_1 """

    # There is only one possible mutation here.
    mutated_test_case_str="""def test_case_0():
    int_0 = 0
    int_1 = 1
    var_0 = lambda x: x + int_0"""

    # Dummy test cluster
    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()

    test_cases, _, _ = deserialize_code_to_testcases(test_case_str, test_cluster)
    assert len(test_cases) == 1
    test_case = test_cases[0]
    orig_test_case = test_case.clone()
    assert len(test_case.statements) == 3
    assert test_case.statements[-1].mutate()
    out_test_case = ExportProvider.get_exporter().export_sequences_to_str([test_case])
    assert out_test_case == mutated_test_case_str
    assert test_case != orig_test_case





# End-to-end test of the ast assign statement's clone and eq. The first has no
# uninterpreted assigns, actually
@pytest.mark.parametrize(
"test_case_str", [
        ("""def test_case_0():
    int_0 = 0
    int_1 = 1
    var_0 = module_0.positional_only(int_0, int_1)
    """),
        ("""def test_case_0():
    int_0 = 0
    int_1 = 1
    var_0 = lambda x: x + int_1
    """)
    ]
)
def test_clone_eq_ast_assign_tc(test_case_str):
    config.configuration.seeding.uninterpreted_statements = True
    config.configuration.seeding.include_partially_parsable = True

    # Dummy test cluster
    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()

    test_cases, _, _ = deserialize_code_to_testcases(test_case_str, test_cluster)
    assert len(test_cases) == 1
    test_case = test_cases[0]
    assert len(test_case.statements) == 3
    clone_test_case = test_case.clone()
    assert clone_test_case is not test_case
    assert clone_test_case.__hash__() == test_case.__hash__()
    assert test_case == clone_test_case


