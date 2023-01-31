#  This file is part of CodaMOSA
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#
from unittest.mock import Mock

import pytest

import pynguin.configuration as config
from pynguin.analyses.seeding import languagemodelseeding
from pynguin.setup.testclustergenerator import TestClusterGenerator


def test_uninterpreted_statement_options():
    config.configuration.seeding.include_partially_parsable = True

    model_mock = Mock()
    model_mock.target_test_case.return_value = """def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = positional_only(var_0, var_2)
    var_4 = lambda x: x
    var_5 = positional_only(var_0, var_4)
    """

    gao_mock = Mock()

    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()
    languagemodelseeding.model = model_mock
    languagemodelseeding.test_cluster = test_cluster

    config.configuration.seeding.uninterpreted_statements = (
        config.UninterpretedStatementUse.NONE
    )
    test_cases = languagemodelseeding.get_targeted_testcase(gao_mock)
    assert len(test_cases) == 1
    test_case = test_cases[0]
    assert len(test_case.statements) == 4

    config.configuration.seeding.uninterpreted_statements = (
        config.UninterpretedStatementUse.ONLY
    )
    test_cases = languagemodelseeding.get_targeted_testcase(gao_mock)
    assert len(test_cases) == 1
    test_case = test_cases[0]
    assert len(test_case.statements) == 6

    config.configuration.seeding.uninterpreted_statements = (
        config.UninterpretedStatementUse.BOTH
    )
    test_cases = languagemodelseeding.get_targeted_testcase(gao_mock)
    assert len(test_cases) == 2
    test_case_lens = {len(test_case.statements) for test_case in test_cases}
    assert test_case_lens == {4, 6}


@pytest.mark.parametrize(
    "return_value,num_statements",
    [
        (
            """def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = [var_1, var_0]
    var_3 = positional_only(var_0, var_2)
    """,
            4,
        ),
    ],
)
def test_uninterpreted_statement_no_reps(return_value, num_statements):
    config.configuration.seeding.include_partially_parsable = True

    model_mock = Mock()
    model_mock.target_test_case.return_value = return_value

    gao_mock = Mock()

    test_cluster = TestClusterGenerator(
        "tests.fixtures.grammar.parameters"
    ).generate_cluster()
    languagemodelseeding.model = model_mock
    languagemodelseeding.test_cluster = test_cluster

    config.configuration.seeding.uninterpreted_statements = (
        config.UninterpretedStatementUse.BOTH
    )
    test_cases = languagemodelseeding.get_targeted_testcase(gao_mock)
    assert len(test_cases) == 1
    test_case_lens = {len(test_case.statements) for test_case in test_cases}
    assert test_case_lens == {num_statements}
