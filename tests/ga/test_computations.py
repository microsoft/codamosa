#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
from unittest.mock import MagicMock, patch

import pytest

import pynguin.ga.computations as ff
import pynguin.ga.testcasechromosome as tcc
import pynguin.ga.testsuitechromosome as tsc
from pynguin.testcase.execution import (
    ExecutionResult,
    ExecutionTrace,
    KnownData,
    TestCaseExecutor,
)


class DummyTestCaseChromosomeComputation(ff.TestCaseChromosomeComputation):
    pass  # pragma: no cover


class DummyTestSuiteChromosomeComputation(ff.TestSuiteChromosomeComputation):
    pass  # pragma: no cover


def test_run_test_case_chromosome_no_result():
    executor = MagicMock()
    result = MagicMock()
    executor.execute.return_value = result
    func = DummyTestCaseChromosomeComputation(executor)
    test_case = tcc.TestCaseChromosome(MagicMock())
    test_case.set_changed(True)
    assert func._run_test_case_chromosome(test_case) == result
    assert test_case.get_last_execution_result() == result


def test_run_test_case_chromosome_has_result():
    executor = MagicMock()
    result = MagicMock()
    executor.execute.return_value = result
    func = DummyTestCaseChromosomeComputation(executor)
    test_case = tcc.TestCaseChromosome(MagicMock())
    test_case.set_changed(False)
    test_case.set_last_execution_result(result)
    assert func._run_test_case_chromosome(test_case) == result
    assert test_case.get_last_execution_result() == result


@pytest.fixture()
def executor_mock():
    return MagicMock(TestCaseExecutor)


@pytest.fixture()
def trace_mock():
    return ExecutionTrace()


@pytest.fixture()
def known_data_mock():
    return KnownData()


def test_test_case_is_minimizing_function(executor_mock):
    func = ff.BranchDistanceTestCaseFitnessFunction(executor_mock, 0)
    assert not func.is_maximisation_function()


def test_test_case_is_maximisation_function(executor_mock):
    func = ff.LineTestSuiteFitnessFunction(executor_mock)
    assert not func.is_maximisation_function()


def test_test_case_compute_fitness_values(known_data_mock, executor_mock, trace_mock):
    tracer = MagicMock()
    tracer.get_known_data.return_value = known_data_mock
    executor_mock.tracer.return_value = tracer
    func = ff.BranchDistanceTestCaseFitnessFunction(executor_mock, 0)
    indiv = MagicMock()
    with patch.object(func, "_run_test_case_chromosome") as run_suite_mock:
        result = ExecutionResult()
        result.execution_trace = trace_mock
        run_suite_mock.return_value = result
        assert func.compute_fitness(indiv) == 0
        run_suite_mock.assert_called_with(indiv)


def test_test_suite_is_maximisation_function(executor_mock):
    func = ff.BranchDistanceTestSuiteFitnessFunction(executor_mock)
    assert not func.is_maximisation_function()


def test_test_suite_compute_branch_distance_fitness_values(
    known_data_mock, executor_mock, trace_mock
):
    tracer = MagicMock()
    tracer.get_known_data.return_value = known_data_mock
    executor_mock.tracer.return_value = tracer
    func = ff.BranchDistanceTestSuiteFitnessFunction(executor_mock)
    indiv = MagicMock()
    with patch.object(func, "_run_test_suite_chromosome") as run_suite_mock:
        result = ExecutionResult()
        result.execution_trace = trace_mock
        run_suite_mock.return_value = [result]
        assert func.compute_fitness(indiv) == 0
        run_suite_mock.assert_called_with(indiv)


def test_test_suite_compute_statements_covered_fitness_values(
    known_data_mock, executor_mock, trace_mock
):
    tracer = MagicMock()
    tracer.get_known_data.return_value = known_data_mock
    executor_mock.tracer.return_value = tracer
    func = ff.LineTestSuiteFitnessFunction(executor_mock)
    indiv = MagicMock()
    with patch.object(func, "_run_test_suite_chromosome") as run_suite_mock:
        result = ExecutionResult()
        result.execution_trace = trace_mock
        run_suite_mock.return_value = [result]
        assert func.compute_fitness(indiv) == 0
        run_suite_mock.assert_called_with(indiv)


def test_run_test_suite_chromosome():
    executor = MagicMock()
    result0 = MagicMock()
    result1 = MagicMock()
    result2 = MagicMock()
    executor.execute.side_effect = [result0, result1]
    ff = DummyTestSuiteChromosomeComputation(executor)
    indiv = tsc.TestSuiteChromosome()
    test_case0 = tcc.TestCaseChromosome(MagicMock())
    test_case0.set_changed(True)
    test_case1 = tcc.TestCaseChromosome(MagicMock())
    test_case1.set_changed(False)
    test_case2 = tcc.TestCaseChromosome(MagicMock())
    test_case2.set_changed(False)
    test_case2.set_last_execution_result(result2)
    indiv.add_test_case_chromosome(test_case0)
    indiv.add_test_case_chromosome(test_case1)
    indiv.add_test_case_chromosome(test_case2)
    assert ff._run_test_suite_chromosome(indiv) == [result0, result1, result2]
    assert test_case0.get_last_execution_result() == result0
    assert test_case1.get_last_execution_result() == result1


def test_run_test_suite_chromosome_cache():
    executor = MagicMock()
    result0 = MagicMock()
    result1 = MagicMock()
    result2 = MagicMock()
    executor.execute.side_effect = [result0, result1]
    func = DummyTestSuiteChromosomeComputation(executor)
    indiv = tsc.TestSuiteChromosome()
    # Executed because it was changed.
    test_case0 = tcc.TestCaseChromosome(MagicMock())
    test_case0.set_changed(True)
    test_case0._computation_cache._fitness_cache = {"foo": "bar"}
    # Executed because it has no result
    test_case1 = tcc.TestCaseChromosome(MagicMock())
    test_case1.set_changed(False)
    test_case1._computation_cache._fitness_cache = {"foo": "bar"}
    # Not executed.
    test_case2 = tcc.TestCaseChromosome(MagicMock())
    test_case2.set_changed(False)
    test_case2._computation_cache._fitness_cache = {"foo": "bar"}
    test_case2.set_last_execution_result(result2)
    indiv.add_test_case_chromosome(test_case0)
    indiv.add_test_case_chromosome(test_case1)
    indiv.add_test_case_chromosome(test_case2)
    assert func._run_test_suite_chromosome(indiv) == [result0, result1, result2]
    assert test_case0._computation_cache._fitness_cache == {}
    assert test_case1._computation_cache._fitness_cache == {}
    assert test_case2._computation_cache._fitness_cache == {"foo": "bar"}