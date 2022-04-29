#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Implements simple constant seeding strategies."""
from __future__ import annotations

import ast
import logging
import os
from abc import abstractmethod
from pathlib import Path
from pkgutil import iter_modules
from typing import TYPE_CHECKING, Any, AnyStr, Optional, Union, cast

from _py_abc import ABCMeta
from ordered_set import OrderedSet
from setuptools import find_packages

import pynguin.configuration as config
import pynguin.ga.testcasechromosome as tcc
import pynguin.testcase.defaulttestcase as dtc
import pynguin.testcase.testfactory as tf
import pynguin.utils.statistics.statistics as stat
from pynguin.analyses.statement_deserializer import StatementDeserializer
from pynguin.ga.computations import compute_branch_coverage
from pynguin.generation.export.pytestexporter import PyTestExporter
from pynguin.testcase.execution import ExecutionResult, TestCaseExecutor
from pynguin.utils import randomness
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

if TYPE_CHECKING:
    from pynguin.setup.testcluster import TestCluster

Types = Union[float, int, str]
logger = logging.getLogger(__name__)


class _ConstantSeeding(metaclass=ABCMeta):
    """An abstract base class for constant seeding strategies."""

    @property
    def has_strings(self) -> bool:
        """Whether we have some strings collected.

        Returns:
            Whether we have some strings collected
        """
        return self.has_constants(str)

    @property
    def has_ints(self) -> bool:
        """Whether we have some ints collected.

        Returns:
            Whether we have some ints collected
        """
        return self.has_constants(int)

    @property
    def has_floats(self) -> bool:
        """Whether we have some floats collected.

        Returns:
            Whether we have some floats collected
        """
        return self.has_constants(float)

    @abstractmethod
    def has_constants(self, type_: type[Types]) -> bool:
        """Returns whether a constant of a given type exists in the pool.

        Args:
            type_: The type of the constant

        Returns:
            Whether a constant of the given type exists  # noqa: DAR202
        """

    @property
    def random_string(self) -> str:
        """Provides a random string from the set of collected strings.

        Returns:
            A random string
        """
        return cast(str, self.random_element(str))

    @property
    def random_int(self) -> int:
        """Provides a random int from the set of collected ints.

        Returns:
            A random int
        """
        return cast(int, self.random_element(int))

    @property
    def random_float(self) -> float:
        """Provides a random float from the set of collected floats.

        Returns:
            A random float
        """
        return cast(float, self.random_element(float))

    @abstractmethod
    def random_element(self, type_: type[Types]) -> Types:
        """Provides a random element of the given type

        Args:
            type_: The given type

        Returns:
            A random element of the given type
        """


class _StaticConstantSeeding(_ConstantSeeding):
    """A simple static constant seeding strategy.

    Extracts all constants from a set of modules by using an AST visitor.
    """

    def __init__(self) -> None:
        self._constants: dict[type[Types], OrderedSet[Types]] = {
            int: OrderedSet(),
            float: OrderedSet(),
            str: OrderedSet(),
        }

    @staticmethod
    def _find_modules(project_path: str | os.PathLike) -> OrderedSet[str]:
        modules: OrderedSet[str] = OrderedSet()
        for package in find_packages(
            project_path,
            exclude=[
                "*.tests",
                "*.tests.*",
                "tests.*",
                "tests",
                "test",
                "test.*",
                "*.test.*",
                "*.test",
            ],
        ):
            package_name = package.replace(".", "/")
            pkg_path = f"{project_path}/{package_name}"
            for info in iter_modules([pkg_path]):
                if not info.ispkg:
                    name = info.name.replace(".", "/")
                    module = f"{package_name}/{name}.py"
                    module_path = Path(project_path) / Path(module)
                    if module_path.exists() and module_path.is_file():
                        # Consider only Python files for constant seeding, as the
                        # seeding relies on the availability of an AST.
                        modules.add(module)
        return modules

    def collect_constants(
        self, project_path: str | os.PathLike
    ) -> dict[type[Types], OrderedSet[Types]]:
        """Collect all constants for a given project.

        Args:
            project_path: The path to the project's root

        Returns:
            A dict of type to set of constants
        """
        assert self._constants is not None
        collector = _ConstantCollector()
        for module in self._find_modules(project_path):
            with open(
                os.path.join(project_path, module), encoding="utf-8"
            ) as module_file:
                try:
                    tree = ast.parse(module_file.read())
                    collector.visit(tree)
                except BaseException as exception:  # pylint: disable=broad-except
                    logger.exception("Cannot collect constants: %s", exception)
        self._constants = collector.constants
        return self._constants

    def has_constants(self, type_: type[Types]) -> bool:
        assert self._constants is not None
        return len(self._constants[type_]) > 0

    def random_element(self, type_: type[Types]) -> Types:
        assert self._constants is not None
        return randomness.choice(tuple(self._constants[type_]))


# pylint: disable=invalid-name, missing-function-docstring
class _ConstantCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self._constants: dict[type[Types], OrderedSet[Types]] = {
            float: OrderedSet(),
            int: OrderedSet(),
            str: OrderedSet(),
        }
        self._string_expressions: OrderedSet[str] = OrderedSet()

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            self._constants[str].add(node.value)
        elif isinstance(node.value, float):
            self._constants[float].add(node.value)
        elif isinstance(node.value, int):
            self._constants[int].add(node.value)
        return self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> Any:
        return self._visit_doc_string(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return self._visit_doc_string(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        return self._visit_doc_string(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self._visit_doc_string(node)

    def _visit_doc_string(self, node: ast.AST) -> Any:
        if docstring := ast.get_docstring(node):
            self._string_expressions.add(docstring)
        return self.generic_visit(node)

    @property
    def constants(self) -> dict[type[Types], OrderedSet[Types]]:
        """Provides the collected constants.

        Returns:
            The collected constants
        """
        self._remove_docstrings()
        return self._constants

    def _remove_docstrings(self) -> None:
        self._constants[str] -= self._string_expressions  # type: ignore


class DynamicConstantSeeding(_ConstantSeeding):
    """Provides a dynamic pool and methods to add and retrieve values.

    The methods in this class are added to the module under test during an instruction
    phase before the main algorithm is executed. During this instruction phase,
    bytecode is added to the module under test which executes the methods adding
    values to the dynamic pool. The instrumentation is implemented in the module
    dynamicseedinginstrumentation.py.

    During the test generation process when a new value of one of the supported types
    is needed, this module provides methods to get values from the dynamic pool
    instead of randomly generating a new one.
    """

    _string_functions_lookup = {
        "isalnum": lambda value: f"{value}!" if value.isalnum() else "isalnum",
        "islower": lambda value: value.upper() if value.islower() else value.lower(),
        "isupper": lambda value: value.lower() if value.isupper() else value.upper(),
        "isdecimal": lambda value: "non_decimal" if value.isdecimal() else "0123456789",
        "isalpha": lambda value: f"{value}1" if value.isalpha() else "isalpha",
        "isdigit": lambda value: f"{value}_" if value.isdigit() else "0",
        "isidentifier": lambda value: f"{value}!"
        if value.isidentifier()
        else "is_Identifier",
        "isnumeric": lambda value: f"{value}A" if value.isnumeric() else "012345",
        "isprintable": lambda value: f"{value}{os.linesep}"
        if value.isprintable()
        else "is_printable",
        "isspace": lambda value: f"{value}a" if value.isspace() else "   ",
        "istitle": lambda value: f"{value} AAA" if value.istitle() else "Is Title",
    }

    def __init__(self):
        self._dynamic_pool: dict[type[Types], OrderedSet[Types]] = {
            int: OrderedSet(),
            float: OrderedSet(),
            str: OrderedSet(),
        }

    def reset(self) -> None:
        """Delete all currently stored dynamic constants"""
        for elem in self._dynamic_pool.values():
            elem.clear()

    def has_constants(self, type_: type[Types]) -> bool:
        assert type_ in self._dynamic_pool
        return len(self._dynamic_pool[type_]) > 0

    def random_element(self, type_: type[Types]) -> Types:
        return randomness.choice(tuple(self._dynamic_pool[type_]))

    def add_value(self, value: Types):
        """Adds the given value to the corresponding set of the dynamic pool.

        Args:
            value: The value to add.
        """
        if isinstance(
            value, bool
        ):  # needed because True and False are accepted as ints otherwise
            return
        if type(value) in self._dynamic_pool:
            self._dynamic_pool[type(value)].add(value)

    def add_value_for_strings(self, value: str, name: str):
        """Add a value of a string.

        Args:
            value: The value
            name: The string
        """
        if not isinstance(value, str):
            return
        self._dynamic_pool[str].add(value)
        self._dynamic_pool[str].add(self._string_functions_lookup[name](value))


class _InitialPopulationSeeding:
    """Class for seeding the initial population with previously existing testcases."""

    def __init__(self):
        self._testcases: list[dtc.DefaultTestCase] = []
        self._test_cluster: TestCluster
        self._executor: Optional[TestCaseExecutor]
        self._sample_with_replacement: bool = True

    @property
    def test_cluster(self) -> TestCluster:
        """Provides the test cluster.

        Returns:
            The test cluster
        """
        return self._test_cluster

    @test_cluster.setter
    def test_cluster(self, test_cluster: TestCluster):
        self._test_cluster = test_cluster

    @property
    def executor(self) -> Optional[TestCaseExecutor]:
        """Provides the test executor.

        Returns:
            The test executor
        """
        return self._executor

    @executor.setter
    def executor(self, executor: Optional[TestCaseExecutor]):
        self._executor = executor

    @property
    def sample_with_replacement(self) -> bool:
        """Provides whether sampling with replacement is performed.

        Returns:
            Whether sampling with replacement is performed
        """
        return self._sample_with_replacement

    @sample_with_replacement.setter
    def sample_with_replacement(self, sample_with_replacement: bool):
        self._sample_with_replacement = sample_with_replacement

    @staticmethod
    def get_ast_tree(module_path: AnyStr | os.PathLike[AnyStr]) -> ast.Module | None:

        """Returns the ast tree from a module

        Args:
            module_path: The path to the project's root

        Returns:
            The ast tree of the given module.
        """
        module_name = config.configuration.module_name.rsplit(".", maxsplit=1)[-1]
        logger.debug("Module name: %s", module_name)
        result: list[AnyStr] = []
        for root, _, files in os.walk(module_path):
            for name in files:
                assert isinstance(name, str)
                if module_name in name and "test_" in name:
                    result.append(os.path.join(root, name))
                    break
        try:
            if len(result) > 0:
                logger.debug("Module name found: %s", result[0])
                stat.track_output_variable(RuntimeVariable.SuitableTestModule, True)
                with open(result[0], encoding="utf-8") as module_file:
                    return ast.parse(module_file.read())
            else:
                logger.debug("No suitable test module found.")
                stat.track_output_variable(RuntimeVariable.SuitableTestModule, False)
                return None
        except BaseException as exception:  # pylint: disable=broad-except
            logger.exception("Cannot read module: %s", exception)
            stat.track_output_variable(RuntimeVariable.SuitableTestModule, False)
            return None

    def collect_testcases(self, module_path: AnyStr | os.PathLike[AnyStr]) -> None:
        """Collect all test cases from a module.

        Args:
            module_path: Path to the module to collect the test cases from
        """
        tree = self.get_ast_tree(module_path)
        if tree is None:
            config.configuration.seeding.initial_population_seeding = False
            logger.info("Provided testcases are not used.")
            return
        transformer = AstToTestCaseTransformer(
            self._test_cluster,
            config.configuration.test_case_output.assertion_generation
            != config.AssertionGenerator.NONE,
        )
        transformer.visit(tree)
        self._testcases = transformer.testcases
        stat.track_output_variable(RuntimeVariable.FoundTestCases, len(self._testcases))
        if not self._testcases:
            config.configuration.seeding.initial_population_seeding = False
            logger.info("None of the provided test cases can be parsed.")
        else:
            logger.info(
                "Number successfully collected test cases: %s", len(self._testcases)
            )
            exporter = PyTestExporter(wrap_code=False)
            logger.info(
                "Imported test cases:\n %s",
                exporter.export_sequences_to_str(self._testcases),  # type: ignore
            )
        stat.track_output_variable(
            RuntimeVariable.CollectedTestCases, len(self._testcases)
        )
        stat.track_output_variable(
            RuntimeVariable.ParsableStatements, transformer.total_statements
        )
        stat.track_output_variable(
            RuntimeVariable.ParsedStatements, transformer.total_parsed_statements
        )
        self._remove_no_coverage_testcases()
        self._mutate_testcases_initially()

    def _mutate_testcases_initially(self):
        """Mutates the initial population."""
        test_factory = tf.TestFactory(self.test_cluster)
        for _ in range(0, config.configuration.seeding.initial_population_mutations):
            for testcase in self._testcases:
                testcase_wrapper = tcc.TestCaseChromosome(testcase, test_factory)
                testcase_wrapper.mutate()
                if not testcase_wrapper.test_case.statements:
                    self._testcases.remove(testcase)

    def _remove_no_coverage_testcases(self):
        """Removes testcases whose coverage is"""
        if config.configuration.seeding.remove_testcases_without_coverage:
            assert self._executor is not None
            num_removed_test_cases = 0
            tracer = self._executor.tracer
            import_coverage = compute_branch_coverage(
                tracer.import_trace, tracer.get_known_data()
            )
            for testcase in self._testcases:
                result: ExecutionResult = self._executor.execute(testcase)
                coverage = compute_branch_coverage(
                    result.execution_trace, tracer.get_known_data()
                )
                if coverage <= import_coverage:
                    self._testcases.remove(testcase)
                num_removed_test_cases += 1
            logger.info(
                "Number test cases removed because they have no coverage: %s",
                num_removed_test_cases,
            )
            stat.track_output_variable(
                RuntimeVariable.NoCoverageTestCases, num_removed_test_cases
            )

    @property
    def seeded_testcase(self) -> dtc.DefaultTestCase:
        """Provides a random seeded test case.

        Returns:
            A random test case
        """
        if self._sample_with_replacement:
            return self._testcases[randomness.next_int(0, len(self._testcases))]
        test_case_idx = randomness.next_int(0, len(self._testcases))
        return self._testcases.pop(test_case_idx)

    @property
    def has_tests(self) -> bool:
        """Whether or not test cases have been found.

        Returns:
            Whether or not test cases have been found
        """
        return len(self._testcases) > 0


# pylint: disable=invalid-name, missing-function-docstring, too-many-instance-attributes
class AstToTestCaseTransformer(ast.NodeVisitor):
    """An AST NodeVisitor that tries to convert an AST into our internal
    test case representation."""

    def __init__(self, test_cluster: TestCluster, create_assertions: bool):
        self._deserializer = StatementDeserializer(test_cluster)
        self._current_parsable: bool = True
        self._testcases: list[dtc.DefaultTestCase] = []
        self._number_found_testcases: int = 0
        self._create_assertions = create_assertions
        self.total_statements = 0
        self.total_parsed_statements = 0
        self._current_parsed_statements = 0
        self._current_max_num_statements = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        # Don't include non-test functions as tests.
        if not node.name.startswith("test_") and not node.name.startswith("seed_test_"):
            return
        self._number_found_testcases += 1
        self._deserializer.reset()
        self._current_parsable = True
        self._current_parsed_statements = 0
        self._current_max_num_statements = len(
            [e for e in node.body if not isinstance(e, ast.Assert)]
        )
        self.generic_visit(node)
        self.total_statements += self._current_max_num_statements
        self.total_parsed_statements += self._current_parsed_statements
        current_testcase = self._deserializer.get_test_case()
        if self._current_parsable:
            self._testcases.append(current_testcase)
            logger.info("Successfully imported %s.", node.name)
        else:
            if (
                self._current_parsed_statements > 0
                and config.configuration.seeding.include_partially_parsable
            ):
                logger.info(
                    "Partially parsed %s. Retrieved %s/%s statements.",
                    node.name,
                    self._current_parsed_statements,
                    self._current_max_num_statements,
                )
                self._testcases.append(current_testcase)
            else:
                logger.info("Failed to parse %s.", node.name)

    def visit_Assign(self, node: ast.Assign) -> Any:
        if self._current_parsable:
            if self._deserializer.add_assign_stmt(node):
                self._current_parsed_statements += 1
            else:
                self._current_parsable = False

    def visit_Assert(self, node: ast.Assert) -> Any:
        if self._current_parsable and self._create_assertions:
            self._deserializer.add_assert_stmt(node)

    @property
    def testcases(self) -> list[dtc.DefaultTestCase]:
        """Provides the testcases that could be generated from the given AST.
        It is possible that not every aspect of the AST could be transformed
        to our internal representation.

        Returns:
            The generated testcases.
        """
        return self._testcases


initialpopulationseeding = _InitialPopulationSeeding()
static_constant_seeding = _StaticConstantSeeding()
dynamic_constant_seeding = DynamicConstantSeeding()
