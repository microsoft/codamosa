#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Implements simple constant seeding strategies."""
from __future__ import annotations

import ast
import inspect
import logging
import os
from abc import abstractmethod
from pathlib import Path
from pkgutil import iter_modules
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from _py_abc import ABCMeta
from ordered_set import OrderedSet
from setuptools import find_packages

import pynguin.configuration as config
import pynguin.ga.testcasechromosome as tcc
import pynguin.ga.testsuitechromosome as tsc
import pynguin.testcase.defaulttestcase as dtc
import pynguin.testcase.testcase as tc
import pynguin.testcase.testfactory as tf
import pynguin.utils.statistics.statistics as stat
from pynguin.analyses.codedeserializer import deserialize_code_to_testcases
from pynguin.ga.computations import compute_branch_coverage
from pynguin.generation.export.pytestexporter import PyTestExporter
from pynguin.languagemodels.outputfixers import fixup_imports
from pynguin.testcase.execution import ExecutionResult, TestCaseExecutor
from pynguin.testcase.statement import ASTAssignStatement
from pynguin.utils import randomness
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
)
from pynguin.utils.report import LineAnnotation, get_coverage_report
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

if TYPE_CHECKING:
    from pynguin.languagemodels.model import _OpenAILanguageModel
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


# pylint: disable=R0902
class _LargeLanguageModelSeeding:
    """Class for seeding the initial population with test cases generated by a large
    language model."""

    def __init__(self):
        self._prompt_gaos: Optional[Dict[GenericCallableAccessibleObject, int]] = None
        self._sample_with_replacement: bool = True
        self._max_samples_per_prompt: int = 1
        self._parsed_statements = 0
        self._parsable_statements = 0
        self._uninterp_statements = 0

    @property
    def model(self) -> _OpenAILanguageModel:
        """Provides the model wrapper object we query from

        Returns:
            The large language model wrapper
        """
        return self._model

    @model.setter
    def model(self, model: _OpenAILanguageModel):
        self._model = model

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

    @property
    def seeded_testcase(self) -> Optional[tc.TestCase]:
        """
        Generate a new test case. Prompt the language model with a generic accessible
        object to test.

        Returns:
            A new generated test case, or None if a test case could not be parsed
        """
        assert self._prompt_gaos is not None
        assert len(self._prompt_gaos) > 0
        prompt_gao = randomness.choice(list(self._prompt_gaos.keys()))
        if not self._sample_with_replacement:
            self._prompt_gaos[prompt_gao] -= 1
            if self._prompt_gaos[prompt_gao] == 0:
                self._prompt_gaos.pop(prompt_gao)
        testcases = self.get_targeted_testcase(prompt_gao)
        if len(testcases) > 0:
            return testcases[0]
        return None

    def get_random_targeted_testcase(self) -> Sequence[tc.TestCase]:
        """
        Generate a new test case (or multiple) aimed at a gao to be selected randomly

        Returns:
            A sequence of generated test cases
        """

        if self._prompt_gaos is None:
            self._setup_gaos()
            assert self._prompt_gaos is not None
        prompt_gao = randomness.choice(list(self._prompt_gaos.keys()))
        return self.get_targeted_testcase(prompt_gao)

    def get_targeted_testcase(
        self, prompt_gao: GenericCallableAccessibleObject, context=""
    ) -> Sequence[tc.TestCase]:
        """
        Generate a new test case aimed at prompt_gao

        Args:
            prompt_gao: the GenericCallableAccessibleObject to target
            context: any additional context to pass

        Returns:
            A sequence of generated test cases
        """
        str_test_case = self._model.target_test_case(prompt_gao, context=context)
        use_uninterp_tuple = config.configuration.seeding.uninterpreted_statements.value
        ret_testcases: Set[tc.TestCase] = set()
        for use_uninterp in use_uninterp_tuple:
            logger.debug("Codex-generated testcase:\n%s", str_test_case)
            (
                testcases,
                parsed_statements,
                parsable_statements,
            ) = deserialize_code_to_testcases(
                str_test_case, self._test_cluster, use_uninterp
            )
            if len(testcases) > 0:
                for testcase in testcases:
                    exporter = PyTestExporter(wrap_code=False)
                    logger.debug(
                        "Imported test case (%i/%i statements parsed):\n %s",
                        parsed_statements,
                        parsable_statements,
                        exporter.export_sequences_to_str([testcase]),
                    )

                    self._parsable_statements += parsable_statements
                    self._parsed_statements += parsed_statements
                    self._uninterp_statements += len(
                        [
                            stmt
                            for stmt in testcase.statements
                            if isinstance(stmt, ASTAssignStatement)
                        ]
                    )
                    stat.track_output_variable(
                        RuntimeVariable.ParsableStatements, self._parsable_statements
                    )
                    stat.track_output_variable(
                        RuntimeVariable.ParsedStatements, self._parsed_statements
                    )
                    stat.track_output_variable(
                        RuntimeVariable.UninterpStatements, self._uninterp_statements
                    )
            ret_testcases.update(testcases)
        return list(ret_testcases)

    @property
    def has_tests(self) -> bool:
        """Whether or not test cases are left to generate

        Returns:
            Whether or not test cases have been found
        """
        if self._prompt_gaos is None:
            self._setup_gaos()
            assert self._prompt_gaos is not None
        return len(self._prompt_gaos) > 0

    def _setup_gaos(self):
        """Sets up the prompt gaos if they are unset."""
        self._prompt_gaos = {
            gao: self._max_samples_per_prompt  # type: ignore
            for gao in self._test_cluster.accessible_objects_under_test
            if issubclass(type(gao), GenericCallableAccessibleObject)
        }

    def target_uncovered_functions(
        self,
        test_suite: tsc.TestSuiteChromosome,
        num_samples: int,
        resources_left: Callable[[], bool],
    ) -> List[tc.TestCase]:

        # pylint: disable=R0914,R0912
        """Generate test cases for functions that are less covered by `test_suite`

        Args:
            test_suite: current best test suite
            num_samples: number of test cases to sample
            resources_left: a callable that returns true if there are resources left
                in the search algorithm

        Returns:
            a list of Codex-generated test cases.
        """
        assert self.executor is not None
        if self._prompt_gaos is None:
            self._setup_gaos()
            assert self._prompt_gaos is not None

        line_annotations: List[LineAnnotation] = get_coverage_report(
            test_suite,
            self.executor,
            config.configuration.statistics_output.coverage_metrics,
        ).line_annotations

        def coverage_in_range(start_line: int, end_line: int) -> Tuple[int, int]:
            """Helper coverage to determine the coverage of consecutive lines.

            Args:
                start_line: first line to consider, inclusive
                end_line: last line to consider, inclusive

            Returns:
                the total number of covered elements (branches, lines) in the line
                range, as well as the total number of coverable elements in that range.
            """
            total_coverage_points = 0
            covered_coverage_points = 0
            for line_annot in line_annotations:
                if start_line <= line_annot.line_no <= end_line:
                    total_coverage_points += line_annot.total.existing
                    covered_coverage_points += line_annot.total.covered
            return covered_coverage_points, total_coverage_points

        ordered_gaos: List[GenericCallableAccessibleObject] = []
        ordered_selection_probabilities: List[float] = []

        for gao in self._prompt_gaos.keys():
            if isinstance(gao, GenericCallableAccessibleObject):
                ordered_gaos.append(gao)
                try:
                    source_lines, start_line = inspect.getsourcelines(gao.callable)
                    covered, total = coverage_in_range(
                        start_line, start_line + len(source_lines) - 1
                    )
                    if total > 0:
                        ordered_selection_probabilities.append(1 - (covered / total))
                    else:
                        ordered_selection_probabilities.append(0)
                except (TypeError, OSError):
                    ordered_selection_probabilities.append(0)

        denominator = sum(ordered_selection_probabilities)
        if denominator == 0:
            # All the top-level callable functions are fully covered. I guess
            # just do some random sampling?
            ordered_selection_probabilities = [
                1 / len(ordered_selection_probabilities)
            ] * len(ordered_selection_probabilities)
        else:
            ordered_selection_probabilities = [
                p / denominator for p in ordered_selection_probabilities
            ]

        if config.configuration.codamosa.test_case_context in (
            config.TestCaseContext.SMALLEST,
            config.TestCaseContext.RANDOM,
        ):
            exporter = PyTestExporter(wrap_code=False)
            ctx_test_cases = [
                (
                    fixup_imports(exporter.export_sequences_to_str([tcc.test_case])),
                    tcc.size(),
                )
                for tcc in test_suite.test_case_chromosomes
                if tcc.size() > 0
            ]
            ctx_test_cases.sort(key=lambda x: x[1])

        targeted_test_cases: List[tc.TestCase] = []
        for gao in randomness.choices(
            ordered_gaos, weights=ordered_selection_probabilities, k=num_samples
        ):
            if not resources_left():
                break
            if (
                config.configuration.codamosa.test_case_context
                == config.TestCaseContext.SMALLEST
                and len(ctx_test_cases) > 0
            ):
                context = ctx_test_cases[0][0] + "\n\n"
            elif (
                config.configuration.codamosa.test_case_context
                == config.TestCaseContext.RANDOM
                and len(ctx_test_cases) > 0
            ):
                context = randomness.choice(ctx_test_cases)[0] + "\n\n"
            else:
                context = ""
            test_cases = self.get_targeted_testcase(gao, context)
            targeted_test_cases.extend(test_cases)
        return targeted_test_cases


class _InitialPopulationSeeding:
    """Class for seeding the initial population with previously existing testcases."""

    def __init__(self):
        self._testcases: list[dtc.DefaultTestCase] = []
        self._test_cluster: TestCluster
        self._executor: Optional[TestCaseExecutor]
        self._sample_with_replacement: bool = True
        stat.track_output_variable(RuntimeVariable.ParsableStatements, 0)
        stat.track_output_variable(RuntimeVariable.ParsedStatements, 0)
        stat.track_output_variable(RuntimeVariable.UninterpStatements, 0)

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
    def get_test_code(module_path: AnyStr | os.PathLike[AnyStr]) -> str | None:
        """Returns test code for the module under test present in the folder
        at `module_path`.

        Args:
            module_path: The path to the project's root

        Returns:
            The code of the given module.
        """
        module_name = config.configuration.module_name.rsplit(".", maxsplit=1)[-1]
        logger.debug("Module name: %s", module_name)
        result: list[AnyStr] = []
        for root, _, files in os.walk(module_path):
            for name in files:
                assert isinstance(name, str)
                if module_name in name and "test_" in name and name.endswith(".py"):
                    result.append(os.path.join(root, name))
                    break
        try:
            if len(result) > 0:
                logger.debug("Module name found: %s", result[0])
                stat.track_output_variable(RuntimeVariable.SuitableTestModule, True)
                with open(result[0], encoding="utf-8") as module_file:
                    return module_file.read()
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

        Raises:
            NotImplementedError: if using option --uninterpreted_statements BOTH

        """
        code = self.get_test_code(module_path)
        if code is None:
            config.configuration.seeding.initial_population_seeding = False
            logger.info("Provided testcases are not used.")
            return
        try:
            use_uninterp_tuple = (
                config.configuration.seeding.uninterpreted_statements.value
            )
            if len(use_uninterp_tuple) > 1:
                raise NotImplementedError(
                    "--uninterpreted_statements BOTH not supported with"
                    " initial population seeding"
                )
            (
                test_cases,
                parsed_statements,
                parsable_statements,
            ) = deserialize_code_to_testcases(
                code, self._test_cluster, use_uninterp_tuple[0]
            )
        # In case ast.parse throws
        except BaseException as exception:  # pylint: disable=broad-except
            logger.exception("Cannot read module: %s", exception)
            stat.track_output_variable(RuntimeVariable.SuitableTestModule, False)
            logger.info("Provided testcases are not used.")
            return

        self._testcases = test_cases
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
            RuntimeVariable.ParsableStatements, parsable_statements
        )
        stat.track_output_variable(RuntimeVariable.ParsedStatements, parsed_statements)
        stat.track_output_variable(
            RuntimeVariable.UninterpStatements,
            len(
                [
                    stmt
                    for testcase in self._testcases
                    for stmt in testcase.statements
                    if isinstance(stmt, ASTAssignStatement)
                ]
            ),
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
            exporter = PyTestExporter(wrap_code=False)
            import_coverage = compute_branch_coverage(
                tracer.import_trace, tracer.get_known_data()
            )
            idxs_to_remove = []
            for idx, testcase in enumerate(self._testcases):
                result: ExecutionResult = self._executor.execute(testcase)
                coverage = compute_branch_coverage(
                    result.execution_trace, tracer.get_known_data()
                )
                if coverage <= import_coverage:
                    idxs_to_remove.append(idx)
                    num_removed_test_cases += 1
                logger.debug(
                    "Test case:\n %s\n has coverage %s vs. import coverage %f",
                    exporter.export_sequences_to_str([testcase]),
                    coverage,
                    import_coverage,
                )
            self._testcases = [
                tc
                for idx, tc in enumerate(self._testcases)
                if idx not in idxs_to_remove
            ]
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


languagemodelseeding = _LargeLanguageModelSeeding()
initialpopulationseeding = _InitialPopulationSeeding()
static_constant_seeding = _StaticConstantSeeding()
dynamic_constant_seeding = DynamicConstantSeeding()
