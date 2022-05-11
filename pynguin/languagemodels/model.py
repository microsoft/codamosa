#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
import ast
import inspect
import json
import logging
import string
from typing import Dict, List

import requests

import pynguin.testcase.testcase as tc
from pynguin.generation.export.pytestexporter import PyTestExporter
from pynguin.languagemodels.functionplaceholderadder import add_placeholder
from pynguin.languagemodels.outputfixers import rewrite_tests
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)

logger = logging.getLogger(__name__)


def approx_number_tokens(line: str):
    """
    We want to estimate the number of tokens in a line of code.
    From https://beta.openai.com/tokenizer it looks like roughly
    sequential whitespace becomes a single token, and a new token
    is created when character "class" changes.

    Args:
        line: a line to get the approximate number of tokens for

    Returns:
        an approximate number of tokens in `line`

    """

    def char_type(c):
        if c in string.ascii_letters:
            return "letter"
        elif c in string.digits:
            return "digit"
        elif c in string.punctuation:
            return "punctuation"
        elif c in string.whitespace:
            return "whitespace"
        else:
            return "other"

    toks = []
    last_type = "other"
    cur_tok = ""
    for c in line:
        if char_type(c) != last_type:
            toks.append(cur_tok)
            last_type = char_type(c)
            cur_tok = c
        else:
            cur_tok += c
    if len(cur_tok) > 0:
        toks.append(cur_tok)
    return len(toks)


class _OpenAILanguageModel:
    """
    An interface for an OpenAI language model to generate/mutate tests as natural language.
    TODO(clemieux): starting by implementing a concrete instance of this.
    """

    def __init__(self):
        self._test_src: str
        self._authorization_key: str
        self._complete_model: str
        self._edit_model: str
        # TODO(clemieux): make configurable; adding a fudge factor
        self._max_query_len = 4096 - 200
        # TODO(clemieux): make configurable
        self._temperature = 1
        self._token_len_cache = {}
        self.num_codex_calls = 0

    @property
    def test_src(self) -> str:
        """Provides the source of the module under test

        Returns:
            The source of the module under test
        """
        return self._test_src

    @test_src.setter
    def test_src(self, test_src: str):
        self._test_src = test_src

    @property
    def authorization_key(self) -> str:
        """Provides the authorization key used to query the model

        Returns:
            The organization id
        """
        return self._authorization_key

    @authorization_key.setter
    def authorization_key(self, authorization_key: str):
        self._authorization_key = authorization_key

    @property
    def complete_model(self) -> str:
        """Provides the name of the model used for completion tasks

        Returns:
            The name of the model used for completion tasks
        """
        return self._complete_model

    @complete_model.setter
    def complete_model(self, complete_model: str):
        self._complete_model = complete_model

    @property
    def edit_model(self) -> str:
        """Provides the name of the model used for editing tasks

        Returns:
            The name of the model used for editing tasks
        """
        return self._edit_model

    @edit_model.setter
    def edit_model(self, edit_model: str):
        self._edit_model = edit_model

    def _get_maximal_source_context(
        self, start_line: int = -1, end_line: int = -1, used_tokens: int = 0
    ):
        """Tries to get the maximal source context that includes start_line to end_line but
        remains under the threshold.

        Args:
            start_line: the start line that should be included
            end_line: the end line that should be included
            used_tokens: the number of tokens to reduce the max allowed by

        Returns:
            as many lines from the source as possible that fit in max_context.
        """

        split_src = self._test_src.split("\n")
        num_lines = len(split_src)

        if end_line == -1:
            end_line = num_lines

        # Return everything if you can
        if (
            sum([self._get_num_tokens_at_line(i) for i in range(1, num_lines + 1)])
            < self._max_query_len
        ):
            return self._test_src

        if (
            sum([self._get_num_tokens_at_line(i) for i in range(1, end_line + 1)])
            < self._max_query_len
        ):
            return "\n".join(split_src[0:end_line])

        # Otherwise greedily take the lines preceding the end line
        cumul_len_of_prefix: List[int] = []
        cumul_len: int = 0
        for i in reversed(range(1, end_line + 1)):
            tok_len = self._get_num_tokens_at_line(i)
            cumul_len += tok_len
            cumul_len_of_prefix.insert(0, cumul_len)

        context_start_line = 0
        for idx, cumul_tok_len in enumerate(cumul_len_of_prefix):
            line_num = idx + 1
            if cumul_tok_len < self._max_query_len - used_tokens:
                context_start_line = line_num
                break

        return "\n".join(split_src[context_start_line:end_line])

    def _call_mutate(self, function_to_mutate: str) -> str:
        """Asks the model to fill in the `??` in the given function

        Args:
            function_to_mutate: a string containing code with a `??` placeholder

        Returns:
            the result of calling the model to edit the given code
        """
        context = self._get_maximal_source_context(
            used_tokens=approx_number_tokens(function_to_mutate)
        )

        url = f"https://api.openai.com/v1/engines/{self.edit_model}/edits"

        payload = {
            "input": context + "\n" + function_to_mutate,
            "instruction": "Fill in the ??",
            "temperature": self._temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._authorization_key}",
        }
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        self.num_codex_calls += 1
        if res.status_code != 200:
            logger.error("Failed to call for edit:\n%s", res.json())
            return ""
        return res.json()["choices"][0]["text"]

    def _call_completion(
        self, function_header: str, context_start: int, context_end: int
    ) -> str:
        """Asks the model to provide a completion of the given function header,
        with the additional context of the target function definition.

        Args:
            function_header: a string containing a def statement to be completed
            context_start: the start line of context that must be included
            context_end: the end line of context that must be included

        Returns:
            the result of calling the model to complete the function header.
        """
        context = self._get_maximal_source_context(context_start, context_end)

        url = f"https://api.openai.com/v1/engines/{self.complete_model}/completions"
        # We want to stop the generation before it spits out a bunch of other tests,
        # because that slows things down
        payload = {
            "prompt": context + "\n" + function_header,
            "max_tokens": 200,
            "temperature": self._temperature,
            "stop": ["\n# Unit test for", "\ndef ", "\nclass "],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._authorization_key}",
        }
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        self.num_codex_calls += 1
        if res.status_code != 200:
            logger.error("Failed to call for completion:\n%s", res.json())
            return ""
        return res.json()["choices"][0]["text"]

    def _get_num_tokens_at_line(self, line_num: int) -> int:
        """Get the approximate number of tokens for the source file at line_num.

        Args:
            line_num: the line number to get the number of tokens for

        Returns:
            the approximate number of tokens
        """
        if len(self._token_len_cache) == 0:
            self._token_len_cache = {
                i + 1: approx_number_tokens(line)
                for i, line in enumerate(self._test_src.split("\n"))
            }
        return self._token_len_cache[line_num]

    def mutate_test_case(self, test_case: tc.TestCase) -> str:
        """Calls a large language model to mutate the test case `tc`

        Args:
            test_case: the tc.TestCase object to mutate.

        Returns:
            the mutated test case as a string
        """
        exporter = PyTestExporter(wrap_code=False)
        str_test_case = exporter.export_sequences_to_str([test_case])
        ast_test_case_module = ast.parse(str_test_case)
        function_with_placeholder = add_placeholder(ast_test_case_module, True)
        mutated = self._call_mutate(function_with_placeholder)

        test_start_idxs = [
            i
            for i, line in enumerate(mutated.split("\n"))
            if line.startswith("def test_")
        ]
        if len(test_start_idxs) == 0:
            print("no testsss....")
            return str_test_case
        mutated_test_as_str = ["\n".join(mutated.split("\n")[test_start_idxs[0] :])]
        mutated_tests_fixed: Dict[str, str] = rewrite_tests(mutated_test_as_str)
        return list(mutated_tests_fixed.values())[0]
        # # TODO: how to transform back into a test case?
        # mutated_str_test_case = ""
        # try:
        #     for elem in ast.parse(mutated).body:
        #         if isinstance(elem, ast.FunctionDef) and elem.name.startswith("test_"):
        #             mutated_str_test_case = ast.unparse(elem)
        #             break
        # except SyntaxError:
        #     print(f"!!!failed to parse \n{mutated}")
        #
        # return mutated_str_test_case

    def target_test_case(self, gao: GenericCallableAccessibleObject) -> str:
        """Provides a test case targeted to the function/method/constructor
        specified in `gao`

        Args:
            gao: a GenericCallableAccessibleObject to target the test to

        Returns:
            A generated test case as natural language.

        """
        if gao.is_method():
            method_gao: GenericMethod = gao  # type: ignore
            function_header = (
                f"# Unit test for method {method_gao.method_name} of "
                f"class {method_gao.owner.__name__}\n"  # type: ignore
                f"def test_{method_gao.owner.__name__}"
                f"_{method_gao.method_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(method_gao.owner)  # type: ignore
                end_line = start_line + len(source_lines) - 1
                if (
                    sum(
                        [
                            self._get_num_tokens_at_line(i)
                            for i in range(start_line, end_line + 1)
                        ]
                    )
                    > self._max_query_len
                ):
                    source_lines, start_line = inspect.getsourcelines(method_gao.owner)  # type: ignore
                    end_line = start_line + len(source_lines) - 1
            except OSError:
                start_line, end_line = -1, -1
        elif gao.is_function():
            fn_gao: GenericFunction = gao  # type: ignore
            function_header = (
                f"# Unit test for function {fn_gao.function_name}"
                f"\ndef test_{fn_gao.function_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(fn_gao.callable)
                end_line = start_line + len(source_lines) - 1
            except OSError:
                start_line, end_line = -1, -1
        elif gao.is_constructor():
            constructor_gao: GenericConstructor = gao  # type: ignore
            class_name = constructor_gao.generated_type().__name__  # type: ignore
            function_header = (
                f"# Unit test for constructor of class {class_name}"
                f"\ndef test_{class_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(
                    constructor_gao.generated_type()  # type: ignore
                )
                end_line = start_line + len(source_lines)
            except OSError:
                start_line, end_line = -1, -1

        completion = self._call_completion(function_header, start_line, end_line)
        generated_tests: Dict[str, str] = rewrite_tests(function_header + completion)
        for test_name in generated_tests:
            if test_name in function_header:
                return generated_tests[test_name]
        return ""


languagemodel = _OpenAILanguageModel()
