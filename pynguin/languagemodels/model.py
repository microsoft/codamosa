#  This file is part of CodaMOSA.
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#
import ast
import inspect
import itertools
import json
import logging
import os
import string
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List

import requests

import pynguin.configuration as config
import pynguin.testcase.testcase as tc
import pynguin.utils.statistics.statistics as stat
from pynguin.generation.export.pytestexporter import PyTestExporter
from pynguin.languagemodels.functionplaceholderadder import add_placeholder
from pynguin.languagemodels.outputfixers import fixup_result, rewrite_tests
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

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


def _openai_api_legacy_request(self, function_header, context):
    # TODO: remove this function as part of Issue #19
    url = f"{self._model_base_url}/v1/engines/{self._complete_model}/completions"
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
    return url, payload, headers


def _openai_api_request(self, function_header, context):
    url = f"{self._model_base_url}{self._model_relative_url}"
    payload = {
        "model": self._complete_model,
        "prompt": context + "\n" + function_header,
        "max_tokens": 200,
        "temperature": self._temperature,
        "stop": ["\n# Unit test for", "\ndef ", "\nclass "],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self._authorization_key}",
    }
    return url, payload, headers


class _OpenAILanguageModel:
    """
    An interface for an OpenAI language model to generate/mutate tests as natural language.
    TODO(ANON): starting by implementing a concrete instance of this.
    """

    def __init__(self):
        self._test_src: str
        self._authorization_key: str
        self._complete_model: str
        self._model_base_url: str
        self._model_relative_url: str
        self._edit_model: str
        self._log_path: str = ""
        # TODO(ANON): make configurable; adding a fudge factor
        self._max_query_len = 4000 - 200
        # TODO(ANON): make configurable
        self._temperature: float
        self._token_len_cache = {}
        self.num_codex_calls: int = 0
        self.time_calling_codex: float = 0

    @property
    def temperature(self) -> float:
        """Provides the temperature being used

        Returns:
            the temperature being used
        """
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float):
        self._temperature = temperature

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

    @property
    def model_base_url(self) -> str:
        """The base url used to interact with the model. Put together, model_base_url and model_relative_url describe
        the url for the model

        Returns:
            The base url used to interact with the model
        """
        return self._model_base_url

    @model_base_url.setter
    def model_base_url(self, model_base_url: str):
        self._model_base_url = model_base_url

    @property
    def model_relative_url(self) -> str:
        """The relative url used to interact with the model. Put together, model_base_url and model_relative_url
        describe the url for the model

        Returns:
            The relative url used to interact with the model
        """
        return self._model_relative_url

    @model_relative_url.setter
    def model_relative_url(self, model_relative_url: str):
        self._model_relative_url = model_relative_url

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
        # context = self._get_maximal_source_context(
        #     used_tokens=approx_number_tokens(function_to_mutate)
        # )
        context = ""
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
        time_start = time.time()
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        self.time_calling_codex += time.time() - time_start
        self.num_codex_calls += 1
        stat.track_output_variable(RuntimeVariable.LLMCalls, self.num_codex_calls)
        stat.track_output_variable(
            RuntimeVariable.LLMQueryTime, self.time_calling_codex
        )
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

        if self.model_base_url == "https://api.openai.com":
            url, payload, headers = _openai_api_legacy_request(
                self, function_header, context
            )
        else:
            url, payload, headers = _openai_api_request(self, function_header, context)

        # We want to stop the generation before it spits out a bunch of other tests,
        # because that slows things down

        time_start = time.time()
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        self.time_calling_codex += time.time() - time_start
        self.num_codex_calls += 1
        stat.track_output_variable(RuntimeVariable.LLMCalls, self.num_codex_calls)
        stat.track_output_variable(
            RuntimeVariable.LLMQueryTime, self.time_calling_codex
        )
        if res.status_code != 200:
            logger.error("Failed to call for completion:\n%s", res.json())
            logger.error(self.complete_model)
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
        function_with_placeholder = add_placeholder(ast_test_case_module)
        # print("Here's the function with placeholder:\n ",function_with_placeholder)
        mutated = self._call_mutate(function_with_placeholder)

        test_start_idxs = [
            i
            for i, line in enumerate(mutated.split("\n"))
            if line.startswith("def test_")
        ]
        if len(test_start_idxs) == 0:
            return str_test_case
        mutated_test_as_str = "\n".join(mutated.split("\n")[test_start_idxs[0] :])
        # print("Here's what codex outputted:\n", mutated_test_as_str)
        mutated_tests_fixed: Dict[str, str] = rewrite_tests(mutated_test_as_str)
        return "\n\n".join(mutated_tests_fixed.values())

    def target_test_case(self, gao: GenericCallableAccessibleObject, context="") -> str:
        """Provides a test case targeted to the function/method/constructor
        specified in `gao`

        Args:
            gao: a GenericCallableAccessibleObject to target the test to
            context: extra context to pass before the function header

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
            except (TypeError, OSError):
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
            except (TypeError, OSError):
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
            except (TypeError, OSError):
                start_line, end_line = -1, -1

        completion = self._call_completion(
            context + function_header, start_line, end_line
        )
        # Remove any trailing statements that don't parse
        generated_test = fixup_result(function_header + completion)
        report_dir = config.configuration.statistics_output.report_dir
        if report_dir != "pynguin-report":
            with open(
                os.path.join(report_dir, "codex_generations.py"),
                "a+",
                encoding="UTF-8",
            ) as log_file:
                log_file.write(f"\n\n# Generated at {datetime.now()}\n")
                log_file.write(generated_test)
        generated_tests: Dict[str, str] = rewrite_tests(generated_test)
        for test_name in generated_tests:
            if test_name in function_header:
                return generated_tests[test_name]
        return ""


class FileMockedModel(_OpenAILanguageModel):
    def __init__(self, filename: str):
        assert os.path.isfile(filename)
        self._generation_bank: Dict[str, Iterable[str]] = {}
        self._initialize_contents(filename)
        super().__init__()

    def _initialize_contents(self, filename):
        contents_bank: Dict[str, List[str]] = defaultdict(list)
        with open(filename, encoding="UTF-8") as generations_file:
            all_lines = generations_file.readlines()
            i = 0
            while i < len(all_lines):
                cur_line = all_lines[i]
                if cur_line.startswith("# Generated at "):
                    if i + 2 > len(all_lines):
                        break
                    header = all_lines[i + 1] + all_lines[i + 2].rstrip()
                    i = i + 3
                    contents = []
                    while i < len(all_lines) and not all_lines[i].startswith(
                        "# Generated at "
                    ):
                        contents.append(all_lines[i])
                        i = i + 1
                    contents_bank[header].append("".join(contents))
                else:
                    i = i + 1
        for header, contents_lst in contents_bank.items():
            if len(contents_lst) > 0:
                self._generation_bank[header] = itertools.cycle(contents_lst)

    def _call_completion(
        self, function_header: str, context_start: int, context_end: int
    ) -> str:
        if function_header in self._generation_bank:
            ret_value = "\n" + next(self._generation_bank[function_header])  # type: ignore
            return ret_value
        else:
            return "\npass\n"


languagemodel = _OpenAILanguageModel()
