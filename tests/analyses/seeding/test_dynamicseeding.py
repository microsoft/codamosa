#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2021 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
import importlib
import os

import pytest

from pynguin.analyses.seeding.dynamicseeding import DynamicSeeding
from pynguin.analyses.seeding.dynamicseedinginstrumentation import (
    DynamicSeedingInstrumentation,
)


@pytest.fixture()
def instr():
    instr = DynamicSeedingInstrumentation()
    return instr


@pytest.fixture()
def dynamic_seeding():
    DynamicSeeding()._dynamic_pool = {int: set(), float: set(), str: set()}


@pytest.fixture()
def dummy_module():
    dummy_module = importlib.import_module(
        "tests.fixtures.seeding.dynamicseeding.dynamicseedingdummies"
    )
    dummy_module = importlib.reload(dummy_module)
    return dummy_module


def test_singleton():
    dynamic_seeding_1 = DynamicSeeding()
    dynamic_seeding_2 = DynamicSeeding()
    assert dynamic_seeding_1 is dynamic_seeding_2


def test_random_int(dynamic_seeding):
    DynamicSeeding()._dynamic_pool[int].add(5)
    value = DynamicSeeding().random_int

    assert value == 5


def test_random_float(dynamic_seeding):
    DynamicSeeding()._dynamic_pool[float].add(5.0)
    value = DynamicSeeding().random_float

    assert value == 5.0


def test_random_string(dynamic_seeding):
    DynamicSeeding()._dynamic_pool[str].add("5")
    value = DynamicSeeding().random_string

    assert value == "5"


def test_compare_op_int(instr, dummy_module, dynamic_seeding):
    dummy_module.compare_op_dummy.__code__ = instr.instrument_module(
        dummy_module.compare_op_dummy.__code__
    )
    res = dummy_module.compare_op_dummy(10, 11)

    assert res == 1
    assert 10 in DynamicSeeding()._dynamic_pool[int]
    assert 11 in DynamicSeeding()._dynamic_pool[int]


def test_compare_op_float(instr, dummy_module, dynamic_seeding):
    dummy_module.compare_op_dummy.__code__ = instr.instrument_module(
        dummy_module.compare_op_dummy.__code__
    )
    res = dummy_module.compare_op_dummy(1.0, 2.5)

    assert res == 1
    assert 1.0 in DynamicSeeding()._dynamic_pool[float]
    assert 2.5 in DynamicSeeding()._dynamic_pool[float]


def test_compare_op_string(instr, dummy_module, dynamic_seeding):
    dummy_module.compare_op_dummy.__code__ = instr.instrument_module(
        dummy_module.compare_op_dummy.__code__
    )
    res = dummy_module.compare_op_dummy("abc", "def")

    assert res == 1
    assert "abc" in DynamicSeeding()._dynamic_pool[str]
    assert "def" in DynamicSeeding()._dynamic_pool[str]


def test_compare_op_other_type(instr, dummy_module, dynamic_seeding):
    dummy_module.compare_op_dummy.__code__ = instr.instrument_module(
        dummy_module.compare_op_dummy.__code__
    )
    res = dummy_module.compare_op_dummy(True, "def")

    assert res == 1
    assert not DynamicSeeding().has_ints
    assert not DynamicSeeding().has_floats
    assert DynamicSeeding().has_strings
    assert "def" in DynamicSeeding()._dynamic_pool[str]


def test_startswith_function(instr, dummy_module, dynamic_seeding):
    dummy_module.startswith_dummy.__code__ = instr.instrument_module(
        dummy_module.startswith_dummy.__code__
    )
    res = dummy_module.startswith_dummy("abc", "ab")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "ababc" in DynamicSeeding()._dynamic_pool[str]


def test_endswith_function(instr, dummy_module, dynamic_seeding):
    dummy_module.endswith_dummy.__code__ = instr.instrument_module(
        dummy_module.endswith_dummy.__code__
    )
    res = dummy_module.endswith_dummy("abc", "bc")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "abcbc" in DynamicSeeding()._dynamic_pool[str]


def test_isalnum_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isalnum_dummy.__code__ = instr.instrument_module(
        dummy_module.isalnum_dummy.__code__
    )
    res = dummy_module.isalnum_dummy("alnumtest")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "alnumtest" in DynamicSeeding()._dynamic_pool[str]
    assert "alnumtest!" in DynamicSeeding()._dynamic_pool[str]


def test_isalnum_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isalnum_dummy.__code__ = instr.instrument_module(
        dummy_module.isalnum_dummy.__code__
    )
    res = dummy_module.isalnum_dummy("alnum_test")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "alnum_test" in DynamicSeeding()._dynamic_pool[str]
    assert "isalnum" in DynamicSeeding()._dynamic_pool[str]


def test_islower_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.islower_dummy.__code__ = instr.instrument_module(
        dummy_module.islower_dummy.__code__
    )
    res = dummy_module.islower_dummy("lower")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "lower" in DynamicSeeding()._dynamic_pool[str]
    assert "LOWER" in DynamicSeeding()._dynamic_pool[str]


def test_islower_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.islower_dummy.__code__ = instr.instrument_module(
        dummy_module.islower_dummy.__code__
    )
    res = dummy_module.islower_dummy("NotLower")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "NotLower" in DynamicSeeding()._dynamic_pool[str]
    assert "notlower" in DynamicSeeding()._dynamic_pool[str]


def test_isupper_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isupper_dummy.__code__ = instr.instrument_module(
        dummy_module.isupper_dummy.__code__
    )
    res = dummy_module.isupper_dummy("UPPER")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "UPPER" in DynamicSeeding()._dynamic_pool[str]
    assert "upper" in DynamicSeeding()._dynamic_pool[str]


def test_isupper_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isupper_dummy.__code__ = instr.instrument_module(
        dummy_module.isupper_dummy.__code__
    )
    res = dummy_module.isupper_dummy("NotUpper")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "NotUpper" in DynamicSeeding()._dynamic_pool[str]
    assert "NOTUPPER" in DynamicSeeding()._dynamic_pool[str]


def test_isdecimal_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isdecimal_dummy.__code__ = instr.instrument_module(
        dummy_module.isdecimal_dummy.__code__
    )
    res = dummy_module.isdecimal_dummy("012345")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "012345" in DynamicSeeding()._dynamic_pool[str]
    assert "non_decimal" in DynamicSeeding()._dynamic_pool[str]


def test_isdecimal_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isdecimal_dummy.__code__ = instr.instrument_module(
        dummy_module.isdecimal_dummy.__code__
    )
    res = dummy_module.isdecimal_dummy("not_decimal")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "not_decimal" in DynamicSeeding()._dynamic_pool[str]
    assert "0123456789" in DynamicSeeding()._dynamic_pool[str]


def test_isalpha_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isalpha_dummy.__code__ = instr.instrument_module(
        dummy_module.isalpha_dummy.__code__
    )
    res = dummy_module.isalpha_dummy("alpha")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "alpha" in DynamicSeeding()._dynamic_pool[str]
    assert "alpha1" in DynamicSeeding()._dynamic_pool[str]


def test_isalpha_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isalpha_dummy.__code__ = instr.instrument_module(
        dummy_module.isalpha_dummy.__code__
    )
    res = dummy_module.isalpha_dummy("not_alpha")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "not_alpha" in DynamicSeeding()._dynamic_pool[str]
    assert "isalpha" in DynamicSeeding()._dynamic_pool[str]


def test_isdigit_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isdigit_dummy.__code__ = instr.instrument_module(
        dummy_module.isdigit_dummy.__code__
    )
    res = dummy_module.isdigit_dummy("012345")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "012345" in DynamicSeeding()._dynamic_pool[str]
    assert "012345_" in DynamicSeeding()._dynamic_pool[str]


def test_isdigit_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isdigit_dummy.__code__ = instr.instrument_module(
        dummy_module.isdigit_dummy.__code__
    )
    res = dummy_module.isdigit_dummy("not_digit")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "not_digit" in DynamicSeeding()._dynamic_pool[str]
    assert "0" in DynamicSeeding()._dynamic_pool[str]


def test_isidentifier_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isidentifier_dummy.__code__ = instr.instrument_module(
        dummy_module.isidentifier_dummy.__code__
    )
    res = dummy_module.isidentifier_dummy("is_identifier")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "is_identifier" in DynamicSeeding()._dynamic_pool[str]
    assert "is_identifier!" in DynamicSeeding()._dynamic_pool[str]


def test_isidentifier_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isidentifier_dummy.__code__ = instr.instrument_module(
        dummy_module.isidentifier_dummy.__code__
    )
    res = dummy_module.isidentifier_dummy("not_identifier!")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "not_identifier!" in DynamicSeeding()._dynamic_pool[str]
    assert "is_Identifier" in DynamicSeeding()._dynamic_pool[str]


def test_isnumeric_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isnumeric_dummy.__code__ = instr.instrument_module(
        dummy_module.isnumeric_dummy.__code__
    )
    res = dummy_module.isnumeric_dummy("44444")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "44444" in DynamicSeeding()._dynamic_pool[str]
    assert "44444A" in DynamicSeeding()._dynamic_pool[str]


def test_isnumeric_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isnumeric_dummy.__code__ = instr.instrument_module(
        dummy_module.isnumeric_dummy.__code__
    )
    res = dummy_module.isnumeric_dummy("not_numeric")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "not_numeric" in DynamicSeeding()._dynamic_pool[str]
    assert "012345" in DynamicSeeding()._dynamic_pool[str]


def test_isprintable_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isprintable_dummy.__code__ = instr.instrument_module(
        dummy_module.isprintable_dummy.__code__
    )
    res = dummy_module.isprintable_dummy("printable")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "printable" in DynamicSeeding()._dynamic_pool[str]
    assert f"printable{os.linesep}" in DynamicSeeding()._dynamic_pool[str]


def test_isprintable_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isprintable_dummy.__code__ = instr.instrument_module(
        dummy_module.isprintable_dummy.__code__
    )
    res = dummy_module.isprintable_dummy("not_printable\n")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert f"not_printable{os.linesep}" in DynamicSeeding()._dynamic_pool[str]
    assert "is_printable" in DynamicSeeding()._dynamic_pool[str]


def test_isspace_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.isspace_dummy.__code__ = instr.instrument_module(
        dummy_module.isspace_dummy.__code__
    )
    res = dummy_module.isspace_dummy(" ")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert " " in DynamicSeeding()._dynamic_pool[str]
    assert " a" in DynamicSeeding()._dynamic_pool[str]


def test_isspace_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.isspace_dummy.__code__ = instr.instrument_module(
        dummy_module.isspace_dummy.__code__
    )
    res = dummy_module.isspace_dummy("no_space")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "no_space" in DynamicSeeding()._dynamic_pool[str]
    assert "   " in DynamicSeeding()._dynamic_pool[str]


def test_istitle_function_true(instr, dummy_module, dynamic_seeding):
    dummy_module.istitle_dummy.__code__ = instr.instrument_module(
        dummy_module.istitle_dummy.__code__
    )
    res = dummy_module.istitle_dummy("Title")

    assert res == 0
    assert DynamicSeeding().has_strings
    assert "Title" in DynamicSeeding()._dynamic_pool[str]
    assert "Title AAA" in DynamicSeeding()._dynamic_pool[str]


def test_istitle_function_false(instr, dummy_module, dynamic_seeding):
    dummy_module.istitle_dummy.__code__ = instr.instrument_module(
        dummy_module.istitle_dummy.__code__
    )
    res = dummy_module.istitle_dummy("no Title")

    assert res == 1
    assert DynamicSeeding().has_strings
    assert "no Title" in DynamicSeeding()._dynamic_pool[str]
    assert "Is Title" in DynamicSeeding()._dynamic_pool[str]
