#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#


import tests.fixtures.cluster.another_dependency as ad  # noqa: F401
import tests.fixtures.cluster.complex_dependency as cd  # noqa: F401
import tests.fixtures.cluster.dependency  # noqa: F401
import tests.fixtures.cluster.typing_parameters_legacy  # noqa: F401
from tests.fixtures.cluster.no_dependencies import Test, test_function  # noqa: F401


def foo(t):
    pass
