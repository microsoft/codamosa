#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#


from tests.fixtures.cluster.no_dependencies import test_function, Test
import tests.fixtures.cluster.another_dependency as ad
import tests.fixtures.cluster.dependency
import tests.fixtures.cluster.typing_parameters_legacy
import tests.fixtures.cluster.complex_dependency as cd

def foo(t):
    pass

