#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
from tests.fixtures.cluster.complex_dependency import SomeOtherType
import tests.fixtures.cluster.typeless_dependency

class SomeClass:
    def __init__(self, arg0):
        pass

    def other_modifier(self, arg1):
        pass
