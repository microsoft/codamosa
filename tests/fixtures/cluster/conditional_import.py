#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
import typing

if typing.TYPE_CHECKING:
    from tests.fixtures.cluster.complex_dependency import SomeOtherType


class SomeClass:
    def __init__(self, arg0: "SomeOtherType"):
        pass
