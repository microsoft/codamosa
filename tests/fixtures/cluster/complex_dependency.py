#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
from __future__ import annotations


class YetAnotherType:
    def __init__(self, arg0: int) -> None:
        pass

    def some_modifier(self, arg0: SomeOtherType) -> None:
        pass


class SomeOtherType:
    def __init__(self, arg0: YetAnotherType):
        pass

    def some_modifier(self, arg0: YetAnotherType) -> None:
        pass
