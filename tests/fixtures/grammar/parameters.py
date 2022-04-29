#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2020 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#


def positional_only(param1, param2=5, /):
    pass


def all_params(param1, /, param2, *param3, param4=0, **param5):
    pass


class A:
    def __init__(self, lvl):
        self.x: int = 4
        self.y = 5
        self.a = A(lvl - 1) if lvl > 0 else None
