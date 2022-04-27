#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

from tests.fixtures.cluster.complex_dependency import YetAnotherType
import tests.fixtures.cluster.dependency
import tests.fixtures.cluster.another_dependency as ad

def from_import_hint(t: YetAnotherType):
    pass

def import_hint(t: tests.fixtures.cluster.dependency.SomeArgumentType):
    pass

def import_as_hint(t: ad.AnotherArgumentType):
    pass
