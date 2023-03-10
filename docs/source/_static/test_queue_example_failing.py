#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#

# Automatically generated by Pynguin.
import queue_example as module_0


def test_case_0():
    try:
        int_0 = -2944
        queue_0 = module_0.Queue(int_0)
    except BaseException:
        pass


def test_case_1():
    try:
        int_0 = 3390
        queue_0 = module_0.Queue(int_0)
        assert queue_0.max == 3390
        assert queue_0.head == 0
        assert queue_0.tail == 0
        assert queue_0.size == 0
        assert len(queue_0.data) == 3390
        bool_0 = queue_0.full()
        assert bool_0 is False
        int_1 = -475
        queue_1 = module_0.Queue(int_1)
    except BaseException:
        pass


def test_case_2():
    try:
        int_0 = 1769
        int_1 = 1080
        int_2 = 3
        queue_0 = module_0.Queue(int_2)
        assert queue_0.max == 3
        assert queue_0.head == 0
        assert queue_0.tail == 0
        assert queue_0.size == 0
        assert len(queue_0.data) == 3
        bool_0 = queue_0.full()
        assert bool_0 is False
        bool_1 = queue_0.empty()
        assert bool_1 is False
        int_3 = 1235
        bool_2 = queue_0.enqueue(int_3)
        assert bool_2 is True
        assert queue_0.tail == 1
        assert queue_0.size == 1
        bool_3 = queue_0.empty()
        assert bool_3 is True
        bool_4 = queue_0.enqueue(int_0)
        assert bool_4 is True
        assert queue_0.tail == 2
        assert queue_0.size == 2
        optional_0 = queue_0.dequeue()
        assert optional_0 == 1235
        assert queue_0.head == 1
        assert queue_0.size == 1
        optional_1 = queue_0.dequeue()
        assert optional_1 == 1769
        assert queue_0.head == 2
        assert queue_0.size == 0
        bool_5 = queue_0.enqueue(int_2)
        assert bool_5 is True
        assert queue_0.tail == 0
        assert queue_0.size == 1
        optional_2 = queue_0.dequeue()
        assert optional_2 == 3
        assert queue_0.head == 0
        assert queue_0.size == 0
        queue_1 = module_0.Queue(int_1)
        assert queue_1.max == 1080
        assert queue_1.head == 0
        assert queue_1.tail == 0
        assert queue_1.size == 0
        assert len(queue_1.data) == 1080
        optional_3 = queue_1.dequeue()
        assert optional_3 is None
        int_4 = -820
        queue_2 = module_0.Queue(int_4)
    except BaseException:
        pass


def test_case_3():
    try:
        int_0 = 1769
        int_1 = 3
        queue_0 = module_0.Queue(int_1)
        assert queue_0.max == 3
        assert queue_0.head == 0
        assert queue_0.tail == 0
        assert queue_0.size == 0
        assert len(queue_0.data) == 3
        bool_0 = queue_0.full()
        assert bool_0 is False
        bool_1 = queue_0.empty()
        assert bool_1 is False
        int_2 = 1272
        bool_2 = queue_0.enqueue(int_2)
        assert bool_2 is True
        assert queue_0.tail == 1
        assert queue_0.size == 1
        int_3 = 435
        bool_3 = queue_0.enqueue(int_3)
        assert bool_3 is True
        assert queue_0.tail == 2
        assert queue_0.size == 2
        bool_4 = queue_0.empty()
        assert bool_4 is True
        bool_5 = queue_0.enqueue(int_0)
        assert bool_5 is True
        assert queue_0.tail == 0
        assert queue_0.size == 3
        optional_0 = queue_0.dequeue()
        assert optional_0 == 1272
        assert queue_0.head == 1
        assert queue_0.size == 2
        int_4 = 1021
        bool_6 = queue_0.enqueue(int_1)
        assert bool_6 is True
        assert queue_0.tail == 1
        assert queue_0.size == 3
        bool_7 = queue_0.enqueue(int_4)
        assert bool_7 is False
        optional_1 = queue_0.dequeue()
        assert optional_1 == 435
        assert queue_0.head == 2
        assert queue_0.size == 2
        bool_8 = queue_0.empty()
        assert bool_8 is True
        bool_9 = queue_0.empty()
        assert bool_9 is True
        int_5 = 688
        queue_1 = module_0.Queue(int_5)
        assert queue_1.max == 688
        assert queue_1.head == 0
        assert queue_1.tail == 0
        assert queue_1.size == 0
        assert len(queue_1.data) == 688
        bool_10 = queue_1.full()
        assert bool_10 is False
        int_6 = 203
        queue_2 = module_0.Queue(int_6)
        assert queue_2.max == 203
        assert queue_2.head == 0
        assert queue_2.tail == 0
        assert queue_2.size == 0
        assert len(queue_2.data) == 203
        optional_2 = queue_1.dequeue()
        assert optional_2 is None
        optional_3 = queue_2.dequeue()
        assert optional_3 is None
        bool_11 = queue_1.empty()
        assert bool_11 is False
        bool_12 = queue_2.empty()
        assert bool_12 is False
        optional_4 = queue_2.dequeue()
        assert optional_4 is None
        int_7 = -256
        queue_3 = module_0.Queue(int_7)
    except BaseException:
        pass
