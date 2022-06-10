from collections import namedtuple
from typing import List, NamedTuple, Tuple, Union


def to_namedtuple(dd) -> Union[NamedTuple, Tuple, List]:
    the_tuple = namedtuple("NamedTuple", list(dd.keys()))
    ret_tuple = the_tuple(*dd.values())
    return ret_tuple
