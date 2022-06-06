from collections import namedtuple
from typing import Union, NamedTuple, Tuple, List


def to_namedtuple(dd) -> Union[NamedTuple, Tuple, List]:
    the_tuple = namedtuple('NamedTuple', list(dd.keys()))
    ret_tuple = the_tuple(*dd.values())
    return ret_tuple
