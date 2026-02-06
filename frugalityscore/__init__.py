"""
frugality-score (a.k.a. `frugalscore`): Fuzzy-Logic-Based Frugality Score toolbox for Python.

Recommended Use
---------------
>>> import skfuzzy as fuzz

"""
__all__ = []

__version__ = "0.1"

from . import mod
from . import tracker
from . import models


import skfuzzy.membership as _membership
from skfuzzy.membership import *
__all__.extend(_membership.__all__)


import skfuzzy.defuzzify as _defuzz
from skfuzzy.defuzzify import *
__all__.extend(_defuzz.__all__)