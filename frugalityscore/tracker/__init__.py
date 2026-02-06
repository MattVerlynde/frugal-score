"Documentation du package frugalityscore."

__version__ = "sub-package tracker"
__all__ = ['codecarbon', 'externalplug']

for _mod in __all__:  # imports programmatiques ≈ from __name__ import _mod
    __import__(__name__ + '.' + _mod, fromlist=[None])