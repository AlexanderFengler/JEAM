"""Expose the installed JEAM package version."""

from importlib.metadata import version

__version__ = version("jeam")
