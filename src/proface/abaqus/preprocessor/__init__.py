# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

"""ProFACE Preprocessor package for Abaqus FEA"""

from ._version import __version__
from .translator import AbaqusTranslatorError, main

__all__ = ["AbaqusTranslatorError", "__version__", "main"]
