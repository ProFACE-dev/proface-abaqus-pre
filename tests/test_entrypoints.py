# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

from importlib.metadata import entry_points

from proface.abaqus.preprocessor import main


def test_entrypoint():
    # check that preprocessor entrypoint is correctly defined
    (ep,) = entry_points(name="abaqus", group="proface.preprocessor")
    main_ep = ep.load()
    assert main_ep is main
