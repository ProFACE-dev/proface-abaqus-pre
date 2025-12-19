# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import h5py
import pytest

from proface.abaqus.preprocessor import translator


@pytest.fixture
def h5_empty(tmp_path):
    return h5py.File(name=tmp_path / "empty", mode="w")


@pytest.fixture
def h5_pth():
    return Path("foobar.toml")


def test_main_args_types(h5_empty, h5_pth):
    with pytest.raises(TypeError) as excinfo:
        translator.main(job="{}", job_path=h5_pth, h5=h5_empty)
    assert excinfo.type is TypeError

    with pytest.raises(TypeError) as excinfo:
        translator.main(job={}, job_path="h5_pth", h5=h5_empty)
    assert excinfo.type is TypeError

    with pytest.raises(TypeError) as excinfo:
        translator.main(job={}, job_path=h5_pth, h5="h5_empty")
    assert excinfo.type is TypeError


def test_file_not_found(h5_empty, h5_pth):
    with pytest.raises(FileNotFoundError) as excinfo:
        translator.main(job={}, job_path=h5_pth, h5=h5_empty)
    assert str(excinfo.value).endswith("'foobar.fil'")

    with pytest.raises(FileNotFoundError) as excinfo:
        translator.main(
            job={"input": {"fil": "user.fil"}}, job_path=h5_pth, h5=h5_empty
        )
    assert str(excinfo.value).endswith("'user.fil'")
