# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

from proface.abaqus.preprocessor import translator

ELBLOC_DTYPE = [
    ("eltyp", "S8"),
    ("elnum", np.int64),
    ("ninc", np.int64, (8,)),
]


def _elbloc(eltype, elnum, nodes):
    return np.array([(eltype, elnum, nodes)], dtype=ELBLOC_DTYPE)


def _meshblock(
    *,
    topology="C3D8",
    eltype="C3D8",
    incidences=None,
    numbers=None,
):
    if incidences is None:
        incidences = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    if numbers is None:
        numbers = np.array([1])
    return translator.MeshBlock(
        topology=topology,
        eltype=eltype,
        incidences=incidences,
        numbers=numbers,
    )


@pytest.fixture
def h5_empty(tmp_path):
    return h5py.File(name=tmp_path / "empty", mode="w")


@pytest.fixture
def h5_pth():
    return Path("foobar.toml")


def test_main_args_types(h5_empty, h5_pth):
    with pytest.raises(TypeError) as excinfo:
        translator.main(job=cast("Any", "{}"), job_path=h5_pth, h5=h5_empty)
    assert excinfo.type is TypeError

    with pytest.raises(TypeError) as excinfo:
        translator.main(job={}, job_path=cast("Any", "h5_pth"), h5=h5_empty)
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


def test_mixed_formulations_same_topology_error(h5_empty):
    fil = SimpleNamespace(
        info={"nelm": 2, "elsiz": 8},
        elm=[
            _elbloc(b"C3D8", 1, np.arange(1, 9)),
            _elbloc(b"C3D8R", 2, np.arange(9, 17)),
        ],
    )

    with pytest.raises(
        translator.AbaqusTranslatorError,
        match=r"^Mixed element formulations",
    ):
        translator._write_element(fil=fil, h5=h5_empty)  # noqa: SLF001


def test_meshblock_merge_different_topology_error():
    c3d8 = _meshblock()
    c3d20 = _meshblock(topology="C3D20", eltype="C3D20")

    with pytest.raises(ValueError, match="Can merge only same topology"):
        c3d8.merge(c3d20)


def test_meshblock_inconsistent_length_error():
    with pytest.raises(ValueError, match="must have the same length"):
        _meshblock(
            incidences=np.array(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [9, 10, 11, 12, 13, 14, 15, 16],
                ]
            ),
            numbers=np.array([1]),
        )


def test_meshblock_merge_incompatible_incidence_shape_error():
    c3d8 = _meshblock()
    c3d8_numbers = c3d8.numbers.copy()
    c3d8_incidences = c3d8.incidences.copy()
    c3d8_wide = _meshblock(
        incidences=np.array([np.arange(1, 21)]),
        numbers=np.array([2]),
    )

    with pytest.raises(ValueError, match=r"^Topology error merging C3D8"):
        c3d8.merge(c3d8_wide)

    np.testing.assert_array_equal(c3d8.numbers, c3d8_numbers)
    np.testing.assert_array_equal(c3d8.incidences, c3d8_incidences)


def test_meshblock_merge_incompatible_dtype_error():
    c3d8 = _meshblock()
    c3d8_numbers = c3d8.numbers.copy()
    c3d8_incidences = c3d8.incidences.copy()
    c3d8_int32 = _meshblock(numbers=np.array([2], dtype=np.int32))

    with pytest.raises(ValueError, match=r"^MeshBlock of incompatible dtype"):
        c3d8.merge(c3d8_int32)

    np.testing.assert_array_equal(c3d8.numbers, c3d8_numbers)
    np.testing.assert_array_equal(c3d8.incidences, c3d8_incidences)


def test_meshblock_unsupported_abaqus_topology_is_none():
    mb = translator.MeshBlock.from_abaqus(_elbloc(b"S4", 1, np.arange(1, 9)))

    assert mb.topology is None
    assert mb.eltype == "S4"


def test_same_formulation_same_topology_merge(h5_empty):
    fil = SimpleNamespace(
        info={"nelm": 2, "elsiz": 8},
        elm=[
            _elbloc(b"C3D8", 2, np.arange(9, 17)),
            _elbloc(b"C3D8", 1, np.arange(1, 9)),
        ],
    )

    translator._write_element(fil=fil, h5=h5_empty)  # noqa: SLF001

    np.testing.assert_array_equal(
        h5_empty["elements"]["C3D8"]["numbers"][:],
        [1, 2],
    )
    np.testing.assert_array_equal(
        h5_empty["elements"]["C3D8"]["incidences"][:],
        [np.arange(1, 9), np.arange(9, 17)],
    )
