# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

"""Translator implementation"""

import collections.abc
import dataclasses
import logging
import re
from pathlib import Path
from typing import Self

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
from suanpan.abqfil import AbqFil  # type: ignore[import-untyped]

from proface.preprocessor import PreprocessorError

from . import __version__

logger = logging.getLogger(__name__)


# translation from Abaqus result codes to ProFACE ids
ABQ_VAR = {"R11": "S", "R401": "SP", "R76": "IVOL", "R8": "COORD"}
# translation from Abaqus location codes to ProFACE ids
ABQ_LOC = {0: "integration_point", 4: "nodal_averaged"}

# function for computing results path in h5
_h5_path = "{var:s}/{loc:s}/{eltype:s}".format

# save results in single precision
RES_CLASS = np.float32
RES_DTYPE = np.dtype(RES_CLASS)
# FIXME: be more specific when suanpan has better typing support
NUM_CLASS = np.integer


class AbaqusTranslatorError(PreprocessorError):
    pass


@dataclasses.dataclass
class MeshBlock:
    """internal structure for caching mesh info"""

    incidences: npt.NDArray[NUM_CLASS]
    numbers: npt.NDArray[NUM_CLASS]

    def merge(self, other: Self) -> None:
        """merge other MeshBlock keeping element number ordering"""

        if np.any(np.isin(other.numbers, self.numbers, assume_unique=True)):
            msg = "'numbers' are not unique after merge"
            raise ValueError(msg)

        for f in (i.name for i in dataclasses.fields(self)):
            cur = getattr(self, f)
            oth = getattr(other, f)
            try:
                new = np.concatenate((cur, oth), axis=0, casting="no")
            except ValueError as exc:
                msg = (
                    f"MeshBlock of incompatible shape at '{f}': "
                    f"{cur.shape}, {oth.shape}"
                )
                raise ValueError(msg) from exc
            setattr(self, f, new)

        # enforce element numbers still strictly sorted after concatenate
        if not np.all(self.numbers[:-1] < self.numbers[1:]):
            arg = np.argsort(self.numbers)
            self.incidences = self.incidences[arg]
            self.numbers = self.numbers[arg]

    @property
    def nodes(self) -> npt.NDArray[NUM_CLASS]:
        return np.unique(self.incidences, sorted=True)


def main(
    *, job: collections.abc.Mapping, job_path: Path, h5: h5py.Group
) -> None:
    """main entrypoint for abaqus preprocessor"""

    # runtime type checking
    if not isinstance(job, collections.abc.Mapping):
        msg = "'job' must be a mapping"
        raise TypeError(msg)
    if not isinstance(job_path, Path):
        msg = "'job_path' must be a pathlib.Path"
        raise TypeError(msg)
    if not isinstance(h5, h5py.Group):
        msg = "'h5' must be a h5py.Group or h5py.File"
        raise TypeError(msg)

    logger.info(
        "\U0001f680 START Abaqus to ProFACE translator, ver. %s",
        __version__,
    )  # 🚀

    # compute .fil path
    userpth = job.get("input", {}).get("fil")
    filpth = (
        job_path.parent / userpth if userpth else job_path.with_suffix(".fil")
    )

    # run translator
    try:
        _pre(filpth=filpth, h5=h5, results=job.get("results", {}))
    except OSError:
        # caller should treat OSError
        raise
    except Exception:
        msg = "Internal Error"
        logger.exception(msg)
        raise AbaqusTranslatorError(msg) from None
    logger.info("\U0001f3c1 END Abaqus to ProFACE translator")  # 🏁


def _pre(filpth, h5: h5py.File, results):
    """Abaqus translator"""

    logger.info("reading %s", filpth.resolve().as_uri())
    logger.info("writing %s", Path(h5.filename).resolve().as_uri())

    fil = AbqFil(filpth)

    _write_meta(fil=fil, h5=h5)
    _write_nodal(fil=fil, h5=h5)
    _write_element(fil=fil, h5=h5)
    _write_sets(fil=fil, h5=h5)

    if not results:
        logger.warning("no results request")
        return
    h5_res = h5.create_group("results")
    for k, v in results.items():
        if "step" not in v or "increment" not in v:
            logger.error(
                "Results request '%s': "
                "both 'step' and 'increment' must be specified.",
                k,
            )
            continue
        _write_results(
            fil=fil,
            h5=h5,
            h5_res=h5_res,
            name=k,
            step=v["step"],
            inc=v["increment"],
        )


def _write_meta(fil, h5):
    #
    # metadata
    #
    logger.info("Abaqus ver. %s", _label(fil.info["ver"]))
    logger.info(
        "Analysis run on %s",
        _label(fil.info["date"]) + " " + _label(fil.info["time"]),
    )
    if fil.heading.strip():
        logger.info("Heading '%s'", _label(fil.heading))
    logger.info("Number of elements: %9d", fil.info["nelm"])
    logger.info("Number of nodes   : %9d", fil.info["nnod"])

    h5.attrs["program"] = "Abaqus"
    h5.attrs["version"] = _label(fil.info["ver"])
    h5.attrs["run_datetime"] = (
        _label(fil.info["date"]) + " " + _label(fil.info["time"])
    )
    h5.attrs["title"] = _label(fil.heading)


def _write_nodal(fil, h5):
    #
    # nodal data
    #
    h5_nodes = h5.create_group("nodes")
    h5_nodes.attrs["number"] = fil.info["nnod"]
    h5_nodes.create_dataset("coordinates", data=fil.coord["coord"])
    h5_nodes.create_dataset("numbers", data=fil.coord["nnum"])


def _write_element(fil, h5):
    #
    # element data
    #
    h5_elements = h5.create_group("elements")
    h5_elements.attrs["number"] = fil.info["nelm"]
    h5_elements.attrs["size"] = fil.info["elsiz"]

    # scan fil for element blocs
    blocs: dict[str, MeshBlock] = {}
    for elbloc in fil.elm:
        # check elbloc is homogeneous
        assert (elbloc["eltyp"] == elbloc["eltyp"][0]).all()
        abqlabel = _label(elbloc["eltyp"][0])
        eltype = _proface_eltype(abqlabel)
        if eltype is None:
            logger.info(
                "Skipping %d elements of type %s", len(elbloc), abqlabel
            )
            continue
        mb = MeshBlock(incidences=elbloc["ninc"], numbers=elbloc["elnum"])
        if eltype not in blocs:
            logger.debug(
                "Storing %d elements of type %s [topology %s]",
                len(elbloc),
                abqlabel,
                eltype,
            )
            blocs[eltype] = mb
        else:
            blocs[eltype].merge(mb)

    # save in h5
    for eltype, mb in blocs.items():
        h5_elgroup = h5_elements.create_group(eltype)
        h5_elgroup.create_dataset("incidences", data=mb.incidences)
        h5_elgroup.create_dataset("numbers", data=mb.numbers)
        h5_elgroup.create_dataset("nodes", data=mb.nodes)


def _write_sets(fil, h5):
    #
    # sets
    #
    h5_sets = h5.create_group("sets")
    h5_sets_node = h5_sets.create_group("node")
    for k, nset in fil.nset.items():
        kh = _safe_label(fil.label.get(k, k))
        h5_sets_node.create_dataset(kh, data=nset)
    h5_sets_element = h5_sets.create_group("element")
    for k, elset in fil.elset.items():
        kh = _safe_label(fil.label.get(k, k))
        h5_sets_element.create_dataset(kh, data=elset)


def _write_results(fil, h5, h5_res, name: str, step: int, inc: int) -> None:  # noqa: PLR0913
    #
    # results
    #
    try:
        i = _find_step_inc(fil.step, step, inc)
    except ValueError as exc:
        logger.error("Results '%s': %s", name, exc)
        ## fixme: raise error ?
        return

    logger.info(
        "Results '%s': step %d, increment %d, time %#.3g, '%s'",
        name,
        *fil.step[i][["step", "incr", "ttime"]],
        _label(fil.step[i]["subheading"]),
    )
    h5_k = h5_res.create_group(name)
    h5_k.attrs["step"] = fil.step[i]["step"]
    h5_k.attrs["increment"] = fil.step[i]["incr"]
    h5_k.attrs["time"] = fil.step[i]["ttime"]

    _write_step_output_blocks(fil, h5, h5_k, i)


def _write_step_output_blocks(fil, h5, h5_k, i):  # noqa: C901, PLR0912
    for data_block in fil.get_step(i):
        flag, elset, abqeltype, data = (
            data_block.flag,
            data_block.set,
            data_block.eltype,
            data_block.data,
        )
        # check block flag: 0 is element output
        if flag != 0:
            logger.warning("Skipping non element output: flag = %d", flag)
            continue

        # check block elset
        if _label(elset) != "":
            logger.error("Results file with element sets not supported")
            continue

        # check block element type
        abqeltype = _label(abqeltype)
        eltype = _proface_eltype(abqeltype)
        if eltype is None:
            logger.debug("Skipping %s elbloc", eltype)

        # check block location
        loc = data["loc"][0]
        assert (data["loc"] == loc).all()
        match ABQ_LOC.get(loc):
            case "integration_point":
                # reshape data to index as [el_num, ip_num]
                nr_ip = _guess_nr_ip(data)
                data = data.reshape(-1, nr_ip)
                # data["num"] is elnum across columns
                if len(data["num"]) != len(
                    h5["elements"][eltype]["numbers"]
                ) or np.any(
                    data["num"]
                    != np.expand_dims(h5["elements"][eltype]["numbers"], -1)
                ):
                    msg = (
                        f"Inconsistent records for {eltype}: "
                        "results cardinality/numbering different "
                        "with respect to mesh definition"
                    )
                    raise ValueError(msg)
                # data["ipnum"] is 1..nr_ip across rows
                assert np.all(data["ipnum"] == 1 + np.arange(nr_ip))
            case "nodal_averaged":
                if not np.all(data["num"] == h5["elements"][eltype]["nodes"]):
                    msg = f"Inconsistent records for {eltype}: node numbers"
                    raise ValueError(msg)
                assert (data["ipnum"] == 0).all()
            case None:
                logger.warning("Unknown location code %d", loc)
                continue

        # save block data
        for name in data.dtype.names:
            if not name.startswith("R"):
                continue
            if name not in ABQ_VAR:
                logger.warning("Unexpected results code %s", name)
                continue
            hdata = data[name].astype(RES_DTYPE)
            *_, n_dim = hdata.shape
            assert tuple(_) == data.shape
            if n_dim == 1:
                # squeeze sigleton last dimension
                hdata = hdata.reshape(hdata.shape[:-1])
            dset = h5_k.create_dataset(
                _h5_path(var=ABQ_VAR[name], loc=ABQ_LOC[loc], eltype=eltype),
                data=hdata,
            )
            logger.debug("Wrote %s: %s", dset.name, dset.shape)


def _label(lab: bytes) -> str:
    return lab.decode("ASCII").strip()


def _safe_label(lab: bytes) -> str:
    slabel = _label(lab)
    if slabel == ".":
        return "._"
    return slabel.replace("/", "|")


# regex parse C3D(.*) abaqus element type codes
_c3dre = re.compile(r"(?P<type>C3D(?P<nodes>\d+))(?P<subtype>[A-Z]*)")


def _proface_eltype(abq: str) -> str | None:
    mo = _c3dre.fullmatch(abq)
    if mo is None:
        return None
    return mo["type"]


def _find_step_inc(stepdata, step, inc) -> int:
    """search for requested step/increment"""

    if len(stepdata) == 0:
        msg = "no stepdata in file"
        raise ValueError(msg)
    logger.debug("Requested: Step %d, increment %d", step, inc)
    if step > 0 and step not in stepdata["step"]:
        # explicit request of inesistent step
        msg = f"step '{step}' not found"
        raise ValueError(msg)
    if step == 0:
        # last step requested
        step = stepdata["step"][-1]

    c_step = stepdata[stepdata["step"] == step]
    if inc > 0 and inc not in c_step["incr"]:
        # explicit request for inesistent increment
        msg = f"increment '{inc}' not found in step '{step}'"
        raise ValueError(msg)
    if inc == 0:
        # last increment requested
        inc = c_step["incr"][-1]

    assert stepdata.ndim == 1
    (i,) = np.nonzero((stepdata["step"] == step) & (stepdata["incr"] == inc))
    assert np.shape(i) == (1,), f"Multiple step blocks found: {i}"

    logger.debug("Found: Step %d, increment %d at position %s", step, inc, i)
    assert stepdata[i][["step", "incr"]].item() == (step, inc)
    return i.item()


def _guess_nr_ip(data):
    """heuristics to find the number of integration points
    for a loc 0 (integration points) record"""

    # assumption is that number of ip's is smallish
    num = data["num"]
    for i, v in enumerate(num):
        if v != num[0]:
            return i

    # edge case: record contains a single element
    assert np.all(num == num[0])
    return len(data)
