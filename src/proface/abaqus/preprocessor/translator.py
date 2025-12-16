# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

"""Translator implementation"""

import collections.abc
import logging
from pathlib import Path

import h5py
import numpy as np
from suanpan.abqfil import AbqFil

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
RES_DTYPE = np.dtype("float32")


class AbaqusTranslatorError(PreprocessorError):
    pass


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
    for elbloc in fil.elm:
        assert (elbloc["eltyp"] == elbloc["eltyp"][0]).all()
        eltype = _safe_label(elbloc["eltyp"][0])
        if eltype in h5_elements:
            label_m = _mangle_element_label(eltype, h5_elements.keys())
            logger.warning(
                "Mapping %s \N{RIGHTWARDS ARROW FROM BAR} %s",
                eltype,
                label_m,
            )
            eltype = label_m
        h5_elgroup = h5_elements.create_group(eltype)
        h5_elgroup.create_dataset("incidences", data=elbloc["ninc"])
        h5_elgroup.create_dataset("numbers", data=elbloc["elnum"])

        h5_elgroup.create_dataset(
            "nodes", data=np.unique(h5_elgroup["incidences"])
        )


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
        flag, elset, eltype, data = (
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
        eltype = _safe_label(eltype)
        if not eltype.startswith("C3D"):
            logger.warning("Results for element %s ignored", eltype)
            continue

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


def _mangle_element_label(label: str, groups) -> str:
    assert label in groups
    i = 1
    for s in groups:
        if s.startswith(f"{label:s}@"):
            i += 1
    mangled = f"{label:2}@{i:d}"
    assert mangled not in groups
    return mangled


def _find_step_inc(stepdata, step, inc):
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
    i = i.item()

    logger.debug("Found: Step %d, increment %d at position %s", step, inc, i)
    assert stepdata[i][["step", "incr"]].item() == (step, inc)
    return i


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
