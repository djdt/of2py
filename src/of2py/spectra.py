from scipy.ndimage import gaussian_filter1d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import argparse

import logging


def read_of2py_file(file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(file, "r") as fp:
        header = fp.readline()
        x = np.loadtxt(
            fp,
            delimiter=",",
            dtype=[("id", int), ("frames", int), ("spectra", float, 2304)],
        )
    shifts = np.fromiter((t[6:] for t in header.split(",")[2:]), dtype=float)
    return shifts, x


def read_raman_spectra(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Raman spectra file.

    Args:
        path: path to csv file

    Returns:
        array of shifts (/cm), structured array of spectra
    """

    def brave_timestamp(x: str) -> np.datetime64:
        return np.datetime64(x[:-4]) + np.timedelta64(int(x[-3:]), "ms")

    dtype = [
        ("id", int),
        ("timestamp", "datetime64[ms]"),
        ("cluster", "U16"),
        ("confidence", float),
        ("frames", int),
        ("spectra", float, 2304),
    ]

    with path.open("r") as fp:
        line = fp.readline()
        shift_header = line.split(";")[5:]
        if len(shift_header) != 2304:
            raise ValueError(f"expected length 2304, not {len(shift_header)}")
        shifts = np.array(
            [float(s[s.find("[") + 1 : s.rfind("]")]) for s in shift_header]
        )

        return shifts, np.loadtxt(
            fp, delimiter=";", converters={1: brave_timestamp}, dtype=dtype
        )


def read_raman_single_spectra(
    path: Path,
    mode: str = "subtracted",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a Raman single spectra file.

    Data can be returned with or without background subtraction depending on `mode`.
    Passing 'raw' returns un-corrected spectra, 'subtracted' returns background subtracted spectra
    and 'background' return the background spectra for each particle.

    Args:
        path: path to csv file
        mode: one of 'raw', 'subtracted', 'background'

    Returns:
        array of shifts (/cm), structured array of single spectra
    """
    if mode not in ["raw", "subtracted", "background"]:
        raise ValueError("mode must be one of 'raw', 'subtracted', 'background'")
    dtype = [
        ("id", int),
        ("frame", int),
        ("type", "U1"),
        ("cluster", "U16"),
        ("confidence", float),
        ("pos", int),
        ("spectra", float, 2304),
    ]
    with path.open("r") as fp:
        line = fp.readline()
        shift_header = line.split(";")[6:]
        if len(shift_header) != 2304:
            raise ValueError(f"expected length 2304, not {len(shift_header)}")
        shifts = np.array(
            [float(s[s.find("[") + 1 : s.rfind("]")]) for s in shift_header]
        )

        data = np.loadtxt(fp, delimiter=";", dtype=dtype)

    y = data[data["type"] == "B"]  # backgrounds
    x = data[data["type"] == "S"]  # subtracted

    if mode == "raw":
        spectra = x["spectra"] + y["spectra"]
    elif mode == "subtracted":
        spectra = x["spectra"]
    else:  # mode == "background"
        spectra = y["spectra"]

    x["spectra"] = spectra

    for id in np.unique(x["id"]):
        x[x["id"] == id]["cluster"] = x[x["id"] == id][-1]["cluster"]
        x[x["id"] == id]["confidence"] = x[x["id"] == id][-1]["confidence"]

    return shifts, x


def reduce_raman_single_spectra(x: np.ndarray) -> np.ndarray:
    """Reduce a single spectra file.

    Convert data read with `read_raman_single_spectra` to a format matching `read_raman_spectra`.
    Sums spectra with the same ID and adds the number of recorded frames.

    """
    ids, counts = np.unique(x["id"], return_counts=True)
    reduced_dtype = [
        ("id", int),
        ("frames", int),
        ("cluster", "U16"),
        ("confidence", float),
        ("spectra", float, 2304),
    ]
    reduced = np.empty(len(ids), dtype=reduced_dtype)
    for i, (id, count) in enumerate(zip(ids, counts)):
        reduced[i]["id"] = id
        reduced[i]["frames"] = count
        reduced[i]["cluster"] = x[x["id"] == id][-1]["cluster"]
        reduced[i]["confidence"] = x[x["id"] == id][-1]["confidence"]
        reduced[i]["spectra"] = np.sum(x[x["id"] == id]["spectra"], axis=0)

    return reduced


def label_peaks(ax, xs: np.ndarray, ys: np.ndarray, peaks: np.ndarray):
    for peak in peaks:
        ax.annotate(
            f"{xs[peak]:.0f}",
            (xs[peak], ys[peak]),
            (0, 2),
            textcoords="offset points",
            va="baseline",
            ha="center",
        )


def init_parser(parser: argparse.ArgumentParser):
    parser.set_defaults(func=main)
    parser.add_argument("files", type=Path, nargs="+", help="CSV output(s) from of2py")
    # input
    parser.add_argument(
        "--mode",
        default="subtracted",
        choices=("raw", "subtracted", "background"),
        help="type of spectra to load from a single_spectra.csv",
    )
    # filtering
    parser.add_argument(
        "--cluster",
        metavar="NAME",
        type=str,
        help="only show spectra with this cluster",
    )
    parser.add_argument(
        "--frames",
        metavar="COUNT",
        type=int,
        help="minimum number of frames",
    )
    parser.add_argument("--id", type=int, help="only show particles with this ID")
    parser.add_argument(
        "--pos",
        type=float,
        nargs=2,
        metavar=("X0", "X1"),
        help="only show particles with an x pos in this range, requires single spectra",
    )
    # preprocessing
    parser.add_argument(
        "--normalise", action="store_true", help="normalise all spectra"
    )
    parser.add_argument(
        "--smooth",
        type=float,
        const=3.0,
        metavar="SIGMA",
        nargs="?",
        help="smooth spectra with Gaussian",
    )
    # arithmetic
    parser.add_argument("--sum", action="store_true", help="sum all spectra")
    parser.add_argument(
        "--mean", action="store_true", help="show a single mean spectra with stddev"
    )
    # display
    parser.add_argument(
        "--stack", action="store_true", help="stack plots instead of overlaying"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="show single spectra, not reduced if possible",
    )
    parser.add_argument(
        "--legend", action="store_true", help="show a legend for each plot"
    )
    parser.add_argument(
        "--label",
        type=float,
        nargs="+",
        help="add labels to peaks at these shifts",
    )


def main(args: argparse.Namespace):
    for file in args.files:
        assert isinstance(file, Path)
        header = file.open("r").readline()
        if "singleSpectraCount" in header:  # is raman_spectra format
            file_type = "brave"
            shifts, data = read_raman_spectra(file)
        elif "materialId" in header:  # still BRAVE format
            file_type = "brave_single"
            shifts, data = read_raman_single_spectra(file, mode=args.mode)
        else:  # assume of2py
            file_type = "of2py"
            shifts, data = read_of2py_file(file)

        if args.single and file_type != "brave_single":
            raise TypeError(
                "--single can only be used with a BRAVE single_spectra.csv file"
            )

        if args.id is not None:
            data = data[data["id"] == args.id]

        if args.cluster is not None:
            if file_type == "of2py":
                logging.warning(
                    "filtering by cluster not availble for 'of2py track' files"
                )
            else:
                data = data[data["cluster"] == args.cluster]

        if args.pos is not None:
            if file_type != "brave_single":
                raise TypeError("--pos requires a BRAVE single_spectra.csv file")
            data = data[
                np.logical_and(data["pos"] > args.pos[0], data["pos"] < args.pos[1])
            ]

        if not args.single:
            data = reduce_raman_single_spectra(data)

        # must be accessed after reduction
        if args.frames is not None:
            data = data[data["frames"] > args.frames]

        if data.size == 0:
            logging.warning(f"all spectra filtered for {file}")
            continue

        stddev = None
        if args.sum:
            spectra = np.sum(data["spectra"], axis=0)
        elif args.mean:
            spectra = np.mean(data["spectra"], axis=0)
            stddev = np.std(data["spectra"], mean=spectra, axis=0)
        else:
            spectra = data["spectra"]

        spectra = np.atleast_2d(spectra)

        if args.stack:
            _, axes = plt.subplots(
                spectra.shape[0], 1, squeeze=False, sharex=True, sharey=True
            )
            axes = axes.ravel()
        else:
            axes = [plt.gca()] * spectra.shape[0]

        for ax, spectrum in zip(axes, spectra):
            if args.smooth:
                spectrum = gaussian_filter1d(spectrum, sigma=args.smooth)
            if args.normalise:
                spectrum /= spectrum.max()

            ax.plot(shifts, spectrum, label=f"{file.stem}")
            if stddev is not None:
                ax.fill_between(shifts, spectrum - stddev, spectrum + stddev, alpha=0.5)

            if args.label is not None:
                peaks = np.searchsorted(shifts, args.label)
                label_peaks(ax, shifts, spectra, peaks)

    if args.legend:
        plt.legend()

    plt.tight_layout()
    plt.show()
