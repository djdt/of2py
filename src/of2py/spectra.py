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


def read_raman_spectra_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def read_raman_single_spectra_file(
    path: Path, backgrounds: bool = False
) -> tuple[np.ndarray, np.ndarray]:
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

        x = np.loadtxt(fp, delimiter=";", dtype=dtype)

    y = x[x["type"] == "B"]
    x = x[x["type"] == "S"]  # remove backgrounds
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
        if backgrounds:
            reduced[i]["spectra"] = np.mean(y[y["id"] == id]["spectra"], axis=0)
        else:
            reduced[i]["spectra"] = np.mean(x[x["id"] == id]["spectra"], axis=0)

    return shifts, reduced


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
    parser.add_argument(
        "--cluster",
        metavar="NAME",
        type=str,
        help="only show spectra with this cluster",
    )

    parser.add_argument(
        "--frames",
        metavar="COUNT",
        default=20,
        type=int,
        help="minimum number of frames",
    )
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
    parser.add_argument(
        "--single", action="store_true", help="only output the spectra with most frames"
    )
    parser.add_argument("--sum", action="store_true", help="sum all spectra")
    parser.add_argument(
        "--mean", action="store_true", help="show a single mean spectra with stddev"
    )
    parser.add_argument(
        "--stack", action="store_true", help="stack plots instead of overlaying"
    )
    # parser.add_argument(
    #     "--remove-background",
    #     action="store_true",
    #     help="remove background and fluorescence",
    # )
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
            shifts, x = read_raman_spectra_file(file)
        elif "materialId" in header:  # still BRAVE format
            file_type = "brave_single"
            shifts, x = read_raman_single_spectra_file(file)
        else:  # assume of2py
            file_type = "of2py"
            shifts, x = read_of2py_file(file)

        if args.cluster is not None:
            if file_type == "of2py":
                logging.warning(
                    "filtering by cluster not availble for 'of2py track' files"
                )
            else:
                x = x[x["cluster"] == args.cluster]

        if args.frames is not None:
            x = x[x["frames"] > args.frames]

        if x.size == 0:
            logging.warning(f"all spectra filtered for {file}")
            continue

        stddev = None
        if args.sum:
            spectra = np.sum(x["spectra"], axis=0)
        elif args.mean:
            spectra = np.mean(x["spectra"], axis=0)
            stddev = np.std(x["spectra"], mean=spectra, axis=0)
        else:
            spectra = x["spectra"]

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
