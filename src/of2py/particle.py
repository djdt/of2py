import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np


def load_of2i_sizes(path: str):
    x = np.loadtxt(
        path,
        delimiter=";",
        skiprows=1,
        usecols=(0, 1, 2, 3, 4, 5),
        dtype=[
            ("id", int),
            ("time", float),
            ("size", float),
            ("small", "U1"),
            ("large", "U1"),
            ("speed", float),
        ],
    )

    filter = np.logical_and(x["small"] != "S", x["large"] != "L")
    return x["size"][filter]


def init_parser(parser: argparse.ArgumentParser):
    parser.set_defaults(func=main)
    parser.add_argument(
        "files", nargs="+", type=Path, help="OF2iDetectedRawParticleList.csv file(s)"
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=20.0,
        metavar="NM",
        help="set the histogram bin size",
    )
    parser.add_argument("--legend", action="store_true", help="show a legend")


def main(args: argparse.Namespace):
    data = []
    labels = []

    for file in args.files:
        assert isinstance(file, Path)
        x = load_of2i_sizes(file)
        data.append(x)
        labels.append(f"{file.stem}")

    max = np.percentile(np.concatenate(data), 95)
    bins = np.arange(0.0, max, args.bin_size)

    plt.hist(data, bins, label=labels)

    if args.legend:
        plt.legend()

    plt.ylabel("Count")
    plt.xlabel("Size (nm)")
    plt.tight_layout()
    plt.show()
