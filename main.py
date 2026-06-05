from pathlib import Path
import argparse

import track
import spectra


def main():
    parser = argparse.ArgumentParser(
        "of2py",
        description="a set of tools for processing raw data from the BRAVE OF2i and OF2i-Raman setups",
    )
    subparsers = parser.add_subparsers(description="of2py processing subcommands")
    parser_track = subparsers.add_parser(
        "track",
        help="track and detect particles in OF2i-Raman raw videos",
    )
    parser_track.set_defaults(func=track.main)
    parser_track.add_argument("video", type=Path, help="path to the .tiff video file")
    parser_track.add_argument("--show", action="store_true")
    parser_track.add_argument(
        "--record", type=Path, help="save a video of the output of --show"
    )
    parser_track.add_argument(
        "--output", type=Path, help="save tracked particles to path"
    )
    parser_track.add_argument(
        "--spectra",
        type=Path,
        help="extract ramen spectra to numpy array",
    )
    parser_track.add_argument(
        "--threshold",
        type=float,
        default=1e3,
        help="minimum value to detect a particle",
    )
    parser_track.add_argument(
        "--smooth",
        type=float,
        metavar="SIGMA",
        nargs="?",
        const=1.0,
        help="smooth video with Gaussian before processing",
    )
    parser_track.add_argument(
        "--distance",
        type=float,
        default=20.0,
        metavar="PIXELS",
        help="minimum distance between particles / maximum distance to track",
    )
    parser_track.add_argument(
        "--spectra-width",
        type=int,
        default=3,
        metavar="PIXELS",
        help="width of spectra to extract",
    )

    parser_spectra = subparsers.add_parser(
        "spectra",
        help="process OF2i-Raman csv files created using 'spectra save' or the 'track' subcommand",
    )
    parser_spectra.set_defaults(func=spectra.main)
    parser_spectra.add_argument(
        "files", type=Path, nargs="+", help="CSV output(s) from of2py"
    )
    parser_spectra.add_argument(
        "--cluster",
        metavar="NAME",
        type=str,
        help="only show spectra with this cluster",
    )

    parser_spectra.add_argument(
        "--frames",
        metavar="COUNT",
        default=20,
        type=int,
        help="minimum number of frames",
    )
    parser_spectra.add_argument(
        "--normalise", action="store_true", help="normalise all spectra"
    )
    parser_spectra.add_argument(
        "--smooth",
        type=float,
        const=3.0,
        metavar="SIGMA",
        nargs="?",
        help="smooth spectra with Gaussian",
    )
    parser_spectra.add_argument(
        "--single", action="store_true", help="only output the spectra with most frames"
    )
    parser_spectra.add_argument("--sum", action="store_true", help="sum all spectra")
    parser_spectra.add_argument(
        "--mean", action="store_true", help="show a single mean spectra with stddev"
    )
    parser_spectra.add_argument(
        "--stack", action="store_true", help="stack plots instead of overlaying"
    )
    parser_spectra.add_argument(
        "--remove-background",
        action="store_true",
        help="remove background and fluorescence",
    )
    parser_spectra.add_argument(
        "--legend", action="store_true", help="show a legend for each plot"
    )
    parser_spectra.add_argument(
        "--label",
        type=float,
        nargs="+",
        help="add labels to peaks at these shifts",
    )

    parser_particle = subparsers.add_parser(
        "particle",
        help="process OF2i particle csv files",
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
