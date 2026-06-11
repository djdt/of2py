import argparse

from of2py import track, spectra, particle


def main():
    parser = argparse.ArgumentParser(
        "of2py",
        description="a set of tools for processing raw data from the BRAVE OF2i and OF2i-Raman setups",
    )
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(description="of2py processing subcommands")
    parser_track = subparsers.add_parser(
        "track",
        help="track and detect particles in OF2i-Raman raw videos",
    )
    track.init_parser(parser_track)

    parser_spectra = subparsers.add_parser(
        "spectra",
        help="process OF2i-Raman csv files created using 'spectra save' or the 'track' subcommand",
    )
    spectra.init_parser(parser_spectra)

    parser_particle = subparsers.add_parser(
        "particle",
        help="process OF2i particle list csv files",
    )
    particle.init_parser(parser_particle)

    args = parser.parse_args()
    if args.func is None:
        parser.error("subcommand required")
    args.func(args)


if __name__ == "__main__":
    main()
