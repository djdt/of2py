import argparse
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import PIL.Image
import scipy.ndimage as ndi

px_to_shift = np.polynomial.Polynomial([8.1583, 1.5711, 2.8063e-4])

SPECTRA_OFFSET = 50  # the expected position for rayleigh scattering, i.e., 0 shift


class Particle:
    def __init__(self, id: int, frame: int, pos: np.ndarray):
        assert pos.size == 2
        self.id = id
        # self.tracked = {frame: (pos, spectra)}

        self.frames = [frame]
        self.positions = [pos]
        self.spectra = []

    def distance(self, other: Particle) -> float:
        return float(np.linalg.norm(self.positions[-1] - other.positions[-1]))


def detect_particles(
    image: np.ndarray,
    threshold: float,
    roi: tuple[int, int, int, int],
    minimum_size: int = 10,
) -> np.ndarray:

    image_roi = image[roi[2] : roi[3], roi[0] : roi[1]]

    thresh = ndi.binary_closing(image_roi > threshold)
    labels, nlabels = ndi.label(thresh)
    centers = ndi.center_of_mass(image_roi, labels, index=np.arange(1, nlabels + 1))
    counts = np.bincount(labels.flat)[1:]
    centers = np.asanyarray(centers)

    valid = counts > minimum_size
    centers = centers[valid]

    if centers.size > 0:
        centers += (roi[2], roi[0])
    return centers


def interpolate_background(
    image: np.ndarray, positions: list, width: int = 10
) -> np.ndarray:
    def interp_row_nans(x: np.ndarray, mask: np.ndarray):
        x[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), x[mask])
        return x

    mask = np.ones(image.shape[1], dtype=bool)
    for _, pos in np.around(positions).astype(int):
        mask[pos - width // 2 : pos + width // 2] = False

    return np.apply_along_axis(interp_row_nans, 1, image.astype(float), mask=mask)


def read_spectra(
    image: np.ndarray, pos: np.ndarray, background: np.ndarray, width: int = 3
) -> np.ndarray:
    py, px = np.around(pos).astype(int)
    spectra = np.mean(image[:, px - width // 2 : px + width // 2 + 1], axis=1)
    spectra_bg = np.mean(background[:, px - width // 2 : px + width // 2 + 1], axis=1)

    shift = image.shape[1] - py
    spectra = np.roll(spectra - spectra_bg, shift - SPECTRA_OFFSET, axis=0)
    return spectra[::-1]


def roll_along_axis(x: np.ndarray, shifts: np.ndarray, axis: int = 0) -> np.ndarray:
    if shifts.size != x.shape[axis]:
        raise ValueError("shifts must be size of x in rolling axis")
    if np.any(shifts < 0):
        raise ValueError("all shifts must be positive")

    x = np.swapaxes(x, axis, -1)
    xx = np.concatenate((x, x), axis=1)

    view = np.lib.stride_tricks.as_strided(
        xx,
        shape=(x.shape[0], x.shape[1], x.shape[1]),
        strides=(xx.strides[0], xx.strides[1], xx.strides[1]),
    )
    x = view[np.arange(x.shape[0]), x.shape[1] - shifts - 1]
    return np.swapaxes(x, -1, axis)


def init_parser(parser: argparse.ArgumentParser):
    parser.set_defaults(func=main)
    parser.add_argument("video", type=Path, help="path to the .tiff video file")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--record", type=Path, help="save a video of the output of --show"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="save tracked particles to csv"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=200.0,
        help="minimum value to detect a particle",
    )
    parser.add_argument(
        "--spectra-width",
        type=int,
        default=3,
        metavar="PIXELS",
        help="width of spectra to extract",
    )
    parser.add_argument(
        "--background-width",
        type=int,
        default=11,
        metavar="PIXELS",
        help="width of background to blank",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=[500, -500, -110, -10],
        metavar=("x", "width", "y", "height"),
        help="roi for particle extraction",
    )
    parser.add_argument(
        "--track-distance",
        type=float,
        default=20.0,
        metavar="PIXELS",
        help="minimum distance between particles / maximum distance to track",
    )
    parser.add_argument(
        "--track-frames",
        type=int,
        default=5,
        help="number of frames a particle must disappear before being removed",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        metavar="SIGMA",
        nargs="?",
        const=1.0,
        help="smooth video with Gaussian before processing",
    )
    parser.add_argument(
        "--min-size", type=int, default=8, help="minmum size of particle, in pixels"
    )
    parser.add_argument(
        "--image-offsets",
        type=Path,
        help="path to a numpy array of shifts for each image row. "
        "Useful when the Raman camera is out of alignment",
    )
    # parser.add_argument(
    #     "--calibration", type=Path, help="path to a Raman shift calibration file"
    # )


def main(args: argparse.Namespace):

    images = PIL.Image.open(args.video)
    frame = 0

    if args.show:
        import cv2

        cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    if args.record is not None:
        import cv2

        writer = cv2.VideoWriter(
            args.record,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (images.width, images.height),
            True,
        )
    offsets = None
    if args.image_offsets is not None:
        offsets = np.load(args.image_offsets)

    particle_id = 0
    exited_particles = []
    tracked_particles = []

    for i in range(4):
        if args.roi[i] < 0:
            args.roi[i] = (images.width if i < 2 else images.height) + args.roi[i]

    print(
        f"{args.video} :: {images.width} x {images.height} :: {images.n_frames} frames :: tracking particles"
    )

    while True:
        try:
            images.seek(frame)
            image = np.array(images)
        except EOFError:
            print(f"\n{args.video} :: end of file")
            break

        if offsets is not None:
            image = roll_along_axis(image, offsets, 1)

        for pos in detect_particles(image, args.threshold, args.roi, args.min_size):
            new = Particle(particle_id, frame, pos)
            particle_id += 1
            is_new = True

            for old in tracked_particles:
                if new.distance(old) < args.track_distance:
                    old.frames.append(frame)
                    old.positions.append(pos)
                    is_new = False
                    break

            if is_new:
                tracked_particles.append(new)

        # remove particles that have exited frame
        for particle in tracked_particles:
            if frame - particle.frames[-1] > args.track_frames:
                exited_particles.append(particle)
                tracked_particles.remove(particle)

        # mask out all particles and interpolate the background over them
        background = interpolate_background(
            image,
            [particle.positions[-1] for particle in tracked_particles],
            width=args.background_width,
        )
        # extract spectra
        for particle in tracked_particles[:]:
            spectra = read_spectra(
                image, particle.positions[-1], background, args.spectra_width
            )
            particle.spectra.append(spectra)

        if args.show or args.record is not None:
            x = np.clip(image, 0.0, np.percentile(image, 90))
            x = (cv2.normalize(x, None, 1, 0, cv2.NORM_MINMAX) * 255.0).astype(np.uint8)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

            color = (0, 0, 255)
            missing_color = (255, 0, 0)

            for particle in tracked_particles:
                current = particle.frames[-1] == frame
                pos = particle.positions[-1]
                p0 = (int(pos[1]) - 5, int(pos[0]) - 5)
                p1 = (int(pos[1]) + 5, int(pos[0]) + 5)
                cv2.rectangle(x, p0, p1, color if current else missing_color, 3)
            cv2.rectangle(
                x,
                (args.roi[0], args.roi[2]),
                (args.roi[1], args.roi[3]),
                (0, 255, 0),
                1,
            )

            if args.record is not None:
                writer.write(x)
            if args.show:
                cv2.imshow("win", x)
                key = cv2.waitKey(10000)
                if key == ord("q"):
                    break
                elif key == ord("a") and frame > 0:
                    frame -= 1
                    continue
        frame += 1
        print(f"\tframe {frame} :: {len(tracked_particles)} active particles", end="\r")

    # end while
    if args.show:
        sys.exit()

    exited_particles.extend(tracked_particles)
    exited_particles = sorted(exited_particles, key=lambda p: p.id)

    shifts = np.arange(images.height) - SPECTRA_OFFSET

    if args.output is not None:
        if args.output.suffix == ".npz":
            size = np.sum([len(p.frames) for p in exited_particles])
            data = np.empty(
                size,
                dtype=[
                    ("id", int),
                    ("frame", int),
                    ("xpos", float),
                    ("ypos", float),
                    ("spectra", float, 2304),
                ],
            )
            i = 0
            for particle in exited_particles:
                for frame, pos, spectra in zip(
                    particle.frames, particle.positions, particle.spectra
                ):
                    data[i] = (particle.id, frame, pos[1], pos[0], spectra)
                    i += 1
            np.savez_compressed(args.output, particles=data, shifts=shifts)
        else:
            with open(args.output, "w") as fp:
                fp.write(f"#of2py track v{version('of2py')}")
                fp.write(
                    f"id,frame,xpos,ypos,{','.join(f'shift[{s:.2f}]' for s in shifts)}\n"
                )
                for particle in exited_particles:
                    for frame, pos, spectra in zip(
                        particle.frames, particle.positions, particle.spectra
                    ):
                        fp.write(
                            f"{particle.id},{frame},{pos[1]:.2f},{pos[0]:.2f},{','.join(f'{s:.6g}' for s in spectra)}\n"
                        )
