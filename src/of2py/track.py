import argparse
import sys
from collections.abc import Generator
from importlib.metadata import version
from pathlib import Path

import numpy as np
import PIL.Image
import scipy.ndimage as ndi


class Particle:
    ID_COUNTER = 0

    def __init__(self, frame: int, image: np.ndarray, offset: tuple[int, int]):
        self.id = Particle.ID_COUNTER
        Particle.ID_COUNTER += 1

        self.images = {frame: (image, offset)}
        self.spectra = {}

    def setFrame(self, frame: int, image: np.ndarray, offset: tuple[int, int]):
        self.images[frame] = (image, offset)

    def lastFrame(self) -> int:
        return list(self.images)[-1]

    def distance(self, other: Particle) -> float:
        return float(np.linalg.norm(np.asanyarray(self.position()) - other.position()))

    def intensity(self, frame: int | None = None) -> float:
        if frame is None:
            frame = self.lastFrame()
        return np.sum(self.images[frame][0])

    def position(self, frame: int | None = None) -> tuple[float, float]:
        if frame is None:
            frame = self.lastFrame()
        return np.asanyarray(self.images[frame][1]) + ndi.center_of_mass(
            self.images[frame][0]
        )

    def size(self, frame: int | None = None) -> int:
        if frame is None:
            frame = self.lastFrame()
        return int(np.count_nonzero(self.images[frame][0]))

    def fwhm(self, frame: int | None = None) -> int:
        if frame is None:
            frame = self.lastFrame()
        hmax = np.amax(self.images[frame][0]) / 2.0
        _, c = np.nonzero(self.images[frame][0] < hmax)
        fwhm = np.amax(np.diff(c) - 1)
        return fwhm


def detect_particles(
    image: np.ndarray, threshold: float, roi: tuple[int, int, int, int]
) -> Generator[tuple[np.ndarray, tuple[int, int]]]:

    image_roi = image[roi[2] : roi[3], roi[0] : roi[1]]

    thresh = ndi.binary_closing(image_roi > threshold)
    labels, _ = ndi.label(thresh)
    slices = ndi.find_objects(labels)
    for i, (sx, sy) in enumerate(slices):
        particle = image_roi[sx, sy].copy()
        particle[labels[sx, sy] != i + 1] = 0.0
        yield particle, (sx.start + roi[2], sy.start + roi[0])


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
    image: np.ndarray,
    pos: np.ndarray,
    background: np.ndarray,
    width: int = 3,
    rayleigh_offset: int = 0,
) -> np.ndarray:
    py, px = np.around(pos).astype(int)
    spectra = np.mean(image[:, px - width // 2 : px + width // 2 + 1], axis=1)
    spectra_bg = np.mean(background[:, px - width // 2 : px + width // 2 + 1], axis=1)

    shift = image.shape[1] - py
    spectra = np.roll(spectra - spectra_bg, shift - rayleigh_offset, axis=0)
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
        default=21,
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
    parser.add_argument(
        "--calibration",
        type=Path,
        help="path to a Raman shift calibration file, usually found in /usr/share/braveanalytics",
    )


def main(args: argparse.Namespace):

    images = PIL.Image.open(args.video)
    assert hasattr(images, "n_frames")  # multi page tiff
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

    if args.calibration is not None:
        shifts = np.loadtxt(args.calibration, delimiter=";")
        if len(shifts) != images.height:
            raise ValueError(
                f"calibration vector should be {images.height}, not {len(shifts)}"
            )
    else:
        shifts = np.arange(images.height)

    rayleigh_offset = int(np.searchsorted(shifts, 0.0))

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

        for particle_image, offset in detect_particles(image, args.threshold, args.roi):
            new = Particle(frame, particle_image, offset)
            if new.size() < args.min_size:
                continue

            dists = [new.distance(old) for old in tracked_particles]
            if len(dists) == 0:
                tracked_particles.append(new)
            else:
                closest = np.argmin(dists)
                if dists[closest] < args.track_distance:
                    old = tracked_particles[closest]
                    if frame != old.lastFrame() or new.intensity() > old.intensity():
                        old.setFrame(frame, particle_image, offset)
                else:
                    tracked_particles.append(new)

        # remove particles that have exited frame
        for particle in tracked_particles:
            if frame - particle.lastFrame() > args.track_frames:
                exited_particles.append(particle)
                tracked_particles.remove(particle)

        # mask out all particles and interpolate the background over them
        background = interpolate_background(
            image,
            [particle.position() for particle in tracked_particles],
            width=args.background_width,
        )
        # extract spectra
        for particle in tracked_particles:
            spectra = read_spectra(
                image,
                particle.position(),
                background,
                # particle.fwhm(),
                args.spectra_width,
                rayleigh_offset,
            )
            # may be a frame where particle is not tracked, thats ok
            particle.spectra[frame] = spectra

        if args.show or args.record is not None:
            x = np.clip(image, 0.0, np.percentile(image, 90))
            x = (cv2.normalize(x, None, 1, 0, cv2.NORM_MINMAX) * 255.0).astype(np.uint8)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

            color = (0, 0, 255)
            missing_color = (255, 0, 0)

            for particle in tracked_particles:
                current = particle.lastFrame() == frame
                pos = particle.position()
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

    if args.output is not None:
        if args.output.suffix == ".npz":
            size = np.sum([len(p.images) for p in exited_particles])
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
                for frame in particle.images:
                    pos = particle.position(frame)
                    spectra = particle.spectra[frame]
                    data[i] = (particle.id, frame, pos[1], pos[0], spectra)
                    i += 1
            np.savez_compressed(args.output, particles=data, shifts=shifts)
        else:
            with open(args.output, "w") as fp:
                fp.write(f"#of2py track v{version('of2py')}\n")
                fp.write(
                    f"id,frame,xpos,ypos,{','.join(f'shift_{s:.2f}' for s in shifts)}\n"
                )
                for particle in exited_particles:
                    for frame in particle.images:
                        pos = particle.position(frame)
                        spectra = particle.spectra[frame]
                        fp.write(
                            f"{particle.id},{frame},{pos[1]:.2f},{pos[0]:.2f},{','.join(f'{s:.6g}' for s in spectra)}\n"
                        )
