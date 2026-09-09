import argparse
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import scipy.ndimage as ndi

px_to_shift = np.polynomial.Polynomial([8.1583, 1.5711, 2.8063e-4])

SPECTRA_OFFSET = 50  # the expected position for rayleigh scattering, i.e., 0 shift


class Particle:
    def __init__(self, id: int, frame: int, pos: np.ndarray, spectra: np.ndarray):
        assert pos.size == 2
        self.id = id
        self.tracked = {frame: (pos, spectra)}

        self.current_pos = pos
        self.current_frame = frame

    def addFrame(self, frame: int, pos: np.ndarray, spectra: np.ndarray):
        self.tracked[frame] = (pos, spectra)
        self.current_pos = pos
        self.current_frame = frame

    def distance(self, other: Particle) -> float:
        return float(np.linalg.norm(self.current_pos - other.current_pos))


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


def interpolate_background(image: np.ndarray, px: int, width: int = 3) -> np.ndarray:
    def interp_row(x: np.ndarray, xs: np.ndarray):
        return np.interp(xs, np.arange(x.size), x)

    bg = image[:, px - width * 5 : px + width * 5]
    xs = np.arange(px + width * 4, px + width * 6)
    return np.apply_along_axis(interp_row, 1, bg, xs)


def read_spectra(image: np.ndarray, pos: np.ndarray, width: int = 3) -> np.ndarray:
    px, py = int(pos[1]), int(pos[0])
    spectra = image[:, px - width : px + width]
    bg = interpolate_background(image, px, width)
    spectra = np.mean(spectra - bg, axis=1)
    shift = image.shape[1] - py
    spectra = np.roll(spectra, shift - SPECTRA_OFFSET, axis=0)
    # spectra[:shift] = 0.0
    return spectra[::-1]


def init_parser(parser: argparse.ArgumentParser):
    parser.set_defaults(func=main)
    parser.add_argument("video", type=Path, help="path to the .tiff video file")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--record", type=Path, help="save a video of the output of --show"
    )
    parser.add_argument("--output", type=Path, help="save tracked particles to csv")
    parser.add_argument(
        "--threshold",
        type=float,
        default=200.0,
        help="minimum value to detect a particle",
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
        "--distance",
        type=float,
        default=20.0,
        metavar="PIXELS",
        help="minimum distance between particles / maximum distance to track",
    )
    parser.add_argument(
        "--spectra-width",
        type=int,
        default=3,
        metavar="PIXELS",
        help="width of spectra to extract",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=[300, -300, -110, -10],
        metavar=("x", "width", "y", "height"),
        help="roi for particle extraction",
    )
    parser.add_argument(
        "--track-frames",
        type=int,
        default=5,
        help="number of frames a particle must disappear before being removed",
    )
    parser.add_argument(
        "--min-size", type=int, default=8, help="minmum size of particle, in pixels"
    )


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

    particle_id = 0
    exited_particles = []
    tracked_particles = []

    for i in range(4):
        if args.roi[i] < 0:
            args.roi[i] = (images.width if i < 2 else images.height) + args.roi[i]

    while True:
        try:
            images.seek(frame)
            image = np.array(images)
        except EOFError:
            print(f"{args.video} :: end of file")
            break

        for pos in detect_particles(image, args.threshold, args.roi, args.min_size):
            spectra = read_spectra(image, pos, args.spectra_width)
            new = Particle(particle_id, frame, pos, spectra)
            particle_id += 1
            is_new = True

            for old in tracked_particles:
                if new.distance(old) < args.distance:
                    old.addFrame(frame, pos, spectra)
                    is_new = False
                    continue

            if is_new:
                tracked_particles.append(new)

        # remove particles that have exited frame
        for particle in tracked_particles:
            if frame - particle.current_frame > args.track_frames:
                exited_particles.append(particle)
                tracked_particles.remove(particle)

        if args.show or args.record is not None:
            x = np.clip(image, 0.0, np.percentile(image, 90))
            x = cv2.normalize(x, None, 1, 0, cv2.NORM_MINMAX)
            x = np.uint8(x * 255.0)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            for particle in tracked_particles:
                p0 = (
                    int(particle.current_pos[1]) - 3,
                    int(particle.current_pos[0]) - 3,
                )
                p1 = (
                    int(particle.current_pos[1]) + 3,
                    int(particle.current_pos[0]) + 3,
                )
                c = (0, 0, 255) if particle.current_frame == frame else (255, 0, 0)
                cv2.rectangle(x, p0, p1, c, 3)
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
    # end while
    if args.show:
        sys.exit()

    exited_particles.extend(tracked_particles)

    if args.output is not None:
        with open(args.output, "w") as fp:
            fp.write("id,frame,y,x\n")
            for particle in exited_particles:
                for frame, (pos, spectra) in particle.tracked.items():
                    images.seek(frame)
                    fp.write(
                        f"{particle.id},{frame},{pos[0]:.4f},{pos[1]:.4f},{','.join(f'{s:.4f}' for s in spectra)}\n"
                    )
    #
    # if args.spectra is not None:
    #     spectras = {}
    #     for particle in exited_particles:
    #         for frame, (pos, _) in particle.tracked.items():
    #             images.seek(frame)
    #             image = np.array(images)
    #             if args.smooth is not None:
    #                 image = ndi.gaussian_filter(image, args.smooth)
    #             array = spectras.get(particle.id, [])
    #             array.append(read_spectra(image, pos, args.spectra_width))
    #             spectras[particle.id] = array
    #
    #     if args.spectra.suffix.lower() == ".npz":
    #         out = {f"p{id}": np.stack(val, axis=0) for id, val in spectras.items()}
    #         np.savez_compressed(args.spectra, **out)  # type: ignore
    #     elif args.spectra.suffix.lower() == ".csv":
    #         data = np.stack([np.mean(val, axis=0) for val in spectras.values()], axis=0)
    #         np.savetxt(args.spectra, data)
    #     else:
    #         raise ValueError("unknown file type for spectra, must be .csv or .npz")
