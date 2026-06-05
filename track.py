import numpy as np
import scipy.ndimage as ndi
import argparse
import PIL.Image


px_to_shift = np.polynomial.Polynomial([8.1583, 1.5711, 2.8063e-4])


class Particle(object):
    def __init__(self, id: int, frame: int, pos: np.ndarray):
        assert pos.size == 2
        self.id = id
        self.positions = {frame: pos}

        self.current_pos = pos

    def addFrame(self, frame: int, pos: np.ndarray):
        self.positions[frame] = pos
        self.current_pos = pos

    def distance(self, other: "Particle") -> float:
        return float(np.linalg.norm(self.current_pos - other.current_pos))


def detect_particles(image: np.ndarray, threshold: float) -> np.ndarray:
    thresh = ndi.binary_opening(image[-150:-50] > threshold)
    labels, nlabels = ndi.label(thresh)
    centers = ndi.center_of_mass(
        image[-150:-50], labels, index=np.arange(1, nlabels + 1)
    )
    if len(centers) == 0:
        return np.array([])
    centers = np.array(centers)
    centers[:, 0] += image.shape[0] - 150
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
    spectra = np.roll(spectra, shift, axis=0)
    spectra[:shift] = 0.0
    return spectra[::-1]


def main(args: argparse.Namespace):
    import cv2

    images = PIL.Image.open(args.video)
    frame = 0

    if args.show:
        cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    if args.record is not None:
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

    while True:
        try:
            images.seek(frame)
            image = np.array(images)
        except EOFError:
            print(f"{args.video} :: end of file")
            break

        for pos in detect_particles(image, args.threshold):
            new = Particle(particle_id, frame, pos)
            particle_id += 1
            is_new = True

            for old in tracked_particles:
                if new.distance(old) < args.distance:
                    old.addFrame(frame, pos)
                    is_new = False
                    continue

            if is_new:
                tracked_particles.append(new)

        # remove particles that have exited frame
        for particle in tracked_particles:
            if frame not in particle.positions:
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
                cv2.rectangle(x, p0, p1, (0, 0, 255), 3)

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
        exit()

    exited_particles.extend(tracked_particles)

    if args.output is not None:
        with open(args.output, "w") as fp:
            fp.write("id,frame,y,x\n")
            for particle in exited_particles:
                for frame, pos in particle.positions.items():
                    fp.write(f"{particle.id},{frame},{pos[0]:.4f},{pos[1]:.4f}\n")

    if args.spectra is not None:
        # background = extract_background(images, exited_particles, args.spectra_width)
        spectras = {}
        for particle in exited_particles:
            for frame, pos in particle.positions.items():
                images.seek(frame)
                image = np.array(images)
                if args.smooth is not None:
                    image = ndi.gaussian_filter(image, args.smooth)
                array = spectras.get(particle.id, [])
                array.append(read_spectra(image, pos, args.spectra_width))
                spectras[particle.id] = array

        if args.spectra.suffix.lower() == ".npz":
            out = {f"p{id}": np.stack(val, axis=0) for id, val in spectras.items()}
            np.savez_compressed(args.spectra, **out)  # type: ignore
        elif args.spectra.suffix.lower() == ".csv":
            data = np.stack([np.mean(val, axis=0) for val in spectras.values()], axis=0)
            np.savetxt(args.spectra, data)
        else:
            raise ValueError("unknown file type for spectra, must be .csv or .npz")
