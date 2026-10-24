import cv2
import numpy as np
import random
import math
import json
import tifffile

# Video parameters
WIDTH, HEIGHT = 512, 512
FRAMES = 500
FPS = 30
OUTPUT_FILE = "pulsating_dots.tif"
DOT_LIST_FILE = "dot_list.json"

# Dot parameters
NUM_DOTS = 200
MOVING_PERCENTAGE = 0.1
STATIC_BRIGHT_PERCENTAGE = 0.05
MAX_DOT_SIZE = 5
MIN_DOT_SIZE = 3
MOVE_SPEED = 2

# Filter parameters
BLUR_KERNEL_SIZE = 15
BLUR_SIGMA = 5.0

# =============================================================================
# NOISE PARAMETERS
# =============================================================================
# Three SNR profiles. Adjust these to control difficulty.
#
# SNR formula:
#   S_peak = peak_intensity * PHOTON_SCALE
#   B      = BACKGROUND_LEVEL * PHOTON_SCALE
#   SNR    = S_peak / sqrt(S_peak + B + READ_NOISE_SIGMA^2)
#
# The noise pipeline applies:
#   1. Add constant background to the clean frame
#   2. Scale to photon counts (frame * PHOTON_SCALE)
#   3. Poisson sampling  (shot noise — variance equals mean)
#   4. Additive Gaussian  (read noise — camera electronics)
#   5. Clip negatives to zero

NOISE_PROFILES = {
    "high": {
        "photon_scale": 1000,
        "read_noise_sigma": 2.0,
        "background_level": 0.005,
    },
    "mid": {
        "photon_scale": 300,
        "read_noise_sigma": 5.0,
        "background_level": 0.02,
    },
    "low": {
        "photon_scale": 80,
        "read_noise_sigma": 8.0,
        "background_level": 0.05,
    },
}

# Select which profile to use
NOISE_PROFILE = "high"  # Change to "mid" or "low" for harder regimes


class Dot:
    def __init__(self, x, y, is_moving=False, is_static_bright=False):
        self.x = x
        self.y = y
        self.initial_x = x
        self.initial_y = y
        self.is_moving = is_moving
        self.is_static_bright = is_static_bright
        self.phase = random.uniform(0, 2 * math.pi)

        if is_moving:
            self.dx = random.uniform(-MOVE_SPEED, MOVE_SPEED)
            self.dy = random.uniform(-MOVE_SPEED, MOVE_SPEED)
        else:
            self.dx = 0
            self.dy = 0

    def update(self, frame_num):
        if self.is_moving:
            self.x += self.dx
            self.y += self.dy

            if self.x <= 0 or self.x >= WIDTH:
                self.dx *= -1
                self.x = max(0, min(WIDTH, self.x))
            if self.y <= 0 or self.y >= HEIGHT:
                self.dy *= -1
                self.y = max(0, min(HEIGHT, self.y))

    def get_size(self, frame_num):
        if self.is_static_bright:
            return MAX_DOT_SIZE * 2

        pulse = math.sin((frame_num * 0.2) + self.phase)
        size = MAX_DOT_SIZE * (pulse + 1) / 2
        return int(size)

    def is_true_event(self):
        return not self.is_moving and not self.is_static_bright

    def to_dict(self):
        return {
            "initial_x": self.initial_x,
            "initial_y": self.initial_y,
            "is_true_event": self.is_true_event(),
            "is_moving": self.is_moving,
            "is_static_bright": self.is_static_bright,
            "is_pulsating": not self.is_static_bright,
        }


def create_dots():
    dots = []
    num_moving = int(NUM_DOTS * MOVING_PERCENTAGE)
    num_static_bright = int(NUM_DOTS * STATIC_BRIGHT_PERCENTAGE)

    for i in range(num_moving):
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        dots.append(Dot(x, y, is_moving=True, is_static_bright=False))

    for i in range(num_static_bright):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        dots.append(Dot(x, y, is_moving=False, is_static_bright=True))

    num_pulsating = NUM_DOTS - num_moving - num_static_bright
    for i in range(num_pulsating):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        dots.append(Dot(x, y, is_moving=False, is_static_bright=False))

    return dots


def draw_clean_frame(dots, frame_num):
    """
    Render a single noise-free frame as a float32 image normalised to [0, 1].
    Identical to the original draw_frame but outputs float32 single-channel.
    """
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    for dot in dots:
        dot.update(frame_num)
        size = dot.get_size(frame_num)

        if size > 0:
            # Draw white circle on a temporary uint8 canvas, then accumulate
            tmp = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            cv2.circle(tmp, (int(dot.x), int(dot.y)), size, 255, -1)
            frame += tmp.astype(np.float32) / 255.0

    # Clip overlapping spots back to [0, 1]
    frame = np.clip(frame, 0.0, 1.0)

    # Gaussian blur (PSF approximation)
    frame = cv2.GaussianBlur(frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)

    return frame


def apply_noise(clean_frame, photon_scale, read_noise_sigma, background_level, rng):
    """
    Apply physically motivated noise model:
      1. Constant background
      2. Poisson shot noise
      3. Gaussian read noise
    """
    # 1. Add background
    frame = clean_frame + background_level

    # 2. Convert to expected photon counts
    mean_photons = np.clip(frame * photon_scale, 0, None)

    # 3. Poisson shot noise (variance = mean for photon statistics)
    photon_counts = rng.poisson(mean_photons).astype(np.float32)

    # 4. Additive Gaussian read noise (camera electronics)
    read_noise = rng.normal(0.0, read_noise_sigma, photon_counts.shape).astype(np.float32)

    # 5. Combine and clip
    noisy_frame = np.clip(photon_counts + read_noise, 0, None)

    return noisy_frame


def calculate_snr(photon_scale, read_noise_sigma, background_level):
    """
    Estimate SNR for a peak-intensity (=1.0) spot after PSF convolution.

    SNR = S_peak / sqrt(S_peak + B + sigma_read^2)

    where S_peak and B are in photon-count units.
    """
    # Peak pixel value after Gaussian blur of a unit-intensity dot
    # For a Gaussian PSF: peak = 1 / (2*pi*sigma^2) * area_of_circle
    # Approximation: peak attenuation from blur
    psf_peak = 1.0 / (2 * math.pi * BLUR_SIGMA ** 2)

    S_peak = psf_peak * photon_scale
    B = background_level * photon_scale

    snr = S_peak / math.sqrt(S_peak + B + read_noise_sigma ** 2)
    return snr


def save_dot_list(dots, path=DOT_LIST_FILE, verbose=True):
    dot_list = [dot.to_dict() for dot in dots]

    with open(path, "w") as f:
        json.dump(dot_list, f, indent=2)

    if verbose:
        true_events = sum(1 for dot in dots if dot.is_true_event())
        false_events = len(dots) - true_events
        print(f"\nDot list saved to {path}")
        print(f"Summary:")
        print(f"  - True events (stationary & pulsating): {true_events}")
        print(f"  - False events (moving or static bright): {false_events}")
        print(f"  - Total dots: {len(dots)}")


def save_labels(dots, positive_json, negative_json):
    """
    Emit detector-format ground truth, split into positive / negative spots.

    Positives = true events (stationary & pulsating). Negatives = everything else
    (moving or static-bright). Written as ``[{"center": {"x": .., "y": ..}}, ..]``
    -- the exact shape the SCP parametrizer's extract_center_coordinates() reads,
    so simulated and GUI-annotated movies load through one code path.
    """
    positives, negatives = [], []
    for dot in dots:
        entry = {"center": {"x": dot.initial_x, "y": dot.initial_y}}
        (positives if dot.is_true_event() else negatives).append(entry)

    with open(positive_json, "w") as f:
        json.dump(positives, f, indent=2)
    with open(negative_json, "w") as f:
        json.dump(negatives, f, indent=2)
    return len(positives), len(negatives)


def generate(seed, out_tif, positive_json=None, negative_json=None,
             dotlist_json=None, frames=FRAMES, noise_profile=NOISE_PROFILE,
             verbose=True):
    """
    Generate one reproducible smTIRF movie and its ground-truth labels.

    Passing the same ``seed`` reproduces the same movie exactly: it seeds BOTH
    RNGs the simulation uses -- Python's ``random`` (dot placement, phase, drift)
    and NumPy's Generator (Poisson + read noise). Different seeds give independent
    movies, which is what lets us build train / validation sets.

    Returns a summary dict (seed, counts, snr, output paths).
    """
    # Seed both RNGs so the whole movie (geometry AND noise) is reproducible.
    random.seed(seed)
    rng = np.random.default_rng(seed)

    noise_cfg = NOISE_PROFILES[noise_profile]
    photon_scale = noise_cfg["photon_scale"]
    read_noise_sigma = noise_cfg["read_noise_sigma"]
    background_level = noise_cfg["background_level"]
    snr = calculate_snr(photon_scale, read_noise_sigma, background_level)

    if verbose:
        print(f"[seed {seed}] {NUM_DOTS} dots, {frames} frames, "
              f"profile={noise_profile} (SNR ~ {snr:.2f})")

    dots = create_dots()

    if dotlist_json:
        save_dot_list(dots, dotlist_json, verbose=verbose)
    n_pos = n_neg = None
    if positive_json and negative_json:
        n_pos, n_neg = save_labels(dots, positive_json, negative_json)
        if verbose:
            print(f"[seed {seed}] labels: {n_pos} positive, {n_neg} negative")

    frame_stack = []
    for frame_num in range(frames):
        if verbose and frame_num % 100 == 0:
            print(f"[seed {seed}] frame {frame_num}/{frames}")
        clean = draw_clean_frame(dots, frame_num)
        noisy = apply_noise(clean, photon_scale, read_noise_sigma, background_level, rng)
        frame_stack.append(noisy)

    frames_array = np.array(frame_stack, dtype=np.float32)
    tifffile.imwrite(out_tif, frames_array, imagej=True, metadata={"axes": "TYX"})
    if verbose:
        print(f"[seed {seed}] saved {out_tif}  shape={frames_array.shape}")

    return {
        "seed": seed,
        "tif": str(out_tif),
        "positive": str(positive_json) if positive_json else None,
        "negative": str(negative_json) if negative_json else None,
        "frames": frames,
        "noise_profile": noise_profile,
        "snr": snr,
        "num_positive": n_pos,
        "num_negative": n_neg,
    }


def main():
    # Single-movie script behaviour, now also emitting detector-format labels.
    # seed=None keeps the original non-deterministic output for ad-hoc runs;
    # use generate_dataset.py to build reproducible, seeded train/val sets.
    generate(
        seed=None,
        out_tif=OUTPUT_FILE,
        positive_json="pulsating_dots_positive.json",
        negative_json="pulsating_dots_negative.json",
        dotlist_json=DOT_LIST_FILE,
        frames=FRAMES,
        noise_profile=NOISE_PROFILE,
    )


if __name__ == "__main__":
    main()