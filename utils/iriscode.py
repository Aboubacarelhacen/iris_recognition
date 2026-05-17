"""
IrisCode feature extraction and matching.

Functions:
    apply_gabor_filters: Apply a 4-orientation Gabor filter bank.
    extract_iriscode:    Encode filter responses into a 2048-bit binary code.
    hamming_distance:    Rotation-tolerant normalized Hamming Distance.
    visualize_iriscode:  Reshape the bit-vector into a 2-D image for display.
"""

import cv2
import numpy as np


GABOR_ORIENTATIONS_DEG = [0, 45, 90, 135]
GABOR_FREQUENCY = 0.1   # cycles / pixel


def _build_gabor_kernel(theta_rad, frequency=GABOR_FREQUENCY, ksize=21):
    """Build an OpenCV Gabor kernel parameterized by orientation and frequency."""
    wavelength = 1.0 / max(frequency, 1e-6)
    sigma = 0.56 * wavelength            # standard half-octave bandwidth
    gamma = 0.5
    return cv2.getGaborKernel(
        (ksize, ksize), sigma, theta_rad, wavelength, gamma, 0, ktype=cv2.CV_32F
    )


def apply_gabor_filters(normalized_iris):
    """
    Apply a bank of 2-D Gabor filters at orientations 0°, 45°, 90°, 135°.

    Returns a list of filter responses (one per orientation), each the same
    shape as `normalized_iris`.
    """
    if normalized_iris is None:
        return []
    img = normalized_iris.astype(np.float32)
    responses = []
    for deg in GABOR_ORIENTATIONS_DEG:
        kernel = _build_gabor_kernel(np.deg2rad(deg))
        responses.append(cv2.filter2D(img, cv2.CV_32F, kernel))
    return responses


def extract_iriscode(normalized_iris):
    """
    Encode Gabor responses into a binary IrisCode.

    Strategy: apply the 4 Gabor filters, downsample each response to 256 bins
    (16 x 16), binarize (>0 → 1), and concatenate → 4 * 256 = 1024 bits per
    sign. We take the sign of both the filtered output and its 90°-shifted
    counterpart (Hilbert pair via the real/imag of a complex Gabor) to obtain
    a 2048-bit code in total.
    """
    if normalized_iris is None:
        return None

    img = normalized_iris.astype(np.float32)
    bits_per_orientation = 256                 # 16 x 16 grid
    grid = (16, 16)

    code_real, code_imag = [], []
    for deg in GABOR_ORIENTATIONS_DEG:
        theta = np.deg2rad(deg)
        # Real (cos) and imag (sin) Gabor — Daugman's quadrature pair
        k_real = cv2.getGaborKernel((21, 21), 5.6, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        k_imag = cv2.getGaborKernel((21, 21), 5.6, theta, 10.0, 0.5, np.pi / 2, ktype=cv2.CV_32F)
        r_real = cv2.filter2D(img, cv2.CV_32F, k_real)
        r_imag = cv2.filter2D(img, cv2.CV_32F, k_imag)
        r_real = cv2.resize(r_real, grid, interpolation=cv2.INTER_AREA)
        r_imag = cv2.resize(r_imag, grid, interpolation=cv2.INTER_AREA)
        code_real.append((r_real > 0).astype(np.uint8).ravel())
        code_imag.append((r_imag > 0).astype(np.uint8).ravel())

    iriscode = np.concatenate(code_real + code_imag)  # 8 * 256 = 2048 bits
    return iriscode.astype(np.uint8)


def _shift_code(code, shift, angular_blocks=16):
    """
    Circularly shift the IrisCode by `shift` angular blocks.

    The 2048-bit code is laid out as 8 chunks of 256 bits (16x16 grids).
    A shift of `s` columns rotates each grid by `s` along the angular axis.
    """
    if shift == 0:
        return code
    chunk = 16 * 16
    n_chunks = code.size // chunk
    out = np.empty_like(code)
    for c in range(n_chunks):
        block = code[c * chunk:(c + 1) * chunk].reshape(16, 16)
        block = np.roll(block, shift, axis=1)
        out[c * chunk:(c + 1) * chunk] = block.ravel()
    return out


def hamming_distance(code1, code2):
    """
    Rotation-tolerant normalized Hamming Distance ∈ [0, 1].

    Tries shifts of {-2, -1, 0, +1, +2} angular columns and returns the
    minimum HD, which compensates for in-plane rotation of the eye.
    """
    if code1 is None or code2 is None or code1.size != code2.size:
        return 1.0
    best = 1.0
    for s in (-2, -1, 0, 1, 2):
        shifted = _shift_code(code2, s)
        hd = float(np.mean(code1 != shifted))
        if hd < best:
            best = hd
    return best


def visualize_iriscode(iriscode):
    """Reshape the 2048-bit IrisCode into a 32x64 image for display."""
    if iriscode is None:
        return None
    n = iriscode.size
    # 2048 -> 32 x 64
    h = 32
    w = n // h
    return (iriscode[:h * w].reshape(h, w) * 255).astype(np.uint8)
