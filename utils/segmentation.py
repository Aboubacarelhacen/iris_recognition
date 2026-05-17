"""
Iris segmentation and normalization (Daugman's pipeline).

Functions:
    load_image:             Read grayscale iris image from disk.
    preprocess_image:       CLAHE histogram equalization.
    detect_iris_and_pupil:  HoughCircles-based detection of iris/pupil boundaries.
    normalize_iris:         Daugman rubber-sheet polar unwrap to a fixed strip.
    visualize_segmentation: Overlay detected circles on the eye image.
"""

import cv2
import numpy as np


def load_image(image_path):
    """Load a grayscale image from `image_path`. Returns None on failure."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[load_image] Failed to read: {image_path}")
            return None
        return img
    except Exception as e:
        print(f"[load_image] Error reading {image_path}: {e}")
        return None


def preprocess_image(image):
    """Apply CLAHE histogram equalization to enhance iris texture contrast."""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    except Exception as e:
        print(f"[preprocess_image] CLAHE failed: {e}")
        return image


def _hough_circle(image, dp, min_dist, p1, p2, r_min, r_max):
    """Thin wrapper around cv2.HoughCircles returning the strongest circle."""
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=p1,
        param2=p2,
        minRadius=r_min,
        maxRadius=r_max,
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype(int)
    # Return the strongest (first) circle as (x, y, r)
    return tuple(circles[0])


def detect_iris_and_pupil(image):
    """
    Detect outer iris and inner pupil circles using HoughCircles.

    Returns:
        (iris_circle, pupil_circle) — each a tuple (x, y, radius), or None.

    If the pupil is not detected, an estimated pupil concentric with the iris
    is returned (radius ≈ 0.3 * iris_radius), so downstream normalization can
    still proceed.
    """
    if image is None:
        return None, None

    h, w = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # --- Iris (outer, larger circle) ------------------------------------
    iris = _hough_circle(
        blurred,
        dp=1.0,
        min_dist=max(h, w),
        p1=60,
        p2=40,
        r_min=int(min(h, w) * 0.20),
        r_max=int(min(h, w) * 0.48),
    )

    # --- Pupil (inner, smaller, very dark circle) -----------------------
    # Threshold to isolate dark pupil region for a more reliable Hough pass.
    _, dark = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.medianBlur(dark, 5)
    pupil = _hough_circle(
        dark,
        dp=1.0,
        min_dist=max(h, w),
        p1=50,
        p2=20,
        r_min=int(min(h, w) * 0.05),
        r_max=int(min(h, w) * 0.20),
    )

    # Fallback pupil if detection failed
    if iris is not None and pupil is None:
        ix, iy, ir = iris
        pupil = (ix, iy, max(int(ir * 0.30), 5))

    return iris, pupil


def normalize_iris(image, iris_circle, pupil_circle, radial_res=64, angular_res=512):
    """
    Daugman Rubber Sheet Model.

    Unwraps the annular iris region between the pupil and iris boundaries
    into a flat rectangle of shape (radial_res, angular_res).

    For each (r, theta):
        x = (1 - r) * x_pupil(theta) + r * x_iris(theta)
        y = (1 - r) * y_pupil(theta) + r * y_iris(theta)
    """
    if image is None or iris_circle is None or pupil_circle is None:
        return None

    ix, iy, ir = iris_circle
    px, py, pr = pupil_circle

    theta = np.linspace(0.0, 2.0 * np.pi, angular_res, endpoint=False)
    r = np.linspace(0.0, 1.0, radial_res)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Boundary points around pupil and iris circles
    x_pupil = px + pr * cos_t
    y_pupil = py + pr * sin_t
    x_iris = ix + ir * cos_t
    y_iris = iy + ir * sin_t

    # Build sampling grid (radial_res x angular_res)
    R = r[:, None]
    xs = (1.0 - R) * x_pupil[None, :] + R * x_iris[None, :]
    ys = (1.0 - R) * y_pupil[None, :] + R * y_iris[None, :]

    # Bilinear remap
    map_x = xs.astype(np.float32)
    map_y = ys.astype(np.float32)
    normalized = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return normalized


def visualize_segmentation(image, iris_circle, pupil_circle):
    """Return a BGR copy of `image` with detected iris/pupil circles drawn."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if iris_circle is not None:
        ix, iy, ir = iris_circle
        cv2.circle(vis, (ix, iy), ir, (0, 255, 0), 2)
        cv2.circle(vis, (ix, iy), 2, (0, 255, 0), 3)
    if pupil_circle is not None:
        px, py, pr = pupil_circle
        cv2.circle(vis, (px, py), pr, (0, 0, 255), 2)
        cv2.circle(vis, (px, py), 2, (0, 0, 255), 3)
    return vis
