from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

MASK_CONFIDENCE_THRESHOLD = 0.45
MIN_MASK_DIMENSION = 8

SKY_BORDER_WORK_HEIGHT = 200
SKY_BORDER_ENERGY_SCALE = 1.0
SKY_BORDER_SMOOTH_WINDOW = 5
SKY_HEURISTIC_SEPARABILITY_SCALE = 0.15

SUBJECT_HOG_WORK_DIMENSION = 900
SUBJECT_HOG_HIT_THRESHOLD = -0.5
SUBJECT_DETECTOR_SIGMOID_CENTER = 0.5
SUBJECT_DETECTOR_SIGMOID_STEEPNESS = 1.5
SUBJECT_GRABCUT_ITERATIONS = 5


@dataclass(slots=True)
class MaskResult:
    """Result of attempting to build a feathered alpha mask for a localized target."""

    target: str
    mask: np.ndarray | None
    confidence: float
    reason: str = ''

    @property
    def available(self) -> bool:
        return self.mask is not None and self.confidence >= MASK_CONFIDENCE_THRESHOLD


@dataclass(slots=True)
class LocalizedApplicationRecord:
    """Metadata describing whether a localized adjustment was applied or skipped."""

    target: str
    status: str
    mask_confidence: float | None = None
    reason: str | None = None
    mask_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {'target': self.target, 'status': self.status}
        if self.mask_confidence is not None:
            data['mask_confidence'] = round(self.mask_confidence, 4)
        if self.reason is not None:
            data['reason'] = self.reason
        if self.mask_path is not None:
            data['mask_path'] = self.mask_path
        return data


def _feather(binary_or_soft_mask: np.ndarray, width: int, height: int) -> np.ndarray:
    feather_sigma = max(width, height) * 0.01 + 3.0
    source = binary_or_soft_mask.astype(np.float32)
    feathered = cv2.GaussianBlur(source, (0, 0), sigmaX=feather_sigma)
    return np.clip(feathered, 0.0, 1.0)


def _build_sky_mask_heuristic(image: np.ndarray) -> MaskResult:
    """Build a feathered sky mask using position, luminance, saturation, and texture cues.

    `image` must be an HxWx3 float32 array with values in [0, 1] (RGB).
    """
    height, width = image.shape[:2]
    if height < MIN_MASK_DIMENSION or width < MIN_MASK_DIMENSION:
        return MaskResult(
            target='sky', mask=None, confidence=0.0, reason='Image too small for sky mask.'
        )

    uint8_image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2GRAY)

    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    row_bias = np.clip(np.linspace(1.0, 0.0, height, dtype=np.float32) * 1.6, 0.0, 1.0) ** 1.2
    position_score = np.repeat(row_bias[:, None], width, axis=1)

    luminance_score = value
    saturation_score = 1.0 - saturation

    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    texture = cv2.GaussianBlur(texture, (9, 9), 0)
    texture_norm = texture / (float(texture.max()) + 1e-6)
    texture_score = 1.0 - np.clip(texture_norm * 3.0, 0.0, 1.0)

    combined = (
        0.40 * position_score
        + 0.25 * luminance_score
        + 0.15 * saturation_score
        + 0.20 * texture_score
    )
    combined_u8 = np.clip(combined * 255.0, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(combined_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    top_row_labels = set(np.unique(labels[0, :])) - {0}
    min_component_area = height * width * 0.01
    kept_area = 0
    for label in top_row_labels:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        keep[labels == label] = 255
        kept_area += area

    if kept_area == 0:
        return MaskResult(
            target='sky',
            mask=None,
            confidence=0.0,
            reason='No sky-like region touching the top edge.',
        )

    top_third = keep[: max(1, height // 3), :]
    top_third_ratio = float(np.count_nonzero(top_third)) / float(top_third.size)
    coverage_ratio = kept_area / float(height * width)

    mask_scores = combined[keep > 0]
    consistency = max(0.0, min(1.0, 1.0 - float(np.std(mask_scores))))

    # Position bias alone can make an arbitrary top/bottom split look "consistent" even when the
    # two regions have near-identical color (e.g. uniform noise), so gate on actual RGB
    # separability between the kept region and the rest of the image.
    sky_rgb = image[keep > 0]
    ground_rgb = image[keep == 0]
    color_distance = float(np.linalg.norm(sky_rgb.mean(axis=0) - ground_rgb.mean(axis=0)))
    separability = max(0.0, min(1.0, color_distance / SKY_HEURISTIC_SEPARABILITY_SCALE))

    confidence = max(
        0.0,
        min(1.0, 0.5 * top_third_ratio + 0.3 * consistency + 0.2 * min(coverage_ratio * 4.0, 1.0)),
    )
    confidence *= separability

    feathered = _feather(keep.astype(np.float32) / 255.0, width, height)
    return MaskResult(target='sky', mask=feathered, confidence=confidence)


def _smooth_border(border: np.ndarray, window: int = SKY_BORDER_SMOOTH_WINDOW) -> np.ndarray:
    """Median-filter a per-column border curve.

    This suppresses single-column spikes caused by thin objects (antennas, birds, wires) that
    poke into the sky, similar to the post-processing step described by Shen & Wang (2013).
    """
    if border.size < window:
        return border.astype(np.float32)
    pad = window // 2
    padded = np.pad(border.astype(np.float32), (pad, pad), mode='edge')
    return np.array(
        [np.median(padded[i : i + window]) for i in range(border.size)], dtype=np.float32
    )


def _build_sky_mask_border_energy(image: np.ndarray) -> MaskResult:
    """Gradient border-curve sky detector.

    Inspired by Shen & Wang (2013), "Sky Region Detection in a Single Image for Autonomous
    Ground Robot Navigation": for a swept set of gradient thresholds, build a per-column
    sky/ground border curve and pick the threshold whose border best separates the two regions
    by color. This uses a Fisher-style separability score rather than the paper's exact
    covariance-eigenvalue energy term, so treat it as inspired-by rather than a literal port.
    """
    height, width = image.shape[:2]
    if height < MIN_MASK_DIMENSION or width < MIN_MASK_DIMENSION:
        return MaskResult(
            target='sky', mask=None, confidence=0.0, reason='Image too small for sky mask.'
        )

    work_height = min(height, SKY_BORDER_WORK_HEIGHT)
    work_width = max(1, int(round(width * (work_height / height))))
    uint8_image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    small = cv2.resize(uint8_image, (work_width, work_height), interpolation=cv2.INTER_AREA)
    small_float = small.astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )

    rows_idx = np.arange(work_height)[:, None]
    total_pixels = work_width * work_height
    best_score = -np.inf
    best_border: np.ndarray | None = None
    for threshold in np.percentile(gradient, np.linspace(50.0, 97.0, 12)):
        exceeds = gradient > threshold
        has_edge = exceeds.any(axis=0)
        first_row = np.argmax(exceeds, axis=0)
        border = np.where(has_edge, first_row, work_height)

        sky_pixel_count = int(border.sum())
        if sky_pixel_count < total_pixels * 0.02 or sky_pixel_count > total_pixels * 0.85:
            continue

        sky_bool = rows_idx < border[None, :]
        sky_pixels = small_float[sky_bool]
        ground_pixels = small_float[~sky_bool]
        n_sky, n_ground = sky_pixels.shape[0], ground_pixels.shape[0]
        if n_sky == 0 or n_ground == 0:
            continue

        mean_diff = sky_pixels.mean(axis=0) - ground_pixels.mean(axis=0)
        between = float(np.sum(mean_diff ** 2)) * n_sky * n_ground / (n_sky + n_ground)
        within = (
            n_sky * float(sky_pixels.var(axis=0).sum())
            + n_ground * float(ground_pixels.var(axis=0).sum())
            + 1e-6
        )
        score = between / within
        if score > best_score:
            best_score, best_border = score, border

    if best_border is None:
        return MaskResult(
            target='sky', mask=None, confidence=0.0, reason='No valid sky/ground border found.'
        )

    smoothed_border = _smooth_border(best_border)
    full_border = np.interp(
        np.linspace(0, work_width - 1, width), np.arange(work_width), smoothed_border
    ) * (height / work_height)

    row_grid = np.arange(height)[:, None]
    binary_mask = (row_grid < full_border[None, :]).astype(np.float32)
    feathered = _feather(binary_mask, width, height)

    confidence = max(0.0, min(1.0, best_score / (best_score + SKY_BORDER_ENERGY_SCALE)))
    return MaskResult(target='sky', mask=feathered, confidence=confidence)


def build_sky_mask(image: np.ndarray) -> MaskResult:
    """Build a feathered sky mask.

    Tries the gradient border-curve detector (`_build_sky_mask_border_energy`) first, since it
    adapts to the actual sky/ground color separation instead of fixed cues. If that isn't
    confident, falls back to the position/luminance/saturation/texture heuristic
    (`_build_sky_mask_heuristic`).
    """
    try:
        border_result = _build_sky_mask_border_energy(image)
    except Exception as exc:  # A failed detector must never fail the whole image.
        border_result = MaskResult(
            target='sky', mask=None, confidence=0.0, reason=f'Border-energy detection failed: {exc}'
        )

    if border_result.available:
        return border_result

    heuristic_result = _build_sky_mask_heuristic(image)
    if heuristic_result.available:
        return heuristic_result

    if border_result.confidence >= heuristic_result.confidence:
        return border_result
    return heuristic_result


def _build_subject_mask_gaussian(image: np.ndarray) -> MaskResult:
    """Build a conservative center/lower-third foreground heuristic mask.

    This is intentionally weak (low confidence) so it is used only when no confident
    detector-based mask (see `_build_subject_mask_hog_grabcut`) is available.
    """
    height, width = image.shape[:2]
    if height < MIN_MASK_DIMENSION or width < MIN_MASK_DIMENSION:
        return MaskResult(
            target='subject', mask=None, confidence=0.0, reason='Image too small for subject mask.'
        )

    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x = width / 2.0
    center_y = height * 0.62
    sigma_x = width * 0.32
    sigma_y = height * 0.38
    x_term = ((x_coords - center_x) ** 2) / (2.0 * sigma_x ** 2)
    y_term = ((y_coords - center_y) ** 2) / (2.0 * sigma_y ** 2)
    gaussian = np.exp(-(x_term + y_term))
    mask = np.clip(gaussian, 0.0, 1.0).astype(np.float32)

    confidence = 0.30
    return MaskResult(target='subject', mask=mask, confidence=confidence)


def _build_subject_mask_hog_grabcut(image: np.ndarray) -> MaskResult:
    """Detect a person with a HOG + linear SVM pedestrian detector, then refine the detected
    box into a soft foreground mask with GrabCut.

    Detection and GrabCut both run on a downscaled copy of the image for speed (GrabCut is far
    too slow to run directly on full RAW resolutions), and the resulting soft mask is resized
    back up and feathered.
    """
    height, width = image.shape[:2]
    if height < 32 or width < 32:
        return MaskResult(
            target='subject', mask=None, confidence=0.0, reason='Image too small for HOG detection.'
        )

    work_scale = SUBJECT_HOG_WORK_DIMENSION / max(height, width)
    work_width = max(1, int(round(width * work_scale)))
    work_height = max(1, int(round(height * work_scale)))
    uint8_image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    small_bgr = cv2.cvtColor(
        cv2.resize(uint8_image, (work_width, work_height), interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2BGR,
    )

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        small_bgr,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.03,
        hitThreshold=SUBJECT_HOG_HIT_THRESHOLD,
    )
    if len(rects) == 0:
        return MaskResult(
            target='subject', mask=None, confidence=0.0, reason='No person detected via HOG.'
        )

    weight_values = weights.ravel()
    best_index = int(np.argmax(weight_values))
    best_weight = float(weight_values[best_index])
    box_x, box_y, box_w, box_h = (int(v) for v in rects[best_index])

    pad_x, pad_y = int(box_w * 0.1), int(box_h * 0.1)
    x0 = max(0, box_x - pad_x)
    y0 = max(0, box_y - pad_y)
    x1 = min(work_width, box_x + box_w + pad_x)
    y1 = min(work_height, box_y + box_h + pad_y)
    if x1 <= x0 or y1 <= y0:
        return MaskResult(
            target='subject', mask=None, confidence=0.0, reason='Degenerate HOG detection box.'
        )

    grabcut_mask = np.zeros((work_height, work_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    rect = (x0, y0, x1 - x0, y1 - y0)
    try:
        cv2.grabCut(
            small_bgr, grabcut_mask, rect, background_model, foreground_model,
            SUBJECT_GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as exc:
        return MaskResult(
            target='subject', mask=None, confidence=0.0, reason=f'GrabCut failed: {exc}'
        )

    foreground = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
    coverage = float(foreground.mean())
    if coverage < 0.01 or coverage > 0.85:
        return MaskResult(
            target='subject',
            mask=None,
            confidence=0.0,
            reason='GrabCut mask coverage out of range.',
        )

    full_foreground = cv2.resize(foreground, (width, height), interpolation=cv2.INTER_LINEAR)
    feathered = _feather(full_foreground, width, height)

    centered_weight = best_weight - SUBJECT_DETECTOR_SIGMOID_CENTER
    sigmoid_input = centered_weight * SUBJECT_DETECTOR_SIGMOID_STEEPNESS
    confidence = max(0.0, min(1.0, float(1.0 / (1.0 + np.exp(-sigmoid_input)))))

    return MaskResult(target='subject', mask=feathered, confidence=confidence)


def build_subject_mask(image: np.ndarray) -> MaskResult:
    """Build a subject mask.

    Tries a HOG pedestrian detector refined with GrabCut first
    (`_build_subject_mask_hog_grabcut`). If no confident person is found, falls back to the
    conservative center/lower-third Gaussian heuristic (`_build_subject_mask_gaussian`).
    """
    try:
        hog_result = _build_subject_mask_hog_grabcut(image)
    except Exception as exc:  # A failed detector must never fail the whole image.
        hog_result = MaskResult(
            target='subject',
            mask=None,
            confidence=0.0,
            reason=f'HOG/GrabCut detection failed: {exc}',
        )

    if hog_result.available:
        return hog_result

    gaussian_result = _build_subject_mask_gaussian(image)
    if gaussian_result.available:
        return gaussian_result

    return hog_result if hog_result.confidence >= gaussian_result.confidence else gaussian_result


_TARGET_BUILDERS: dict[str, Callable[[np.ndarray], MaskResult]] = {
    'sky': build_sky_mask,
    'subject': build_subject_mask,
}


def build_mask_for_target(target: str, image: np.ndarray) -> MaskResult:
    """Dispatch to the builder for `target`, returning an unavailable result if unsupported."""
    normalized = target.strip().lower()
    builder = _TARGET_BUILDERS.get(normalized)
    if builder is None:
        return MaskResult(
            target=target,
            mask=None,
            confidence=0.0,
            reason=f'Unsupported localized target: {target!r}.',
        )
    result = builder(image)
    result.target = target
    return result


def save_mask_debug_image(mask: np.ndarray, path: Path) -> None:
    """Save a feathered alpha mask (float32 [0, 1]) as an 8-bit grayscale PNG for debugging."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    grayscale = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(grayscale, mode='L').save(path)
