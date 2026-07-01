from __future__ import annotations

import numpy as np

import photo_director.masks as masks_module
from photo_director.masks import (
    MASK_CONFIDENCE_THRESHOLD,
    MaskResult,
    build_mask_for_target,
    build_sky_mask,
    build_subject_mask,
    save_mask_debug_image,
)


def _synthetic_landscape(height: int = 240, width: int = 320) -> np.ndarray:
    """Build a synthetic image: flat bright/low-saturation sky on top, textured ground below."""
    image = np.zeros((height, width, 3), dtype=np.float32)
    horizon = int(height * 0.45)

    image[:horizon, :, :] = 0.82

    rng = np.random.default_rng(seed=42)
    ground_base = np.stack(
        [
            np.full((height - horizon, width), 0.20),
            np.full((height - horizon, width), 0.45),
            np.full((height - horizon, width), 0.15),
        ],
        axis=-1,
    )
    ground_noise = rng.uniform(-0.08, 0.08, size=ground_base.shape).astype(np.float32)
    image[horizon:, :, :] = np.clip(ground_base + ground_noise, 0.0, 1.0)

    return image


def test_build_sky_mask_isolates_top_region() -> None:
    image = _synthetic_landscape()
    result = build_sky_mask(image)

    assert result.mask is not None
    assert result.available
    assert result.confidence >= MASK_CONFIDENCE_THRESHOLD

    height = image.shape[0]
    top_mean = float(result.mask[: height // 3, :].mean())
    bottom_mean = float(result.mask[2 * height // 3 :, :].mean())
    assert top_mean > 0.8
    assert bottom_mean < 0.2


def test_build_sky_mask_rejects_tiny_image() -> None:
    tiny = np.zeros((4, 4, 3), dtype=np.float32)
    result = build_sky_mask(tiny)

    assert result.mask is None
    assert not result.available
    assert result.confidence == 0.0
    assert result.reason


def test_build_sky_mask_confidence_drops_for_ambiguous_scenes() -> None:
    clear_sky = _synthetic_landscape()

    # Dense, noisy texture across the whole frame (e.g. foliage or a crowd filling every row)
    # gives no clean low-texture band for the sky cues to lock onto.
    rng = np.random.default_rng(seed=7)
    noise = rng.uniform(-0.3, 0.3, size=clear_sky.shape)
    ambiguous = np.clip(0.35 + noise, 0.0, 1.0).astype(np.float32)

    clear_result = build_sky_mask(clear_sky)
    ambiguous_result = build_sky_mask(ambiguous)

    assert clear_result.confidence > ambiguous_result.confidence


def test_build_sky_mask_border_energy_finds_clean_horizon() -> None:
    image = _synthetic_landscape()
    result = masks_module._build_sky_mask_border_energy(image)

    assert result.mask is not None
    assert result.available


def test_build_sky_mask_border_energy_rejects_uniform_image() -> None:
    uniform = np.full((120, 160, 3), 0.3, dtype=np.float32)
    result = masks_module._build_sky_mask_border_energy(uniform)

    assert result.mask is None
    assert not result.available


def test_build_sky_mask_falls_back_to_heuristic_when_border_energy_not_confident(
    monkeypatch,
) -> None:
    def _fake_border_energy(image: np.ndarray) -> MaskResult:
        return MaskResult(target='sky', mask=None, confidence=0.0, reason='forced fallback')

    monkeypatch.setattr(masks_module, '_build_sky_mask_border_energy', _fake_border_energy)

    result = masks_module.build_sky_mask(_synthetic_landscape())

    assert result.available
    assert result.confidence >= MASK_CONFIDENCE_THRESHOLD


def test_build_subject_mask_hog_grabcut_finds_no_person_on_plain_landscape() -> None:
    image = _synthetic_landscape()
    result = masks_module._build_subject_mask_hog_grabcut(image)

    assert result.mask is None
    assert not result.available
    assert 'No person detected' in result.reason


def test_build_subject_mask_uses_hog_grabcut_result_when_confident(monkeypatch) -> None:
    image = _synthetic_landscape()
    height, width = image.shape[:2]
    fake_mask = np.ones((height, width), dtype=np.float32)

    def _fake_hog_grabcut(img: np.ndarray) -> MaskResult:
        return MaskResult(target='subject', mask=fake_mask, confidence=0.9)

    monkeypatch.setattr(masks_module, '_build_subject_mask_hog_grabcut', _fake_hog_grabcut)

    result = masks_module.build_subject_mask(image)

    assert result.available
    assert result.confidence == 0.9
    assert np.array_equal(result.mask, fake_mask)


def test_build_subject_mask_falls_back_to_gaussian_when_hog_not_confident(monkeypatch) -> None:
    def _fake_hog_grabcut(img: np.ndarray) -> MaskResult:
        return MaskResult(target='subject', mask=None, confidence=0.0, reason='no person')

    monkeypatch.setattr(masks_module, '_build_subject_mask_hog_grabcut', _fake_hog_grabcut)

    result = masks_module.build_subject_mask(_synthetic_landscape())

    assert not result.available
    assert result.confidence == 0.30


def test_build_subject_mask_is_conservative_and_below_threshold() -> None:
    image = _synthetic_landscape()
    result = build_subject_mask(image)

    assert result.mask is not None
    assert result.confidence < MASK_CONFIDENCE_THRESHOLD
    assert not result.available

    height, width = image.shape[:2]
    center_weight = result.mask[int(height * 0.62), width // 2]
    corner_weight = result.mask[0, 0]
    assert center_weight > corner_weight


def test_build_mask_for_target_dispatches_known_targets() -> None:
    image = _synthetic_landscape()
    sky_result = build_mask_for_target('Sky', image)
    subject_result = build_mask_for_target(' subject ', image)

    assert sky_result.target == 'Sky'
    assert sky_result.available
    assert subject_result.target == ' subject '
    assert subject_result.mask is not None


def test_build_mask_for_target_rejects_unsupported_target() -> None:
    image = _synthetic_landscape()
    result = build_mask_for_target('foreground_object', image)

    assert result.mask is None
    assert not result.available
    assert 'Unsupported' in result.reason


def test_save_mask_debug_image_writes_grayscale_png(tmp_path) -> None:
    mask = np.linspace(0.0, 1.0, num=100, dtype=np.float32).reshape(10, 10)
    output_path = tmp_path / 'debug' / 'mask.png'

    save_mask_debug_image(mask, output_path)

    assert output_path.exists()
    from PIL import Image

    saved = Image.open(output_path)
    assert saved.mode == 'L'
    assert saved.size == (10, 10)
