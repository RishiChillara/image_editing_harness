from __future__ import annotations

from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageFilter

from .masks import LocalizedApplicationRecord, build_mask_for_target, save_mask_debug_image
from .schema import EditPlan, GlobalAdjustments


def _as_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def _tone_curve(image: np.ndarray, adjustments: GlobalAdjustments) -> np.ndarray:
    result = image * (2.0 ** adjustments.exposure)

    contrast_factor = 1.0 + adjustments.contrast / 160.0
    result = (result - 0.5) * contrast_factor + 0.5

    luminance = (
        0.2126 * result[:, :, 0]
        + 0.7152 * result[:, :, 1]
        + 0.0722 * result[:, :, 2]
    )
    shadow_mask = np.clip((0.55 - luminance) / 0.55, 0.0, 1.0)[:, :, None]
    highlight_mask = np.clip((luminance - 0.45) / 0.55, 0.0, 1.0)[:, :, None]

    result += shadow_mask * (adjustments.shadows / 100.0) * 0.28
    result += highlight_mask * (adjustments.highlights / 100.0) * 0.28
    result += (adjustments.whites / 100.0) * np.power(np.clip(result, 0.0, 1.0), 2.0) * 0.18
    result += (adjustments.blacks / 100.0) * np.power(1.0 - np.clip(result, 0.0, 1.0), 2.0) * 0.18
    return result


def _white_balance(image: np.ndarray, adjustments: GlobalAdjustments) -> np.ndarray:
    red_gain = 1.0 + adjustments.temperature / 250.0 + adjustments.tint / 600.0
    green_gain = 1.0 - adjustments.tint / 350.0
    blue_gain = 1.0 - adjustments.temperature / 250.0 + adjustments.tint / 600.0
    gains = np.asarray([red_gain, green_gain, blue_gain], dtype=np.float32)
    return image * gains[None, None, :]


def _color_presence(image: np.ndarray, adjustments: GlobalAdjustments) -> np.ndarray:
    gray = (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    )[:, :, None]
    saturation_factor = 1.0 + adjustments.saturation / 120.0
    result = gray + (image - gray) * saturation_factor

    chroma = np.mean(np.abs(result - gray), axis=2, keepdims=True)
    vibrance_weight = np.clip(1.0 - chroma * 2.4, 0.0, 1.0)
    result = gray + (result - gray) * (1.0 + vibrance_weight * adjustments.vibrance / 130.0)
    return result


def _clarity(image: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 0.01:
        return image
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    blur = pil_image.filter(ImageFilter.GaussianBlur(radius=8))
    base = np.asarray(pil_image).astype(np.float32) / 255.0
    blurred = np.asarray(blur).astype(np.float32) / 255.0
    return base + (base - blurred) * (amount / 120.0)


def _apply_adjustments(image: np.ndarray, adjustments: GlobalAdjustments) -> np.ndarray:
    result = _white_balance(image, adjustments)
    result = _tone_curve(result, adjustments)
    result = _color_presence(result, adjustments)
    return np.clip(_clarity(result, adjustments.clarity), 0.0, 1.0)


def _apply_localized_adjustments(
    base_image: np.ndarray,
    global_image: np.ndarray,
    plan: EditPlan,
    raw_stem: str,
    mask_debug_dir: Path | None,
) -> tuple[np.ndarray, list[LocalizedApplicationRecord]]:
    """Blend localized deltas into `global_image`, masked by heuristic target masks.

    Masks are derived from `base_image` (before global tone changes) since scene geometry
    cues like sky position are more reliable before exposure/contrast adjustments.
    """
    current = global_image
    records: list[LocalizedApplicationRecord] = []
    for local_adjustment in plan.localized_adjustments:
        target = local_adjustment.target
        try:
            # A failed local mask must never fail the whole image, so this catch is broad by design.
            mask_result = build_mask_for_target(target, base_image)
        except Exception as exc:
            records.append(
                LocalizedApplicationRecord(
                    target=target, status='skipped', reason=f'Mask generation failed: {exc}'
                )
            )
            continue

        if not mask_result.available:
            reason = mask_result.reason or 'No confident mask available for target.'
            records.append(
                LocalizedApplicationRecord(
                    target=target,
                    status='skipped',
                    mask_confidence=mask_result.confidence,
                    reason=reason,
                )
            )
            continue

        layer = _apply_adjustments(current, local_adjustment.delta)
        alpha = mask_result.mask[:, :, None]
        current = current * (1.0 - alpha) + layer * alpha

        mask_path_str = None
        if mask_debug_dir is not None:
            debug_dir = Path(mask_debug_dir).expanduser().resolve()
            mask_path = debug_dir / f'{raw_stem}_{target.strip().lower()}.png'
            save_mask_debug_image(mask_result.mask, mask_path)
            mask_path_str = str(mask_path)

        records.append(
            LocalizedApplicationRecord(
                target=target,
                status='applied',
                mask_confidence=mask_result.confidence,
                mask_path=mask_path_str,
            )
        )

    return current, records


def apply_edit_plan(
    raw_path: Path,
    output_path: Path,
    plan: EditPlan,
    quality: int = 95,
    mask_debug_dir: Path | None = None,
) -> list[LocalizedApplicationRecord]:
    raw_path = raw_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=8,
            no_auto_bright=True,
            gamma=(2.222, 4.5),
        )

    base_image = _as_float(rgb)
    global_image = _apply_adjustments(base_image, plan.final_settings())

    final_image = global_image
    localized_application: list[LocalizedApplicationRecord] = []
    if plan.localized_adjustments:
        final_image, localized_application = _apply_localized_adjustments(
            base_image, global_image, plan, raw_path.stem, mask_debug_dir
        )

    final_image = np.clip(final_image, 0.0, 1.0)
    output_bytes = (final_image * 255).astype(np.uint8)
    Image.fromarray(output_bytes).save(output_path, quality=quality, optimize=True)
    return localized_application
