from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photo_director.engine import apply_edit_plan
from photo_director.schema import EditPlan, GlobalAdjustments, LocalizedAdjustment

SAMPLE_DNG = Path(__file__).resolve().parent.parent / 'input_raws' / 'DSC02765.dng'
_SAMPLE_DNG_AVAILABLE = SAMPLE_DNG.exists()
_SKIP_REASON = 'Sample DNG fixture is not available in this environment.'
requires_sample_dng = pytest.mark.skipif(not _SAMPLE_DNG_AVAILABLE, reason=_SKIP_REASON)


def _sample_plan() -> EditPlan:
    sky_delta = GlobalAdjustments(exposure=-0.6, saturation=15.0)
    return EditPlan(
        baseline_settings=GlobalAdjustments(),
        global_delta=GlobalAdjustments(exposure=0.2, contrast=5.0),
        localized_adjustments=[
            LocalizedAdjustment(target='sky', delta=sky_delta),
            LocalizedAdjustment(target='subject', delta=GlobalAdjustments(exposure=0.4)),
        ],
    )


@requires_sample_dng
def test_apply_edit_plan_creates_sky_mask_and_nonblank_output(tmp_path) -> None:
    plan = _sample_plan()
    output_path = tmp_path / 'edited.jpg'
    mask_debug_dir = tmp_path / 'masks'

    records = apply_edit_plan(
        SAMPLE_DNG, output_path, plan, quality=90, mask_debug_dir=mask_debug_dir
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    rendered = np.asarray(Image.open(output_path))
    assert rendered.std() > 1.0

    statuses = {record.target: record for record in records}
    assert statuses['sky'].status == 'applied'
    assert statuses['sky'].mask_confidence is not None
    assert statuses['sky'].mask_confidence >= 0.45

    # `apply_edit_plan` resolves symlinks, so the debug filename is derived from the
    # resolved raw path's stem rather than the (possibly symlinked) fixture path's stem.
    sky_mask_path = mask_debug_dir / f'{SAMPLE_DNG.resolve().stem}_sky.png'
    assert sky_mask_path.exists()

    assert statuses['subject'].status == 'skipped'
    assert statuses['subject'].reason


@requires_sample_dng
def test_apply_edit_plan_without_localized_adjustments_matches_global_only(tmp_path) -> None:
    global_only_plan = EditPlan(
        baseline_settings=GlobalAdjustments(),
        global_delta=GlobalAdjustments(exposure=0.2, contrast=5.0),
    )
    output_path = tmp_path / 'global_only.jpg'

    records = apply_edit_plan(SAMPLE_DNG, output_path, global_only_plan, quality=90)

    assert records == []
    assert output_path.exists()
    rendered = np.asarray(Image.open(output_path))
    assert rendered.std() > 1.0


@requires_sample_dng
def test_apply_edit_plan_skips_local_adjustment_without_mask_debug_dir(tmp_path) -> None:
    plan = _sample_plan()
    output_path = tmp_path / 'edited_no_debug.jpg'

    records = apply_edit_plan(SAMPLE_DNG, output_path, plan, quality=90, mask_debug_dir=None)

    assert output_path.exists()
    statuses = {record.target: record for record in records}
    assert statuses['sky'].mask_path is None
