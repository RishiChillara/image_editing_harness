from __future__ import annotations

import json
from pathlib import Path

import pytest

from photo_director.batch import (
    build_job,
    read_failed_list,
    resolve_routed_paths,
    work_preview_path,
)
from photo_director.schema import (
    ACCEPTED_BUCKET,
    DEFAULT_WORTH_SAVING_THRESHOLD,
    SUGGESTED_DISCARDED_BUCKET,
    EditPlan,
    GlobalAdjustments,
    bucket_for_worth_saving,
)


@pytest.mark.parametrize(
    ('worth_saving', 'threshold', 'expected_bucket'),
    [
        (0.25, 0.25, SUGGESTED_DISCARDED_BUCKET),
        (0.24, 0.25, SUGGESTED_DISCARDED_BUCKET),
        (0.26, 0.25, ACCEPTED_BUCKET),
        (1.0, DEFAULT_WORTH_SAVING_THRESHOLD, ACCEPTED_BUCKET),
    ],
)
def test_bucket_for_worth_saving(worth_saving: float, threshold: float, expected_bucket: str) -> None:
    assert bucket_for_worth_saving(worth_saving, threshold) == expected_bucket


def test_edit_plan_round_trip_includes_worth_saving_fields() -> None:
    plan = EditPlan.from_dict(
        {
            'global_delta': GlobalAdjustments(exposure=0.1).to_dict(),
            'localized_adjustments': [],
            'rationale': 'Recoverable exposure.',
            'confidence': 0.8,
            'worth_saving': 0.1,
            'discard_reason': 'Severely out of focus.',
        }
    )

    payload = plan.to_dict(worth_saving_threshold=0.25)

    assert payload['worth_saving'] == pytest.approx(0.1)
    assert payload['discard_reason'] == 'Severely out of focus.'
    assert payload['worth_saving_threshold'] == pytest.approx(0.25)


def test_resolve_routed_paths_uses_bucket_subdirectories(tmp_path: Path) -> None:
    paths = resolve_routed_paths(tmp_path, ACCEPTED_BUCKET, 'IMG_0001')

    assert paths.preview_path == tmp_path / 'accepted' / 'previews' / 'IMG_0001_preview.jpg'
    assert paths.output_path == tmp_path / 'accepted' / 'edited' / 'IMG_0001_edited.jpg'
    assert paths.masks_dir == tmp_path / 'accepted' / 'masks'


def test_build_job_and_work_preview_path(tmp_path: Path) -> None:
    raw_path = tmp_path / 'roll' / 'IMG_0001.dng'
    raw_path.parent.mkdir(parents=True)
    raw_path.touch()
    output_dir = tmp_path / 'run'

    job = build_job(raw_path, output_dir)

    assert job.raw_path == raw_path.resolve()
    assert job.output_dir == output_dir.resolve()
    assert work_preview_path(output_dir, 'IMG_0001') == output_dir / '.work' / 'IMG_0001_preview.jpg'


def test_read_failed_list_supports_jsonl_and_relative_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / 'raws'
    input_dir.mkdir()
    raw_path = input_dir / 'IMG_0001.dng'
    raw_path.touch()
    failed_list = tmp_path / 'failed.jsonl'
    failed_list.write_text(
        '\n'.join(
            [
                json.dumps({'source': str(raw_path), 'error': 'timeout'}),
                'relative.dng\terror',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    relative = input_dir / 'relative.dng'
    relative.touch()

    paths = read_failed_list(failed_list, input_dir=input_dir)

    assert paths == [raw_path.resolve(), relative.resolve()]
