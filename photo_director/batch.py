from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .analysis import analyze_raw, write_analysis
from .engine import apply_edit_plan
from .llm import request_edit_plan
from .schema import SUGGESTED_DISCARDED_BUCKET, bucket_for_worth_saving

RAW_EXTENSIONS = {
    '.3fr',
    '.arw',
    '.cr2',
    '.cr3',
    '.dng',
    '.nef',
    '.orf',
    '.raf',
    '.raw',
    '.rw2',
}


@dataclass(slots=True)
class EditJob:
    raw_path: Path
    output_dir: Path


@dataclass(slots=True)
class RoutedPaths:
    bucket: str
    preview_path: Path
    analysis_path: Path
    adjustments_path: Path
    output_path: Path
    masks_dir: Path


@dataclass(slots=True)
class EditResult:
    raw_path: Path
    output_path: Path | None
    ok: bool
    bucket: str | None = None
    worth_saving: float | None = None
    discard_reason: str = ''
    error: str | None = None


class JsonlWriter:
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path.expanduser().resolve()
        self._lock = threading.Lock()

    def append(self, record: dict[str, object]) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(',', ':'))
        with self._lock:
            with self.jsonl_path.open('a', encoding='utf-8') as handle:
                handle.write(line + '\n')


def discover_raw_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    pattern = '**/*' if recursive else '*'
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS
    )


def read_failed_list(failed_list: Path, input_dir: Path | None = None) -> list[Path]:
    base_dir = input_dir.expanduser().resolve() if input_dir else None
    paths = []
    for line in failed_list.expanduser().read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
            source = data.get('source', '')
        except json.JSONDecodeError:
            source = stripped.split('\t', 1)[0]
        path = Path(source).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        paths.append(path.resolve())
    return paths


def build_job(raw_path: Path, output_dir: Path) -> EditJob:
    return EditJob(
        raw_path=raw_path.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
    )


def work_preview_path(output_dir: Path, stem: str) -> Path:
    return output_dir / '.work' / f'{stem}_preview.jpg'


def resolve_routed_paths(output_dir: Path, bucket: str, stem: str) -> RoutedPaths:
    base = output_dir / bucket
    return RoutedPaths(
        bucket=bucket,
        preview_path=base / 'previews' / f'{stem}_preview.jpg',
        analysis_path=base / 'analysis' / f'{stem}_analysis.json',
        adjustments_path=base / 'adjustments' / f'{stem}_adjustments.json',
        output_path=base / 'edited' / f'{stem}_edited.jpg',
        masks_dir=base / 'masks',
    )


def _finalize_preview(work_preview: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if work_preview.resolve() != destination.resolve():
        if destination.exists():
            destination.unlink()
        shutil.move(str(work_preview), str(destination))


def process_job(
    job: EditJob,
    intent: str,
    model: str,
    preview_size: int,
    quality: int,
    timeout: int,
    max_retries: int,
    retry_backoff: float,
    worth_saving_threshold: float,
) -> EditResult:
    stem = job.raw_path.stem
    work_preview = work_preview_path(job.output_dir, stem)
    try:
        analysis = analyze_raw(job.raw_path, work_preview, preview_size)
        plan = request_edit_plan(
            analysis=analysis,
            preview_path=work_preview,
            intent=intent,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        bucket = bucket_for_worth_saving(plan.worth_saving, worth_saving_threshold)
        paths = resolve_routed_paths(job.output_dir, bucket, stem)
        _finalize_preview(work_preview, paths.preview_path)
        analysis['preview'] = str(paths.preview_path)
        write_analysis(analysis, paths.analysis_path)
        localized_application = apply_edit_plan(
            job.raw_path,
            paths.output_path,
            plan,
            quality=quality,
            mask_debug_dir=paths.masks_dir,
        )
        plan_dict = plan.to_dict(worth_saving_threshold=worth_saving_threshold)
        plan_dict['bucket'] = bucket
        plan_dict['localized_application'] = [record.to_dict() for record in localized_application]
        paths.adjustments_path.parent.mkdir(parents=True, exist_ok=True)
        paths.adjustments_path.write_text(json.dumps(plan_dict, indent=2), encoding='utf-8')
        return EditResult(
            raw_path=job.raw_path,
            output_path=paths.output_path,
            ok=True,
            bucket=bucket,
            worth_saving=plan.worth_saving,
            discard_reason=plan.discard_reason,
        )
    except Exception as exc:
        if work_preview.exists():
            work_preview.unlink(missing_ok=True)
        return EditResult(raw_path=job.raw_path, output_path=None, ok=False, error=str(exc))


def process_jobs(
    jobs: Iterable[EditJob],
    output_dir: Path,
    intent: str,
    model: str,
    preview_size: int,
    quality: int,
    timeout: int,
    max_retries: int,
    retry_backoff: float,
    worth_saving_threshold: float,
    workers: int,
) -> list[EditResult]:
    output_dir = output_dir.expanduser().resolve()
    failed_writer = JsonlWriter(output_dir / 'failed.jsonl')
    discarded_writer = JsonlWriter(output_dir / 'suggested_discarded.jsonl')
    results: list[EditResult] = []
    job_list = list(jobs)
    accepted_count = 0
    suggested_discarded_count = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                process_job,
                job,
                intent,
                model,
                preview_size,
                quality,
                timeout,
                max_retries,
                retry_backoff,
                worth_saving_threshold,
            ): job
            for job in job_list
        }
        with tqdm(total=len(futures), desc='Processing', unit='img') as progress:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.ok:
                    if result.bucket == SUGGESTED_DISCARDED_BUCKET:
                        suggested_discarded_count += 1
                        tqdm.write(
                            f'DISCARD {result.raw_path} -> {result.output_path} '
                            f'(worth_saving={result.worth_saving:.2f})'
                        )
                        discarded_writer.append(
                            {
                                'source': str(result.raw_path),
                                'output': str(result.output_path),
                                'worth_saving': result.worth_saving,
                                'worth_saving_threshold': worth_saving_threshold,
                                'discard_reason': result.discard_reason,
                                'discarded_at': datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    else:
                        accepted_count += 1
                        tqdm.write(f'OK {result.raw_path} -> {result.output_path}')
                else:
                    failed_count += 1
                    error = result.error or 'unknown error'
                    tqdm.write(f'FAILED {result.raw_path}: {error}')
                    failed_writer.append(
                        {
                            'source': str(result.raw_path),
                            'error': error,
                            'failed_at': datetime.now(timezone.utc).isoformat(),
                        }
                    )
                progress.set_postfix(
                    accepted=accepted_count,
                    discard=suggested_discarded_count,
                    failed=failed_count,
                    refresh=False,
                )
                progress.update(1)
    return results
