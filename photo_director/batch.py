from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .analysis import analyze_raw, write_analysis
from .engine import apply_edit_plan
from .llm import request_edit_plan

RAW_EXTENSIONS = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".nef",
    ".orf",
    ".raf",
    ".raw",
    ".rw2",
}


@dataclass(slots=True)
class EditJob:
    raw_path: Path
    preview_path: Path
    analysis_path: Path
    adjustments_path: Path
    output_path: Path


@dataclass(slots=True)
class EditResult:
    raw_path: Path
    output_path: Path | None
    ok: bool
    error: str | None = None


class FailedListWriter:
    def __init__(self, failed_list: Path | None) -> None:
        self.failed_list = failed_list.expanduser().resolve() if failed_list else None
        self._lock = threading.Lock()

    def append(self, raw_path: Path, error: str) -> None:
        if self.failed_list is None:
            return
        self.failed_list.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "source": str(raw_path.expanduser().resolve()),
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            with self.failed_list.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def discover_raw_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS
    )


def read_failed_list(failed_list: Path, input_dir: Path | None = None) -> list[Path]:
    base_dir = input_dir.expanduser().resolve() if input_dir else None
    paths = []
    for line in failed_list.expanduser().read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
            source = data.get("source", "")
        except json.JSONDecodeError:
            source = stripped.split("\t", 1)[0]
        path = Path(source).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        paths.append(path.resolve())
    return paths


def build_job(raw_path: Path, output_dir: Path) -> EditJob:
    stem = raw_path.stem
    output_dir = output_dir.expanduser().resolve()
    return EditJob(
        raw_path=raw_path.expanduser().resolve(),
        preview_path=output_dir / "previews" / f"{stem}_preview.jpg",
        analysis_path=output_dir / "analysis" / f"{stem}_analysis.json",
        adjustments_path=output_dir / "adjustments" / f"{stem}_adjustments.json",
        output_path=output_dir / "edited" / f"{stem}_edited.jpg",
    )


def process_job(
    job: EditJob,
    intent: str,
    model: str,
    preview_size: int,
    quality: int,
    timeout: int,
    max_retries: int,
    retry_backoff: float,
) -> EditResult:
    try:
        analysis = analyze_raw(job.raw_path, job.preview_path, preview_size)
        write_analysis(analysis, job.analysis_path)
        plan = request_edit_plan(
            analysis=analysis,
            preview_path=job.preview_path,
            intent=intent,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        job.adjustments_path.parent.mkdir(parents=True, exist_ok=True)
        job.adjustments_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        apply_edit_plan(job.raw_path, job.output_path, plan, quality=quality)
        return EditResult(raw_path=job.raw_path, output_path=job.output_path, ok=True)
    except Exception as exc:
        return EditResult(raw_path=job.raw_path, output_path=None, ok=False, error=str(exc))


def process_jobs(
    jobs: Iterable[EditJob],
    intent: str,
    model: str,
    preview_size: int,
    quality: int,
    timeout: int,
    max_retries: int,
    retry_backoff: float,
    workers: int,
    failed_list: Path | None,
) -> list[EditResult]:
    writer = FailedListWriter(failed_list)
    results: list[EditResult] = []
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
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.ok:
                print(f"OK {result.raw_path} -> {result.output_path}")
            else:
                error = result.error or "unknown error"
                print(f"FAILED {result.raw_path}: {error}")
                writer.append(result.raw_path, error)
    return results
