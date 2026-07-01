from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_path(raw_path: Path, suffix: str) -> Path:
    return raw_path.with_name(f"{raw_path.stem}{suffix}")


def edit_command(args: argparse.Namespace) -> int:
    from .analysis import analyze_raw, write_analysis
    from .engine import apply_edit_plan
    from .llm import PhotoDirectorLLMError, request_edit_plan

    raw_path = Path(args.raw_file)
    preview_path = Path(args.preview) if args.preview else _default_path(raw_path, "_preview.jpg")
    output_path = Path(args.output) if args.output else _default_path(raw_path, "_edited.jpg")

    analysis = analyze_raw(raw_path, preview_path, args.preview_size)
    if args.analysis:
        print(f"Writing analysis to {args.analysis}")
        write_analysis(analysis, Path(args.analysis))

    print(f"Requesting edit plan for {args.intent}")
    try:
        plan = request_edit_plan(
            analysis=analysis,
            preview_path=preview_path,
            intent=args.intent,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )
    except PhotoDirectorLLMError as exc:
        print(f"Failed to get edit plan: {exc}", file=sys.stderr)
        return 1
    print(f"Edit plan: {plan}")
    if args.adjustments:
        print(f"Writing adjustments to {args.adjustments}")
        adjustments_path = Path(args.adjustments).expanduser().resolve()
        adjustments_path.parent.mkdir(parents=True, exist_ok=True)
        adjustments_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    print(f"Applying edit plan to {raw_path} and writing to {output_path}")

    apply_edit_plan(raw_path, output_path, plan, quality=args.quality)
    print(f"Final output: {output_path}")
    print(json.dumps({"preview": str(preview_path), "output": str(output_path), "plan": plan.to_dict()}, indent=2))
    return 0


def batch_command(args: argparse.Namespace) -> int:
    from .batch import build_job, discover_raw_files, process_jobs, read_failed_list

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.failed_list_input:
        raw_files = read_failed_list(Path(args.failed_list_input), input_dir=input_dir)
        print(f"Retrying {len(raw_files)} images from {args.failed_list_input}")
    else:
        raw_files = discover_raw_files(input_dir, recursive=args.recursive)
        print(f"Discovered {len(raw_files)} RAW images in {input_dir}")

    jobs = [build_job(path, output_dir) for path in raw_files]
    results = process_jobs(
        jobs=jobs,
        intent=args.intent,
        model=args.model,
        preview_size=args.preview_size,
        quality=args.quality,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        workers=args.workers,
        failed_list=Path(args.failed_list) if args.failed_list else None,
    )
    failed = [result for result in results if not result.ok]
    print(json.dumps({"total": len(results), "succeeded": len(results) - len(failed), "failed": len(failed)}, indent=2))
    return 1 if failed else 0


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--intent", default="Auto-Edit", help="Aesthetic direction for the LLM.")
    parser.add_argument("--model", default="openrouter/free", help="OpenRouter model or router to use.")
    parser.add_argument("--preview-size", type=int, default=1400, help="Maximum preview width/height.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for final output.")
    parser.add_argument("--timeout", type=int, default=90, help="OpenRouter request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries after the first OpenRouter attempt.")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Initial retry delay in seconds.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo-director",
        description="LLM-directed RAW photo editing using RawPy and OpenRouter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    edit = subparsers.add_parser("edit", help="Analyze, direct, and render one RAW image.")
    edit.add_argument("raw_file", help="Path to a RAW file such as .CR3, .NEF, .ARW, .DNG.")
    edit.add_argument("--preview", help="Where to write the LLM preview JPEG.")
    edit.add_argument("--analysis", help="Optional path for analysis JSON.")
    edit.add_argument("--adjustments", help="Optional path for LLM adjustment JSON.")
    edit.add_argument("--output", help="Where to write the final edited JPEG.")
    add_common_llm_args(edit)
    edit.set_defaults(func=edit_command)

    batch = subparsers.add_parser("batch", help="Edit a folder of RAW images with concurrent workers.")
    batch.add_argument("input_dir", help="Folder containing RAW files. Also used to resolve retry-list relative paths.")
    batch.add_argument("--output-dir", required=True, help="Directory for previews, analysis, adjustments, and final edits.")
    batch.add_argument("--workers", type=int, default=2, help="Number of images to process concurrently.")
    batch.add_argument("--recursive", action="store_true", help="Search input_dir recursively.")
    batch.add_argument("--failed-list", help="JSONL file to append failed image records to.")
    batch.add_argument("--failed-list-input", help="Retry only images listed in a previous failed-list JSONL/text file.")
    add_common_llm_args(batch)
    batch.set_defaults(func=batch_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
