from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_path(raw_path: Path, suffix: str) -> Path:
    return raw_path.with_name(f'{raw_path.stem}{suffix}')


def edit_command(args: argparse.Namespace) -> int:
    from .analysis import analyze_raw, write_analysis
    from .batch import (
        _finalize_preview,
        resolve_routed_paths,
        work_preview_path,
    )
    from .engine import apply_edit_plan
    from .llm import PhotoDirectorLLMError, request_edit_plan
    from .schema import bucket_for_worth_saving

    raw_path = Path(args.raw_file)
    worth_saving_threshold = args.worth_saving_threshold
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if output_dir is not None:
        stem = raw_path.stem
        work_preview = work_preview_path(output_dir, stem)
        analysis = analyze_raw(raw_path, work_preview, args.preview_size)
        if args.analysis:
            print(f'Writing analysis to {args.analysis}')
            write_analysis(analysis, Path(args.analysis))
    else:
        preview_path = Path(args.preview) if args.preview else _default_path(raw_path, '_preview.jpg')
        analysis = analyze_raw(raw_path, preview_path, args.preview_size)
        if args.analysis:
            print(f'Writing analysis to {args.analysis}')
            write_analysis(analysis, Path(args.analysis))
        work_preview = preview_path

    print(f'Requesting edit plan for {args.intent}')
    try:
        plan = request_edit_plan(
            analysis=analysis,
            preview_path=work_preview,
            intent=args.intent,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )
    except PhotoDirectorLLMError as exc:
        print(f'Failed to get edit plan: {exc}', file=sys.stderr)
        return 1

    bucket = bucket_for_worth_saving(plan.worth_saving, worth_saving_threshold)
    if output_dir is not None:
        paths = resolve_routed_paths(output_dir, bucket, raw_path.stem)
        if work_preview.resolve() != paths.preview_path.resolve():
            _finalize_preview(work_preview, paths.preview_path)
        analysis['preview'] = str(paths.preview_path)
        write_analysis(analysis, paths.analysis_path)
        output_path = paths.output_path
        mask_debug_dir = paths.masks_dir
        adjustments_path = paths.adjustments_path
        preview_for_summary = str(paths.preview_path)
    else:
        output_path = Path(args.output) if args.output else _default_path(raw_path, '_edited.jpg')
        mask_debug_dir = Path(args.mask_debug_dir) if args.mask_debug_dir else None
        adjustments_path = Path(args.adjustments).expanduser().resolve() if args.adjustments else None
        preview_for_summary = str(work_preview)

    print(
        f'Edit plan (worth_saving={plan.worth_saving:.2f}, bucket={bucket}): {plan}'
    )
    print(f'Applying edit plan to {raw_path} and writing to {output_path}')
    localized_application = apply_edit_plan(
        raw_path, output_path, plan, quality=args.quality, mask_debug_dir=mask_debug_dir
    )

    plan_dict = plan.to_dict(worth_saving_threshold=worth_saving_threshold)
    plan_dict['bucket'] = bucket
    plan_dict['localized_application'] = [record.to_dict() for record in localized_application]

    if adjustments_path is not None:
        print(f'Writing adjustments to {adjustments_path}')
        adjustments_path.parent.mkdir(parents=True, exist_ok=True)
        adjustments_path.write_text(json.dumps(plan_dict, indent=2), encoding='utf-8')
    elif args.adjustments:
        adjustments_path = Path(args.adjustments).expanduser().resolve()
        print(f'Writing adjustments to {adjustments_path}')
        adjustments_path.parent.mkdir(parents=True, exist_ok=True)
        adjustments_path.write_text(json.dumps(plan_dict, indent=2), encoding='utf-8')

    print(f'Final output: {output_path}')
    print(
        json.dumps(
            {
                'preview': preview_for_summary,
                'output': str(output_path),
                'bucket': bucket,
                'worth_saving': plan.worth_saving,
                'worth_saving_threshold': worth_saving_threshold,
                'plan': plan_dict,
            },
            indent=2,
        )
    )
    return 0


def batch_command(args: argparse.Namespace) -> int:
    from .batch import build_job, discover_raw_files, process_jobs, read_failed_list
    from .schema import ACCEPTED_BUCKET, SUGGESTED_DISCARDED_BUCKET

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    failed_list = output_dir / 'failed.jsonl'
    if args.retry_failed:
        if not failed_list.exists():
            print(f'No failed list found at {failed_list}', file=sys.stderr)
            return 1
        raw_files = read_failed_list(failed_list, input_dir=input_dir)
        print(f'Retrying {len(raw_files)} images from {failed_list}')
    else:
        raw_files = discover_raw_files(input_dir, recursive=args.recursive)
        print(f'Discovered {len(raw_files)} RAW images in {input_dir}')

    jobs = [build_job(path, output_dir) for path in raw_files]
    results = process_jobs(
        jobs=jobs,
        output_dir=output_dir,
        intent=args.intent,
        model=args.model,
        preview_size=args.preview_size,
        quality=args.quality,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        worth_saving_threshold=args.worth_saving_threshold,
        workers=args.workers,
    )
    failed = [result for result in results if not result.ok]
    accepted = [result for result in results if result.ok and result.bucket == ACCEPTED_BUCKET]
    suggested_discarded = [
        result for result in results if result.ok and result.bucket == SUGGESTED_DISCARDED_BUCKET
    ]
    print(
        json.dumps(
            {
                'total': len(results),
                'accepted': len(accepted),
                'suggested_discarded': len(suggested_discarded),
                'failed': len(failed),
            },
            indent=2,
        )
    )
    return 1 if failed else 0


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    from .schema import DEFAULT_WORTH_SAVING_THRESHOLD

    parser.add_argument('--intent', default='Auto-Edit', help='Aesthetic direction for the LLM.')
    parser.add_argument(
        '--model',
        default='google/gemma-4-26b-a4b-it:free',
        help='OpenRouter model or router to use.',
    )
    parser.add_argument('--preview-size', type=int, default=1400, help='Maximum preview width/height.')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality for final output.')
    parser.add_argument('--timeout', type=int, default=90, help='OpenRouter request timeout in seconds.')
    parser.add_argument('--max-retries', type=int, default=2, help='Retries after the first OpenRouter attempt.')
    parser.add_argument('--retry-backoff', type=float, default=2.0, help='Initial retry delay in seconds.')
    parser.add_argument(
        '--worth-saving-threshold',
        type=float,
        default=DEFAULT_WORTH_SAVING_THRESHOLD,
        help='Images with worth_saving at or below this route to suggested_discarded/.',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='photo-director',
        description='LLM-directed RAW photo editing using RawPy and OpenRouter.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    edit = subparsers.add_parser('edit', help='Analyze, direct, and render one RAW image.')
    edit.add_argument('raw_file', help='Path to a RAW file such as .CR3, .NEF, .ARW, .DNG.')
    edit.add_argument(
        '--output-dir',
        help='Route outputs under accepted/ or suggested_discarded/ inside this directory.',
    )
    edit.add_argument('--preview', help='Where to write the LLM preview JPEG.')
    edit.add_argument('--analysis', help='Optional path for analysis JSON.')
    edit.add_argument('--adjustments', help='Optional path for LLM adjustment JSON.')
    edit.add_argument('--output', help='Where to write the final edited JPEG.')
    edit.add_argument(
        '--mask-debug-dir', help='Optional directory to save localized mask debug PNGs.'
    )
    add_common_llm_args(edit)
    edit.set_defaults(func=edit_command)

    batch = subparsers.add_parser('batch', help='Edit a folder of RAW images with concurrent workers.')
    batch.add_argument(
        'input_dir',
        help='Folder containing RAW files. Also used to resolve retry-list relative paths.',
    )
    batch.add_argument(
        '--output-dir',
        required=True,
        help='Directory for accepted/, suggested_discarded/, and failed.jsonl.',
    )
    batch.add_argument('--workers', type=int, default=2, help='Number of images to process concurrently.')
    batch.add_argument('--recursive', action='store_true', help='Search input_dir recursively.')
    batch.add_argument(
        '--retry-failed',
        action='store_true',
        help='Retry only images listed in <output-dir>/failed.jsonl.',
    )
    add_common_llm_args(batch)
    batch.set_defaults(func=batch_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
