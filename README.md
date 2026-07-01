# Photo Director

Photo Director is an MVP for semantic RAW photo editing. It analyzes a RAW file deterministically, asks a vision LLM on OpenRouter for structured slider adjustments, and applies those adjustments back to a high-bit-depth RAW render.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."
```

## Run

```bash
python -m photo_director edit input.CR3 --intent "Auto-Edit" --output final.jpg
```

Useful options:

```bash
python -m photo_director edit input.NEF \
  --intent "Make it look like warm 70s film" \
  --preview outputs/preview.jpg \
  --adjustments outputs/adjustments.json \
  --analysis outputs/analysis.json \
  --output outputs/final.jpg
```

By default, the LLM model is `google/gemma-4-26b-a4b-it:free`, a free vision-capable OpenRouter
model that supports structured JSON output. Override it with:

```bash
python -m photo_director edit input.ARW --model google/gemma-4-26b-a4b-it:free
```

The LLM call is retried with exponential backoff. By default, each image gets the initial OpenRouter request plus two retries:

```bash
python -m photo_director edit input.DNG --max-retries 2 --retry-backoff 2
```

If all LLM attempts fail, the image is not rendered with a local heuristic substitute. The command fails so you can retry it later.

## Batch Folders

Process a folder of RAW images concurrently. Each image is edited fully, then routed by
`worth_saving` into either `accepted/` (keepers) or `suggested_discarded/` (model thinks you
can drop it):

```bash
python -m photo_director batch raw_folder \
  --output-dir outputs/batch_run \
  --workers 4 \
  --worth-saving-threshold 0.25
```

Use `--recursive` to search nested folders.

Output layout:

```text
outputs/batch_run/
  accepted/
    previews/
    analysis/
    adjustments/
    edited/
    masks/
  suggested_discarded/
    previews/
    analysis/
    adjustments/
    edited/
    masks/
  failed.jsonl
  suggested_discarded.jsonl
```

Failed images are appended to `<output-dir>/failed.jsonl`. Retry only those images later with
the same input folder for path resolution:

```bash
python -m photo_director batch raw_folder \
  --output-dir outputs/batch_run \
  --retry-failed \
  --workers 2
```

The LLM returns `worth_saving` (0–1) and `discard_reason` in the same edit-plan response.
Images with `worth_saving <= --worth-saving-threshold` route to `suggested_discarded/`; the
rest go to `accepted/`. Edits are always applied in both buckets.

## Localized Editing

When the LLM returns `localized_adjustments` (currently `sky` and `subject`), the engine
builds a heuristic feathered mask for each target with OpenCV and blends the target's
delta sliders into the globally-adjusted image inside that mask. No extra CLI step is
required; this happens automatically for any plan that includes localized adjustments.

- `sky` first tries a gradient border-curve detector inspired by Shen & Wang (2013), "Sky Region
  Detection in a Single Image for Autonomous Ground Robot Navigation": it sweeps a set of gradient
  thresholds, builds a per-column sky/ground border for each, and keeps the border whose two sides
  separate best by color. If that isn't confident (e.g. a very noisy or textureless frame), it falls
  back to a position/luminance/saturation/texture heuristic.
- `subject` first tries a classic HOG + linear-SVM pedestrian detector refined into a soft mask with
  GrabCut (both run on a downscaled copy for speed). If no confident person is detected, it falls
  back to a conservative center/lower-third Gaussian heuristic that is intentionally kept below the
  confidence threshold, so weak/uncertain subject masks are skipped rather than applied.
- If a mask cannot be built with enough confidence (or mask generation raises an error), that
  localized adjustment is skipped; the rest of the image (global edits, and any other localized
  adjustment) is still applied. A failed local mask never fails the whole image.

Save the feathered masks for debugging with `--mask-debug-dir`:

```bash
python -m photo_director edit input.DNG --output outputs/final.jpg --mask-debug-dir outputs/masks
```

Batch runs always save mask debug PNGs under `<output-dir>/<bucket>/masks/<stem>_<target>.png`.

The adjustment JSON (written via `--adjustments`, or always in batch mode) includes a
`localized_application` array recording, per target, whether it was `applied` or `skipped`,
the mask confidence, and (when applied and debugging is enabled) the mask file path:

```json
"localized_application": [
  {"target": "sky", "status": "applied", "mask_confidence": 0.78, "mask_path": "outputs/masks/image_sky.png"},
  {"target": "subject", "status": "skipped", "mask_confidence": 0.3, "reason": "No confident mask available for target."}
]
```

TODO: Replace CV based approaches with https://github.com/facebookresearch/sam2

## Architecture

1. `photo_director.analysis` reads the RAW file with RawPy, renders a color preview, extracts EXIF when available, computes histograms, clipping, and dominant colors.
2. `photo_director.llm` sends the preview plus metadata, baseline settings, and user intent to OpenRouter using structured JSON schema output. The model returns slider deltas, not absolute final settings.
3. `photo_director.engine` adds the model deltas to the baseline settings, maps the final settings into deterministic image operations, exports the edited image, and blends in any confident localized adjustments from `photo_director.masks`.
4. `photo_director.masks` builds OpenCV-based feathered masks for supported localized targets (`sky`, `subject`), each with a primary detector and a simpler heuristic fallback, and reports a confidence score used to decide whether to apply or skip each one.

## Tests

```bash
pip install -r requirements.txt
pytest
```

The engine integration tests use the sample RAW file linked at `input_raws/DSC02765.dng` and are
skipped automatically if that fixture is not present in your environment.
