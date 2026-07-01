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

By default, the LLM model is `openrouter/free`, OpenRouter's free model router. Override it with:

```bash
python -m photo_director edit input.ARW --model openrouter/free
```

The LLM call is retried with exponential backoff. By default, each image gets the initial OpenRouter request plus two retries:

```bash
python -m photo_director edit input.DNG --max-retries 2 --retry-backoff 2
```

If all LLM attempts fail, the image is not rendered with a local heuristic substitute. The command fails so you can retry it later.

## Batch Folders

Process a folder of RAW images concurrently:

```bash
python -m photo_director batch raw_folder \
  --output-dir outputs/batch_run \
  --workers 4 \
  --failed-list outputs/batch_run/failed.jsonl
```

Use `--recursive` to search nested folders.

Failed images are appended to a JSONL file. Retry only those images later with the same input folder for path resolution:

```bash
python -m photo_director batch raw_folder \
  --output-dir outputs/retry_run \
  --failed-list-input outputs/batch_run/failed.jsonl \
  --failed-list outputs/retry_run/failed.jsonl \
  --workers 2
```

## Architecture

1. `photo_director.analysis` reads the RAW file with RawPy, renders a color preview, extracts EXIF when available, computes histograms, clipping, and dominant colors.
2. `photo_director.llm` sends the preview plus metadata, baseline settings, and user intent to OpenRouter using structured JSON schema output. The model returns slider deltas, not absolute final settings.
3. `photo_director.engine` adds the model deltas to the baseline settings, maps the final settings into deterministic image operations, and exports the edited image.

The first version supports global edits. The schema already includes a `localized_adjustments` array for future OpenCV or segmentation-backed masks.
