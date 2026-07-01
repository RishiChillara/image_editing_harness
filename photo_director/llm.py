from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

from .schema import EDIT_PLAN_JSON_SCHEMA, EditPlan, GlobalAdjustments

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class PhotoDirectorLLMError(RuntimeError):
    """Raised when the LLM director cannot return a usable edit plan."""


def _load_dotenv_api_key(start: Path) -> str | None:
    for parent in (start.resolve(), *start.resolve().parents):
        env_path = parent / ".env"
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    return None


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _compact_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_settings": analysis.get("baseline_settings", {}),
        "baseline_strategy": analysis.get("baseline_strategy", ""),
        "raw_specs": analysis.get("raw_specs", {}),
        "exif": analysis.get("exif", {}),
        "histogram": analysis.get("histogram", {}),
        "dominant_palette": analysis.get("dominant_palette", []),
    }


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM response content was empty.")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])

    raise ValueError(f"LLM response did not contain a JSON object: {stripped[:240]}")


def _parse_edit_plan_response(data: dict[str, Any]) -> EditPlan:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response did not include choices.")

    message = choices[0].get("message") or {}
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return EditPlan.from_dict(parsed)

    content = _message_content_to_text(message.get("content"))
    return EditPlan.from_dict(_extract_json_object(content))


def _retry_delay(exc: Exception, attempt: int, backoff: float) -> float:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return backoff * (2 ** (attempt - 1))


def _is_retryable_request_error(exc: requests.RequestException) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in {408, 429, 500, 502, 503, 504}
    return True


def _request_error_message(exc: requests.RequestException) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        body = exc.response.text.strip()
        if body:
            return f"{exc}; response body: {body[:500]}"
    return str(exc)


def request_edit_plan(
    analysis: dict[str, Any],
    preview_path: Path,
    intent: str,
    model: str = "openrouter/free",
    api_key: str | None = None,
    timeout: int = 90,
    max_retries: int = 2,
    retry_backoff: float = 2.0,
) -> EditPlan:
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or _load_dotenv_api_key(preview_path)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to request an edit plan.")

    system_prompt = (
        "You are Photo Director, a professional RAW photo editing director. "
        "Interpret the user's aesthetic intent and the technical image analysis. "
        "Return conservative, production-ready Lightroom-style slider deltas from the provided baseline_settings. "
        "Do not return absolute final slider values. "
        "The response must be only the requested JSON object. "
        "Do not invent unsupported local masks; use localized_adjustments only when a target is obvious."
    )
    user_text = {
        "intent": intent,
        "slider_ranges": {
            "temperature": "-100 cool to +100 warm",
            "tint": "-100 green to +100 magenta",
            "exposure": "-3 to +3 EV",
            "contrast": "-100 to +100",
            "highlights": "-100 to +100",
            "shadows": "-100 to +100",
            "whites": "-100 to +100",
            "blacks": "-100 to +100",
            "saturation": "-100 to +100",
            "vibrance": "-100 to +100",
            "clarity": "-100 to +100",
        },
        "delta_instruction": (
            "baseline_settings are already applied by the renderer. "
            "Return global_delta values to add to that baseline. "
            "Small deltas are preferred; use exposure deltas roughly between -0.5 and +0.5 unless the preview is severely wrong."
        ),
        "analysis": _compact_analysis(analysis),
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(user_text, separators=(",", ":"))},
                    {"type": "image_url", "image_url": {"url": _image_data_url(preview_path)}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": EDIT_PLAN_JSON_SCHEMA,
        },
        "provider": {"require_parameters": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/photo-director",
        "X-Title": "Photo Director",
    }
    attempts = max_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            plan = _parse_edit_plan_response(data)
            if plan.baseline_settings.to_dict() == GlobalAdjustments().to_dict():
                plan.baseline_settings = GlobalAdjustments.from_dict(analysis.get("baseline_settings", {}))
            return plan
        except requests.RequestException as exc:
            last_error = exc
            retryable = _is_retryable_request_error(exc)
            if not retryable or attempt == attempts:
                break
            delay = _retry_delay(exc, attempt, retry_backoff)
            print(
                f"OpenRouter request failed on attempt {attempt}/{attempts}; "
                f"retrying in {delay:.1f}s: {_request_error_message(exc)}",
                file=sys.stderr,
            )
            time.sleep(delay)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = _retry_delay(exc, attempt, retry_backoff)
            print(f"OpenRouter response was not a valid edit plan on attempt {attempt}/{attempts}; retrying in {delay:.1f}s: {exc}", file=sys.stderr)
            time.sleep(delay)

    if isinstance(last_error, requests.RequestException):
        detail = _request_error_message(last_error)
    else:
        detail = str(last_error)
    raise PhotoDirectorLLMError(f"OpenRouter did not return a usable edit plan after {attempts} attempts: {detail}")
