"""Thin Gemini client: key loading, JSON-mode helper, and 429 backoff.

Why a wrapper: every LLM call in ProbeAI (moderator, participant, judge) goes
through here, so retry/backoff and JSON parsing live in exactly one place. The
*judge* deliberately constructs its own client with a different model/temperature
than the moderator (see evals/judge.py) — that isolation is the whole point of
the wrapper being cheap to instantiate.

Free-tier reality: Gemini throttles aggressively, returning HTTP 429
(RESOURCE_EXHAUSTED). We retry with exponential backoff + jitter.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # pull GEMINI_API_KEY (and model overrides) from .env if present

# Defaults; overridable per-call or via env. The moderator defaults to flash-lite
# (most calls -> needs the larger free quota); the judge intentionally runs on a
# DIFFERENT model — see README "Evaluation limitations".
DEFAULT_MODERATOR_MODEL = os.getenv("PROBEAI_MODERATOR_MODEL", "gemini-2.5-flash-lite")
DEFAULT_JUDGE_MODEL = os.getenv("PROBEAI_JUDGE_MODEL", "gemini-2.5-flash")

_MAX_RETRIES = 5
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 30.0


class LLMError(RuntimeError):
    """Raised when the model call fails after exhausting retries."""


# Retry on rate limits (429) AND transient server errors (5xx) — the Gemini free
# tier returns both: 429/RESOURCE_EXHAUSTED when throttled, 503 UNAVAILABLE when
# the model is briefly overloaded. Both are temporary and worth backing off on.
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_MARKERS = (
    "429",
    "resource_exhausted",
    "rate limit",
    "503",
    "unavailable",
    "overloaded",
    "high demand",
    "500",
    "internal error",
    "502",
    "504",
    "deadline",
)


def _is_retryable(exc: Exception) -> bool:
    """Best-effort detection of a transient error across SDK error shapes."""
    text = str(exc).lower()
    # A per-DAY free-tier quota won't recover within our backoff window — fail
    # fast instead of burning minutes. (Per-minute 429s stay retryable below.)
    if any(p in text for p in ("perday", "per day", "prepayment credits", "credits are depleted")):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in _RETRYABLE_CODES:
        return True
    return any(marker in text for marker in _RETRYABLE_MARKERS)


# Gemini 429s include the exact wait in a RetryInfo field; honor it when present
# rather than guessing with exponential backoff.
_RETRY_DELAY_RE = re.compile(r"retry(?:delay|\sin)['\":\s]+([0-9]+(?:\.[0-9]+)?)s", re.I)


def _server_retry_delay(exc: Exception) -> float | None:
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) if m else None


@dataclass
class LLMClient:
    """A configured Gemini client.

    Instantiate separately for moderator vs. judge so they never share context
    or config. ``temperature`` defaults low for deterministic-ish moderation.
    """

    model: str = DEFAULT_MODERATOR_MODEL
    temperature: float = 0.4
    api_key: str | None = None

    def __post_init__(self) -> None:
        # Imported lazily so the rest of the package (config, coverage, the
        # deterministic decision logic, tests) imports without the SDK installed.
        from google import genai

        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add a "
                "free key from https://aistudio.google.com/apikey"
            )
        self._genai = genai
        self._client = genai.Client(api_key=key)

    def complete(
        self,
        user: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        """Return the model's text for a single user message.

        Retries 429s with exponential backoff + jitter. ``json_mode`` asks Gemini
        to emit raw JSON (no markdown fences).
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature if temperature is None else temperature,
            response_mime_type="application/json" if json_mode else None,
        )

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=user, config=config
                )
                return (resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001 - normalize SDK error variety
                last_exc = exc
                if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                    break
                # Prefer the server's stated retry delay; fall back to exponential.
                server_delay = _server_retry_delay(exc)
                if server_delay is not None:
                    delay = min(server_delay + 0.5, _MAX_DELAY_SECONDS)
                else:
                    delay = min(_BASE_DELAY_SECONDS * (2**attempt), _MAX_DELAY_SECONDS)
                delay += random.uniform(0, delay * 0.25)  # jitter
                time.sleep(delay)

        raise LLMError(f"Gemini call failed: {last_exc}") from last_exc

    def complete_json(
        self,
        user: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Like ``complete`` but parses the response as JSON.

        Tolerates the occasional stray ```json fence the model adds despite
        json_mode, so callers always get a dict.
        """
        raw = self.complete(
            user, system=system, json_mode=True, temperature=temperature
        )
        return _parse_json(raw)


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse model output as JSON, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        # ```json\n...\n``` -> drop first and last fence lines
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) >= 2 else text
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model did not return valid JSON: {raw!r}") from exc
