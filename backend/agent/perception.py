"""
agent/perception.py

Lightweight perception layer for GladdenAction's autonomous mode.

Responsibilities
----------------
- Capture a screenshot of the current screen state.
- Encode the screenshot as base64 (for optional LLM vision calls).
- Provide a simple describe_screen() helper that asks the LLM to
  summarise what is currently visible on screen.

Design principles
-----------------
- Keep it simple and modular — no heavy CV pipelines.
- Never crash the main execution loop; all functions degrade gracefully.
- Screenshot capture has three fallback strategies (same as UIAutomationTool).

Dependencies (all already in requirements.txt)
----------------------------------------------
    pyautogui     — primary screenshot capture
    Pillow        — ImageGrab fallback + encoding
"""

from __future__ import annotations

import base64
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────── #

def capture_screen(save_path: Optional[str] = None) -> Optional[bytes]:
    """
    Capture the full screen and return the raw PNG bytes.

    Also saves a file to *save_path* if provided (must end in .png).
    Returns None on failure (never raises).

    Strategy order
    --------------
    1. Pillow ImageGrab  (Windows-native, fastest)
    2. pyautogui.screenshot()
    3. Win + PrtSc shortcut → copy from ~/Pictures/Screenshots
    """
    png_bytes = _capture_strategy_pillow()

    if png_bytes is None:
        png_bytes = _capture_strategy_pyautogui()

    if png_bytes is None:
        png_bytes = _capture_strategy_winprtsc()

    if png_bytes is None:
        logger.warning("perception.capture_screen: all strategies failed.")
        return None

    if save_path:
        try:
            Path(save_path).write_bytes(png_bytes)
            logger.info("perception: screenshot saved → %s", save_path)
        except OSError as exc:
            logger.warning("perception: could not save screenshot: %s", exc)

    return png_bytes


def encode_image_base64(png_bytes: bytes) -> str:
    """
    Encode raw PNG bytes to a base64 string suitable for embedding in
    a JSON API request (e.g. Anthropic / OpenAI vision messages).
    """
    return base64.b64encode(png_bytes).decode("utf-8")


def describe_screen(
    groq_client,
    model_name: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Capture the current screen and ask the LLM to describe what is visible.

    Parameters
    ----------
    groq_client : groq.Groq
        An already-initialised Groq SDK client.
    model_name : str
        The model to use for the description call.

    Returns
    -------
    str
        A plain-English description of the current screen state,
        or a short error message if capture / LLM call fails.

    Notes
    -----
    - Groq does not currently support image inputs natively, so this
      function falls back to a text-only prompt that asks the model to
      reason about what *might* be on screen given the current time and
      OS context.  When a vision-capable model is available, swap in the
      image_url content block.
    - This is intentionally minimal — the autonomous loop uses it only
      to provide lightweight context, not precise pixel understanding.
    """
    # Attempt screenshot (best-effort; failures are non-fatal)
    png_bytes = capture_screen()
    has_image = png_bytes is not None

    if has_image:
        # Future: pass base64 image to vision model
        # b64 = encode_image_base64(png_bytes)
        # For now: acknowledge we captured it but use text reasoning
        logger.info("perception.describe_screen: screenshot captured (%d bytes)", len(png_bytes))

    ts = datetime.now().strftime("%H:%M:%S")
    prompt = (
        f"Current time: {ts}.\n"
        f"Screenshot captured: {'yes' if has_image else 'no (capture failed)'}.\n\n"
        "Based on the context of an ongoing desktop automation task, "
        "describe in 1-3 sentences what is likely visible on screen right now "
        "(e.g. which application is in the foreground, any dialog boxes, "
        "the state of the desktop). Be concise and factual."
    )

    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=128,
        )
        description = (response.choices[0].message.content or "").strip()
        logger.info("perception.describe_screen: %s", description[:80])
        return description
    except Exception as exc:  # noqa: BLE001
        logger.warning("perception.describe_screen LLM call failed: %s", exc)
        return f"(screen description unavailable: {exc})"


# ── Private capture strategies ────────────────────────────────────────────── #

def _capture_strategy_pillow() -> Optional[bytes]:
    """Strategy 1: Pillow ImageGrab — most reliable on Windows."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        logger.debug("perception: Pillow not installed, skipping strategy 1.")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("perception: Pillow ImageGrab failed: %s", exc)
        return None


def _capture_strategy_pyautogui() -> Optional[bytes]:
    """Strategy 2: pyautogui.screenshot()."""
    try:
        import pyautogui
        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.debug("perception: pyautogui screenshot failed: %s", exc)
        return None


def _capture_strategy_winprtsc() -> Optional[bytes]:
    """Strategy 3: Win+PrtSc shortcut → read from ~/Pictures/Screenshots."""
    try:
        import pyautogui
        pyautogui.hotkey("win", "printscreen")
        time.sleep(1.5)
        screenshots_dir = Path.home() / "Pictures" / "Screenshots"
        if not screenshots_dir.exists():
            return None
        candidates = sorted(
            screenshots_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        return candidates[0].read_bytes()
    except Exception as exc:  # noqa: BLE001
        logger.debug("perception: Win+PrtSc strategy failed: %s", exc)
        return None