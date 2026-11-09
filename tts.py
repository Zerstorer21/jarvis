"""
tts.py — Fish Audio synthesis, one request per sentence chunk.

The response is streamed so time-to-first-byte can be measured; the chunk is
returned whole because the browser decodes one complete MP3 per chunk.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger("jarvis.tts")

FISH_TTS_URL = "https://api.fish.audio/v1/tts"


@dataclass
class SynthResult:
    audio: bytes
    first_byte_sec: float
    total_sec: float


# ── Piper: voce locale, nessun servizio esterno ─────────────────────────────
# JARVIS_TTS=piper la attiva. JARVIS_PIPER_VOICE è il percorso del file .onnx
# (accanto deve esserci il suo .onnx.json); predefinito data/voices/it_IT-paola-medium.onnx.
_piper_voice = None
_piper_lock = threading.Lock()


def piper_enabled() -> bool:
    return os.getenv("JARVIS_TTS", "").lower() == "piper"


def _piper_path() -> Path:
    raw = os.getenv("JARVIS_PIPER_VOICE", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent / "data" / "voices" / "it_IT-paola-medium.onnx"


def piper_wav(text: str) -> bytes:
    """Testo -> WAV (bytes). Bloccante: chiamare in un thread."""
    global _piper_voice
    with _piper_lock:
        if _piper_voice is None:
            from piper import PiperVoice
            _piper_voice = PiperVoice.load(str(_piper_path()))
            log.info(f"piper: loaded {_piper_path().name}")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            _piper_voice.synthesize_wav(text, wav)
        return buf.getvalue()


async def _synthesize_piper(text: str) -> Optional[SynthResult]:
    t0 = time.monotonic()
    try:
        audio = await asyncio.to_thread(piper_wav, text)
    except Exception as e:
        log.error(f"piper TTS error: {e}")
        return None
    took = time.monotonic() - t0
    return SynthResult(audio, took, took) if audio else None


async def synthesize_chunk(text: str, *, api_key: str, voice_id: str,
                           client: Optional[httpx.AsyncClient] = None,
                           latency: str = "balanced", timeout: float = 15.0) -> Optional[SynthResult]:
    text = (text or "").strip()
    if text and piper_enabled():
        return await _synthesize_piper(text)
    if not text or not api_key:
        return None
    own = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    t0 = time.monotonic()
    first: Optional[float] = None
    buf = bytearray()
    try:
        async with client.stream(
            "POST", FISH_TTS_URL,
            # Senza header `model` Fish usa s2.1-pro, a pagamento.
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                     "model": os.getenv("FISH_MODEL", "s2.1-pro-free")},
            json={"text": text, "reference_id": voice_id, "format": "mp3", "mp3_bitrate": 128, "latency": latency},
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                log.error(f"TTS {resp.status_code} for {text[:40]!r}")
                return None
            async for part in resp.aiter_bytes():
                if first is None:
                    first = time.monotonic() - t0
                buf.extend(part)
    except (httpx.HTTPError, OSError) as e:
        log.error(f"TTS error: {e}")
        return None
    finally:
        if own:
            try:
                await client.aclose()
            except Exception as e:      # never turn a clean None into an exception
                log.debug(f"TTS client close failed: {e}")
    if not buf:
        return None
    return SynthResult(bytes(buf), first if first is not None else 0.0, time.monotonic() - t0)
