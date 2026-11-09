"""
stt.py — riconoscimento vocale locale con faster-whisper.

Attivo con JARVIS_STT=local. Il browser registra una frase (WAV mono 16 kHz,
16 bit) e la manda a POST /api/stt; qui diventa testo, senza servizi esterni.

Impostazioni (.env):
  JARVIS_STT_MODEL   tiny | base | small | medium | large-v3   (predefinito: small)
  JARVIS_STT_DEVICE  cpu | cuda                                 (predefinito: cpu)
  JARVIS_LANG        es. it-IT -> lingua "it"; vuoto = rilevamento automatico
"""
from __future__ import annotations

import io
import logging
import os
import re
import threading
import wave

log = logging.getLogger("jarvis.stt")

MAX_WAV_BYTES = 4_000_000          # ~2 minuti a 16 kHz: ben oltre una frase

_model = None
_lock = threading.Lock()

# Frasi che Whisper "allucina" su silenzio o rumore (sottotitoli dei dataset).
_HALLUCINATIONS = re.compile(
    r"amara\.org|sottotitoli|subtitles by|grazie per la visione|"
    r"thanks? (for|you for) watching|iscriviti al canale|"
    r"^\W*(grazie|thank you)\W*\Z",
    re.IGNORECASE)


def enabled() -> bool:
    return os.getenv("JARVIS_STT", "").lower() == "local"


def _language() -> str | None:
    lang = os.getenv("JARVIS_LANG", "").strip()
    return lang[:2].lower() if lang else None


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        name = os.getenv("JARVIS_STT_MODEL", "small")
        device = os.getenv("JARVIS_STT_DEVICE", "cpu")
        compute = "int8" if device == "cpu" else "float16"
        _model = WhisperModel(name, device=device, compute_type=compute)
        log.info(f"stt: loaded whisper {name} on {device} ({compute})")
    return _model


def warm() -> None:
    """Carica il modello in anticipo, così la prima frase non aspetta."""
    with _lock:
        _load()


def _samples(wav_bytes: bytes):
    import numpy as np
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != 16000:
            raise ValueError("expected 16 kHz mono 16-bit WAV")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(wav_bytes: bytes) -> str:
    """WAV -> testo. Bloccante: chiamare in un thread."""
    audio = _samples(wav_bytes)
    if audio.size < 16000 * 0.3:
        return ""
    with _lock:
        model = _load()
        segments, _info = model.transcribe(
            audio, language=_language(), beam_size=1,
            condition_on_previous_text=False, vad_filter=False)
        kept = []
        for s in segments:
            if s.no_speech_prob > 0.6 or s.avg_logprob < -1.0:
                continue
            text = s.text.strip()
            if text and not _HALLUCINATIONS.search(text):
                kept.append(text)
    return " ".join(kept).strip()
