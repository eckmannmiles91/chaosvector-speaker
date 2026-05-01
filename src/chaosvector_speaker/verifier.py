"""Speaker verification — cosine similarity against enrolled voice prints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .embedder import AudioEmbedder

logger = logging.getLogger(__name__)


class SpeakerVerifier:
    """Load enrolled speaker embeddings and verify new audio against them.

    Enrollment file is a numpy .npz where each key is a speaker name and
    the value is their averaged d-vector embedding (float32, shape [D]).
    An optional key ``__thresholds__`` maps speaker names to per-speaker
    cosine-similarity thresholds (JSON-encoded dict stored as a string).
    """

    def __init__(
        self,
        enrollment_file: str | Path,
        model_name: str = "ecapa_tdnn",
        model_path: Optional[str] = None,
        default_threshold: float = 0.65,
    ) -> None:
        self.enrollment_file = Path(enrollment_file)
        self.default_threshold = default_threshold
        self.embedder = AudioEmbedder(model_name=model_name, model_path=model_path)

        # speaker_name -> embedding (float32, shape [D])
        self.speakers: dict[str, np.ndarray] = {}
        # speaker_name -> per-speaker threshold override
        self.thresholds: dict[str, float] = {}

        if self.enrollment_file.exists():
            self._load_enrollment()

    # ── public API ──────────────────────────────────────────────────────

    def verify(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[Optional[str], float]:
        """Identify speaker from raw audio samples.

        Returns
        -------
        (speaker_name, confidence) where speaker_name is None if no
        enrolled speaker exceeds their threshold.
        """
        if not self.speakers:
            logger.warning("No speakers enrolled — returning None")
            return None, 0.0

        embedding = self.embedder.embed(audio, sample_rate=sample_rate)
        return self._match(embedding)

    def verify_file(self, path: str | Path) -> tuple[Optional[str], float]:
        """Convenience wrapper — load a WAV and verify."""
        audio, sr = self._load_wav(path)
        return self.verify(audio, sample_rate=sr)

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> None:
        """Add (or overwrite) an enrolled speaker."""
        self.speakers[name] = embedding.astype(np.float32)
        if threshold is not None:
            self.thresholds[name] = threshold
        self._save_enrollment()
        logger.info("Enrolled speaker %r (%d-dim embedding)", name, embedding.shape[0])

    def remove(self, name: str) -> bool:
        """Remove an enrolled speaker. Returns True if they existed."""
        removed = self.speakers.pop(name, None) is not None
        self.thresholds.pop(name, None)
        if removed:
            self._save_enrollment()
        return removed

    def list_speakers(self) -> list[str]:
        return list(self.speakers.keys())

    # ── matching ────────────────────────────────────────────────────────

    def _match(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        """Find the best-matching enrolled speaker via cosine similarity."""
        best_name: Optional[str] = None
        best_score: float = -1.0

        for name, enrolled in self.speakers.items():
            score = self._cosine_similarity(embedding, enrolled)
            if score > best_score:
                best_score = score
                best_name = name

        # Apply per-speaker threshold (or default)
        if best_name is not None:
            threshold = self.thresholds.get(best_name, self.default_threshold)
            if best_score < threshold:
                logger.debug(
                    "Best match %r (%.3f) below threshold %.3f",
                    best_name,
                    best_score,
                    threshold,
                )
                return None, best_score

        return best_name, best_score

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors, clamped to [-1, 1]."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, sim))

    # ── persistence ─────────────────────────────────────────────────────

    def _load_enrollment(self) -> None:
        import json

        data = np.load(self.enrollment_file, allow_pickle=False)
        for key in data.files:
            if key == "__thresholds__":
                self.thresholds = json.loads(str(data[key]))
            else:
                self.speakers[key] = data[key].astype(np.float32)
        logger.info(
            "Loaded %d enrolled speaker(s) from %s",
            len(self.speakers),
            self.enrollment_file,
        )

    def _save_enrollment(self) -> None:
        import json

        save_dict: dict[str, np.ndarray] = dict(self.speakers)
        if self.thresholds:
            save_dict["__thresholds__"] = np.array(json.dumps(self.thresholds))
        np.savez(self.enrollment_file, **save_dict)

    # ── audio I/O ───────────────────────────────────────────────────────

    @staticmethod
    def _load_wav(path: str | Path) -> tuple[np.ndarray, int]:
        """Load WAV as float32 mono, return (samples, sample_rate)."""
        import wave

        path = Path(path)
        with wave.open(str(path), "rb") as wf:
            assert wf.getsampwidth() == 2, "Only 16-bit WAV supported"
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Mix to mono if stereo
        if wf.getnchannels() == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        return audio, sr
