"""Enrollment pipeline — create voice prints from audio samples."""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .embedder import AudioEmbedder
from .verifier import SpeakerVerifier

logger = logging.getLogger(__name__)


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a 16-bit WAV as float32 mono."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, f"Only 16-bit WAV supported, got {path}"
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, sr


def enroll_speaker(
    name: str,
    audio_dir: str,
    enrollment_file: str = "speakers.npz",
    model_name: str = "ecapa_tdnn",
    model_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Enroll a speaker from a directory of WAV samples.

    Extracts an embedding from each sample, averages them to produce a
    stable voice print, and saves it to the enrollment file.

    Parameters
    ----------
    name : str
        Speaker name (used as key in the .npz file).
    audio_dir : str
        Directory containing 3-5 WAV files for this speaker.
    enrollment_file : str
        Path to the .npz enrollment database.
    model_name : str
        Embedding model name.
    model_path : str, optional
        Path to ONNX model file.
    threshold : float, optional
        Per-speaker verification threshold override.

    Returns
    -------
    np.ndarray
        The averaged, L2-normalized voice print.
    """
    audio_path = Path(audio_dir)
    wav_files = sorted(audio_path.glob("*.wav"))

    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")
    if len(wav_files) < 3:
        logger.warning(
            "Only %d sample(s) found — recommend at least 3 for stable enrollment",
            len(wav_files),
        )

    embedder = AudioEmbedder(model_name=model_name, model_path=model_path)

    embeddings: list[np.ndarray] = []
    for wav in wav_files:
        logger.info("Processing %s", wav.name)
        audio, sr = load_wav(wav)
        emb = embedder.embed(audio, sample_rate=sr)
        embeddings.append(emb)
        logger.info("  -> embedding shape %s, norm %.4f", emb.shape, np.linalg.norm(emb))

    # Average embeddings and re-normalize
    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    # Save via verifier
    verifier = SpeakerVerifier(
        enrollment_file=enrollment_file,
        model_name=model_name,
        model_path=model_path,
    )
    verifier.enroll(name, avg_embedding, threshold=threshold)

    logger.info(
        "Enrolled %r from %d samples -> %s",
        name,
        len(embeddings),
        enrollment_file,
    )
    print(f"Enrolled '{name}' from {len(embeddings)} samples into {enrollment_file}")
    return avg_embedding


def record_and_enroll(
    name: str,
    n_samples: int = 3,
    duration_s: float = 5.0,
    enrollment_file: str = "speakers.npz",
    model_name: str = "ecapa_tdnn",
    model_path: Optional[str] = None,
) -> np.ndarray:
    """Interactive enrollment — record samples via sounddevice.

    Requires the ``sounddevice`` optional dependency.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is required for recording. "
            "Install with: pip install chaosvector-speaker[record]"
        )

    sr = 16000
    embedder = AudioEmbedder(model_name=model_name, model_path=model_path)
    embeddings: list[np.ndarray] = []

    for i in range(n_samples):
        input(f"\nPress Enter to record sample {i+1}/{n_samples} ({duration_s}s)...")
        print("Recording...")
        audio = sd.rec(
            int(duration_s * sr),
            samplerate=sr,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.squeeze()
        print(f"  Recorded {len(audio)/sr:.1f}s")

        emb = embedder.embed(audio, sample_rate=sr)
        embeddings.append(emb)

    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    verifier = SpeakerVerifier(
        enrollment_file=enrollment_file,
        model_name=model_name,
        model_path=model_path,
    )
    verifier.enroll(name, avg_embedding)

    print(f"\nEnrolled '{name}' from {n_samples} recordings into {enrollment_file}")
    return avg_embedding
