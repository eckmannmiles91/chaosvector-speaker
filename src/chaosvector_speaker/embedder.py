"""Audio embedding extraction via ONNX Runtime.

Supports ECAPA-TDNN, TitaNet, or custom ONNX models that produce a fixed-size
d-vector from a mel spectrogram (or raw waveform, depending on model).
Target: <50ms inference on Raspberry Pi CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── model registry ──────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict] = {
    "ecapa_tdnn": {
        "sample_rate": 16000,
        "n_mels": 80,
        "embedding_dim": 192,
        "window_ms": 25,
        "hop_ms": 10,
        "max_duration_s": 8.0,
        "input_type": "mel",  # model expects mel spectrogram
    },
    "titanet": {
        "sample_rate": 16000,
        "n_mels": 80,
        "embedding_dim": 192,
        "window_ms": 25,
        "hop_ms": 10,
        "max_duration_s": 8.0,
        "input_type": "mel",
    },
    "custom": {
        "sample_rate": 16000,
        "n_mels": 80,
        "embedding_dim": 256,
        "window_ms": 25,
        "hop_ms": 10,
        "max_duration_s": 8.0,
        "input_type": "raw",  # model takes raw waveform
    },
}


class AudioEmbedder:
    """Extract speaker embeddings from audio using an ONNX model."""

    def __init__(
        self,
        model_name: str = "ecapa_tdnn",
        model_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["custom"]).copy()
        self.model_path = Path(model_path) if model_path else None
        self._session = None  # lazy-loaded ONNX session

    # ── public API ──────────────────────────────────────────────────────

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract d-vector embedding from raw float32 mono audio.

        Returns
        -------
        np.ndarray of shape [embedding_dim], dtype float32.
        """
        audio = self._preprocess(audio, sample_rate)
        model_input = self._prepare_model_input(audio)
        embedding = self._infer(model_input)
        return self._l2_normalize(embedding)

    def embed_batch(
        self, segments: list[np.ndarray], sample_rate: int = 16000
    ) -> np.ndarray:
        """Embed multiple audio segments, return (N, D) array."""
        embeddings = [self.embed(seg, sample_rate) for seg in segments]
        return np.stack(embeddings, axis=0)

    # ── preprocessing pipeline ──────────────────────────────────────────

    def _preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Full preprocessing: resample -> VAD trim -> normalize -> clip length."""
        target_sr = self.config["sample_rate"]

        # Resample if needed
        if sample_rate != target_sr:
            audio = self._resample(audio, sample_rate, target_sr)

        # VAD trimming — remove leading/trailing silence
        audio = self._vad_trim(audio, target_sr)

        # Peak normalization
        audio = self._normalize(audio)

        # Truncate to max duration
        max_samples = int(self.config["max_duration_s"] * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return audio

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling (good enough for voice)."""
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        n_out = int(len(audio) * ratio)
        indices = np.arange(n_out) / ratio
        indices = np.clip(indices, 0, len(audio) - 1)
        left = indices.astype(np.int64)
        right = np.minimum(left + 1, len(audio) - 1)
        frac = (indices - left).astype(np.float32)
        return audio[left] * (1.0 - frac) + audio[right] * frac

    @staticmethod
    def _vad_trim(
        audio: np.ndarray,
        sample_rate: int,
        frame_ms: int = 30,
        energy_threshold: float = 0.01,
    ) -> np.ndarray:
        """Simple energy-based VAD to trim silence from start/end."""
        frame_len = int(sample_rate * frame_ms / 1000)
        n_frames = len(audio) // frame_len

        if n_frames == 0:
            return audio

        # Compute per-frame energy
        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.mean(frames ** 2, axis=1)

        # Find first and last frame above threshold
        active = np.where(energy > energy_threshold)[0]
        if len(active) == 0:
            return audio  # all silence, return as-is

        start = active[0] * frame_len
        end = min((active[-1] + 1) * frame_len, len(audio))
        return audio[start:end]

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """Peak-normalize to [-1, 1]."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    # ── feature extraction ──────────────────────────────────────────────

    def _prepare_model_input(self, audio: np.ndarray) -> np.ndarray:
        """Convert preprocessed audio to the format the ONNX model expects."""
        if self.config["input_type"] == "raw":
            # Model takes raw waveform: shape [1, T]
            return audio.reshape(1, -1).astype(np.float32)

        # Mel spectrogram input: shape [1, n_mels, T]
        mel = self._log_mel_spectrogram(audio)
        return mel[np.newaxis, :, :].astype(np.float32)

    def _log_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram using numpy (no librosa dependency).

        Returns shape [n_mels, T].
        """
        sr = self.config["sample_rate"]
        n_mels = self.config["n_mels"]
        win_len = int(sr * self.config["window_ms"] / 1000)
        hop_len = int(sr * self.config["hop_ms"] / 1000)
        n_fft = 2 ** int(np.ceil(np.log2(win_len)))

        # STFT
        window = np.hanning(win_len).astype(np.float32)
        n_frames = 1 + (len(audio) - win_len) // hop_len
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)

        for i in range(n_frames):
            start = i * hop_len
            frame = audio[start : start + win_len] * window
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[:win_len] = frame
            spectrum = np.fft.rfft(padded)
            stft[:, i] = np.abs(spectrum) ** 2

        # Mel filterbank
        mel_basis = self._mel_filterbank(sr, n_fft, n_mels)
        mel_spec = mel_basis @ stft

        # Log compression
        mel_spec = np.log(np.maximum(mel_spec, 1e-10))
        return mel_spec

    @staticmethod
    def _mel_filterbank(
        sample_rate: int, n_fft: int, n_mels: int
    ) -> np.ndarray:
        """Create a mel-scale filterbank matrix [n_mels, n_fft//2+1]."""
        low_freq = 0.0
        high_freq = sample_rate / 2.0

        def hz_to_mel(hz: float) -> float:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_low = hz_to_mel(low_freq)
        mel_high = hz_to_mel(high_freq)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        n_freqs = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    # ── ONNX inference ──────────────────────────────────────────────────

    def _get_session(self):
        """Lazy-load ONNX Runtime session."""
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. "
                f"Download a speaker embedding model and set --model-path."
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Loaded ONNX model from %s", self.model_path)
        return self._session

    def _infer(self, model_input: np.ndarray) -> np.ndarray:
        """Run ONNX inference, return raw embedding vector."""
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: model_input})
        # Most speaker models return shape [1, D] — squeeze batch dim
        embedding = outputs[0].squeeze()
        return embedding.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v
