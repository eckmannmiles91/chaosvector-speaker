# ChaosVector Speaker — Architecture Notes

## Purpose

Drop-in replacement for Resemblyzer-based speaker verification in the homelab voice pipeline. Designed to run at wake-word detection time so speaker identity is known before the STT/intent step.

## Design decisions

- **Pure numpy mel spectrogram** — avoids librosa/torchaudio dependency. The mel filterbank and STFT are implemented from scratch in `embedder.py`. Good enough for speaker verification (not production ASR).
- **ONNX Runtime only** — no PyTorch at runtime. Models must be exported to ONNX first. This gets ~10x speedup on CPU and avoids the 500MB+ PyTorch wheel on Pi.
- **Enrollment as .npz** — simple, no database. Each speaker is a key mapping to their averaged d-vector. Thresholds stored as JSON in a special `__thresholds__` key.
- **Energy VAD for trimming** — simple but effective for cutting silence before/after speech. Not a full VAD (no webrtcvad dependency).

## File layout

```
src/chaosvector_speaker/
  __init__.py       — version
  __main__.py       — CLI (enroll / verify / serve)
  embedder.py       — audio preprocessing + ONNX inference
  verifier.py       — cosine similarity matching against enrolled speakers
  enroll.py         — enrollment pipeline (file-based and interactive)
  wyoming_server.py — optional TCP server for Wyoming protocol
```

## Key classes

- `AudioEmbedder` — stateless audio-to-embedding pipeline. Handles resampling, VAD trim, normalization, mel spectrogram, ONNX inference, L2 normalization.
- `SpeakerVerifier` — loads enrollment file, delegates to embedder, runs cosine similarity against all enrolled speakers, applies per-speaker thresholds.

## Performance targets

- <50ms embedding extraction on Raspberry Pi 4 CPU
- <5ms cosine similarity matching (negligible for <20 speakers)
- Total pipeline: <100ms from audio buffer to speaker identity

## Future work

- Quantized INT8 ONNX models for even faster Pi inference
- Online enrollment update (add samples without re-averaging from scratch)
- Integration with openWakeWord to share the audio buffer directly
- Cluster-based rejection (detect unknown speakers more robustly)
