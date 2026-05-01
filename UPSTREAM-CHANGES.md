# Changes from Upstream

ChaosVector Speaker replaces [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (MIT) for speaker verification.

## Why we forked
- Resemblyzer uses PyTorch for inference — ~1s per verification on Pi CPU
- No ONNX export support
- Generic embeddings not optimized for family voice differentiation
- Enrollment requires interactive recording sessions

## What we changed

### ONNX Runtime Inference
- **Upstream:** PyTorch model (~1s on Pi CPU)
- **Ours:** ONNX Runtime with CPU/GPU execution providers. Target: <50ms per verification on Pi CPU (~20x speedup).

### Per-Speaker Confidence Thresholds
- **Upstream:** Single global threshold for all speakers
- **Ours:** Configurable threshold per enrolled speaker. Miles might need 0.65, Eli might need 0.70 — voices with similar characteristics get tighter thresholds.

### File-Based Enrollment
- **Upstream:** Requires interactive recording with microphone
- **Ours:** Enrollment from pre-recorded WAV files. `chaosvector-speaker enroll --name Miles --audio-dir ./samples/miles/`. Can also record interactively if preferred.

### Numpy Mel Spectrogram
- **Upstream:** Depends on librosa for audio processing
- **Ours:** Pure numpy mel spectrogram + energy-based VAD trimming. Same feature extraction as chaosvector-wake — shared code path, consistent behavior.

### Combined Wake + Speaker ID
- **Upstream:** Speaker verification runs as a separate step after wake word detection
- **Ours:** Architecture supports receiving audio from the wake word detector directly. Identify the speaker AT wake time using the same audio — eliminates the separate verification step and reduces total latency.

## Target Performance
| Metric | Resemblyzer | ChaosVector Speaker |
|--------|------------|-------------------|
| Inference | ~1000ms | <50ms (target) |
| Framework | PyTorch | ONNX Runtime |
| Enrollment | Interactive only | WAV files or interactive |
| Thresholds | Global | Per-speaker |
