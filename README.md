# ChaosVector Speaker

Speaker verification and identification engine with family voice enrollment. Designed as a fast, trainable replacement for Resemblyzer-based speaker verification.

## Key features

- **ONNX inference** instead of PyTorch (~10x faster on CPU)
- **Custom enrollment** with family voices (3-5 samples per speaker)
- **Wake-word integration** — runs at wake-word time using the same audio buffer
- **Per-speaker thresholds** for tunable accept/reject behavior
- **Wyoming protocol server** for remote speaker verification in Home Assistant pipelines

## Quick start

```bash
pip install -e .

# Enroll a speaker from WAV samples
chaosvector-speaker enroll --name Miles --audio-dir ./samples/miles/

# Verify a speaker
chaosvector-speaker verify --audio test.wav

# Start Wyoming server
chaosvector-speaker serve --port 10600
```

## Enrollment

Place 3-5 WAV files (16-bit, 16kHz preferred) per speaker in a directory, then run:

```bash
chaosvector-speaker enroll --name Miles --audio-dir ./samples/miles/
chaosvector-speaker enroll --name Sarah --audio-dir ./samples/sarah/
```

Embeddings are averaged and saved to `speakers.npz`.

## Models

Supply an ONNX speaker embedding model. Supported architectures:

| Model | Embedding dim | Notes |
|-------|--------------|-------|
| ECAPA-TDNN | 192 | Default, good accuracy/speed tradeoff |
| TitaNet | 192 | NVIDIA's model, slightly better accuracy |
| Custom | configurable | Any ONNX model producing a fixed-size vector |

## Architecture

```
audio -> preprocess (VAD trim, normalize, resample)
      -> mel spectrogram (numpy, no librosa)
      -> ONNX model inference
      -> L2-normalized d-vector
      -> cosine similarity vs enrolled speakers
      -> (speaker_name | None, confidence)
```

## Wyoming integration

The Wyoming server accepts audio chunks over TCP and returns speaker identification results. Use it as a satellite in a Home Assistant voice pipeline to gate commands by speaker identity.

## License

MIT
