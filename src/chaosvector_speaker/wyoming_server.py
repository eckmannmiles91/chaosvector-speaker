"""Optional Wyoming protocol server for remote speaker verification.

Implements a minimal Wyoming satellite that accepts audio chunks,
runs speaker verification, and returns the identified speaker as
an event attribute.

Requires: pip install chaosvector-speaker[wyoming]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    verifier,
) -> None:
    """Handle a single Wyoming client connection.

    Protocol sketch (simplified):
    1. Client sends audio-start event with sample rate
    2. Client streams audio-chunk events
    3. Client sends audio-stop
    4. Server runs verification, replies with a describe event containing
       speaker name and confidence
    """
    import json
    import numpy as np

    addr = writer.get_extra_info("peername")
    logger.info("Connection from %s", addr)

    audio_chunks: list[bytes] = []
    sample_rate = 16000

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            try:
                event = json.loads(line.decode("utf-8").strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            event_type = event.get("type", "")

            if event_type == "audio-start":
                audio_chunks.clear()
                sample_rate = event.get("data", {}).get("rate", 16000)
                logger.debug("Audio start, sr=%d", sample_rate)

            elif event_type == "audio-chunk":
                # Expect base64-encoded int16 PCM
                import base64

                raw = base64.b64decode(event.get("data", {}).get("audio", ""))
                audio_chunks.append(raw)

            elif event_type == "audio-stop":
                if not audio_chunks:
                    continue

                # Concatenate and convert to float32
                raw = b"".join(audio_chunks)
                audio = (
                    np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                )

                speaker, confidence = verifier.verify(audio, sample_rate=sample_rate)

                response = {
                    "type": "speaker-result",
                    "data": {
                        "speaker": speaker,
                        "confidence": round(confidence, 4),
                    },
                }
                writer.write((json.dumps(response) + "\n").encode("utf-8"))
                await writer.drain()
                logger.info("Result: %s (%.3f)", speaker, confidence)
                audio_chunks.clear()

    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        await writer.wait_closed()
        logger.info("Connection closed: %s", addr)


def run_server(
    host: str = "0.0.0.0",
    port: int = 10600,
    enrollment_file: str = "speakers.npz",
    model_name: str = "ecapa_tdnn",
    model_path: Optional[str] = None,
) -> None:
    """Start the Wyoming speaker verification server."""
    from .verifier import SpeakerVerifier

    verifier = SpeakerVerifier(
        enrollment_file=enrollment_file,
        model_name=model_name,
        model_path=model_path,
    )

    logger.info(
        "Starting Wyoming server on %s:%d with %d enrolled speaker(s)",
        host,
        port,
        len(verifier.speakers),
    )

    async def _serve() -> None:
        server = await asyncio.start_server(
            lambda r, w: _handle_client(r, w, verifier),
            host,
            port,
        )
        async with server:
            await server.serve_forever()

    asyncio.run(_serve())
