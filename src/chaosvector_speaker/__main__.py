"""CLI entry point for chaosvector-speaker."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="chaosvector-speaker",
        description="Speaker verification and identification with family voice enrollment",
    )
    sub = parser.add_subparsers(dest="command")

    # ── enroll ──────────────────────────────────────────────────────────
    enroll_p = sub.add_parser("enroll", help="Enroll a new speaker")
    enroll_p.add_argument("--name", required=True, help="Speaker name")
    enroll_p.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing 3-5 WAV samples",
    )
    enroll_p.add_argument(
        "--enrollment-file",
        default="speakers.npz",
        help="Path to enrollment .npz file (default: speakers.npz)",
    )
    enroll_p.add_argument(
        "--model",
        default="ecapa_tdnn",
        choices=["ecapa_tdnn", "titanet", "custom"],
        help="Embedding model to use (default: ecapa_tdnn)",
    )

    # ── verify ──────────────────────────────────────────────────────────
    verify_p = sub.add_parser("verify", help="Verify speaker from audio")
    verify_p.add_argument("--audio", required=True, help="Path to WAV file")
    verify_p.add_argument(
        "--enrollment-file",
        default="speakers.npz",
        help="Path to enrollment .npz file",
    )
    verify_p.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold (default: 0.65)",
    )
    verify_p.add_argument("--model", default="ecapa_tdnn")

    # ── serve ───────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start Wyoming protocol server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=10600)
    serve_p.add_argument("--enrollment-file", default="speakers.npz")
    serve_p.add_argument("--model", default="ecapa_tdnn")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "enroll":
        from .enroll import enroll_speaker

        enroll_speaker(
            name=args.name,
            audio_dir=args.audio_dir,
            enrollment_file=args.enrollment_file,
            model_name=args.model,
        )

    elif args.command == "verify":
        from .verifier import SpeakerVerifier

        verifier = SpeakerVerifier(
            enrollment_file=args.enrollment_file,
            model_name=args.model,
            default_threshold=args.threshold,
        )
        speaker, confidence = verifier.verify_file(args.audio)
        if speaker is not None:
            print(f"Identified: {speaker} (confidence={confidence:.3f})")
        else:
            print(f"Unknown speaker (best confidence={confidence:.3f})")

    elif args.command == "serve":
        from .wyoming_server import run_server

        run_server(
            host=args.host,
            port=args.port,
            enrollment_file=args.enrollment_file,
            model_name=args.model,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
