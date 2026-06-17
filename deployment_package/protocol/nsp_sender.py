"""
nsp_sender.py
Project STREAMSENSE — Track A
NSP Protocol v1.2 — TCP client/sender.

Reads WAV files (mono or stereo, any sample rate/length), converts each to
exactly 16000 float32 samples at 16 kHz (pad/crop, stereo->mono via simple
averaging on read), and streams them to a peer (Kavish's NSP receiver) as
NSP v1.2 DATA frames, followed by a single EOF frame at the end of session.

Usage:
    python nsp_sender.py --host 127.0.0.1 --port 50007 --wav-dir ..\\recordings
    python nsp_sender.py --host 127.0.0.1 --port 50007 --wav-file test.wav

Per NSP v1.2 / project rules:
    - TCP_NODELAY = 1 (Nagle disabled)
    - 4-byte LE length prefix + 48-byte header + payload
    - sequence_no starts at 0, +1 per DATA frame (not incremented for EOF)
    - session_id constant per TCP session (derived from connect time)
    - payload_bytes = 64000 (16000 samples * float32), sample_rate = 16000,
      frame_length = 16000
    - This script implements SENDER. A corresponding RECEIVER mode of this
      node (acting as server for Kavish's sender) is implemented separately
      in nsp_receiver.py — both Track A and Track B run both roles.
"""

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torchaudio

import nsp_protocol as nsp


FRAME_LENGTH = nsp.FRAME_LENGTH   # 16000
SAMPLE_RATE  = nsp.SAMPLE_RATE    # 16000


def load_wav_as_frames(path: Path):
    """
    Load a WAV file and split/pad it into one or more 16000-sample
    float32 mono frames at 16 kHz.

    Returns:
        list[np.ndarray] — each array shape [16000], dtype float32
    """
    waveform, sr = torchaudio.load(str(path))  # waveform: [C, T]

    # Resample to 16 kHz if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    samples = waveform.squeeze(0).numpy().astype(np.float32)  # [T]
    total_len = samples.shape[0]

    if total_len == 0:
        return []

    frames = []
    if total_len <= FRAME_LENGTH:
        # Single frame, zero-pad on the right
        frame = np.zeros(FRAME_LENGTH, dtype=np.float32)
        frame[:total_len] = samples
        frames.append(frame)
    else:
        # Split into consecutive FRAME_LENGTH chunks, pad the last chunk
        n_full = total_len // FRAME_LENGTH
        for i in range(n_full):
            frames.append(samples[i * FRAME_LENGTH:(i + 1) * FRAME_LENGTH].copy())

        remainder = total_len - n_full * FRAME_LENGTH
        if remainder > 0:
            frame = np.zeros(FRAME_LENGTH, dtype=np.float32)
            frame[:remainder] = samples[n_full * FRAME_LENGTH:]
            frames.append(frame)

    return frames


def gather_wav_files(wav_dir: Path = None, wav_file: Path = None):
    if wav_file is not None:
        if not wav_file.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_file}")
        return [wav_file]

    if wav_dir is not None:
        if not wav_dir.exists():
            raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
        files = sorted(wav_dir.rglob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No .wav files found under: {wav_dir}")
        return files

    raise ValueError("Either --wav-dir or --wav-file must be provided")


def run_sender(host: str, port: int, wav_paths, session_id: int = None):
    if session_id is None:
        session_id = int(time.time() * 1_000_000) & 0xFFFFFFFFFFFFFFFF

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    nsp.configure_socket(sock)

    print(f"[sender] connecting to {host}:{port} ...")
    sock.connect((host, port))
    print(f"[sender] connected. session_id={session_id}")

    sequence_no = 0
    total_frames_sent = 0

    try:
        for wav_path in wav_paths:
            print(f"[sender] loading {wav_path}")
            frames = load_wav_as_frames(wav_path)

            for frame in frames:
                payload = frame.tobytes()  # float32 LE, 64000 bytes

                if len(payload) != nsp.PAYLOAD_BYTES:
                    raise ValueError(
                        f"Frame payload size mismatch: {len(payload)} "
                        f"!= expected {nsp.PAYLOAD_BYTES}"
                    )

                wire_frame = nsp.build_frame(
                    msg_type=nsp.MSG_TYPE_DATA,
                    sequence_no=sequence_no,
                    session_id=session_id,
                    payload=payload,
                )
                sock.sendall(wire_frame)

                print(f"[sender] sent DATA  seq={sequence_no:6d}  "
                      f"file={wav_path.name}  bytes={len(wire_frame)}")

                sequence_no += 1
                total_frames_sent += 1

        # Send EOF frame (no payload). sequence_no continues from last DATA
        # frame's sequence (i.e. this EOF gets the next sequence number).
        eof_frame = nsp.build_frame(
            msg_type=nsp.MSG_TYPE_EOF,
            sequence_no=sequence_no,
            session_id=session_id,
            payload=b"",
        )
        sock.sendall(eof_frame)
        print(f"[sender] sent EOF   seq={sequence_no:6d}")

    finally:
        sock.close()
        print(f"[sender] done. total DATA frames sent: {total_frames_sent}")


def main():
    parser = argparse.ArgumentParser(
        description="STREAMSENSE NSP v1.2 sender — streams WAV frames to a peer receiver."
    )
    parser.add_argument("--host", required=True, help="Receiver host/IP")
    parser.add_argument("--port", type=int, required=True, help="Receiver port")
    parser.add_argument("--wav-dir", type=Path, default=None,
                         help="Directory of .wav files to send (recursive)")
    parser.add_argument("--wav-file", type=Path, default=None,
                         help="Single .wav file to send")
    parser.add_argument("--session-id", type=int, default=None,
                         help="Override session_id (default: derived from current time)")

    args = parser.parse_args()

    try:
        wav_paths = gather_wav_files(args.wav_dir, args.wav_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"[sender] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    run_sender(args.host, args.port, wav_paths, session_id=args.session_id)


if __name__ == "__main__":
    main()
