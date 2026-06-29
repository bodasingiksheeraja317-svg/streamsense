"""
nsp_sender.py
Project STREAMSENSE — Track A (WA-1, D5-D6)

NSP v1.2 Sender — sends streaming audio frames from Track A to Track B.

Implements the Network Stream Protocol v1.2 (Freeze Candidate) exactly as
specified in "Untitled document (2).md" and the dual-mode TCP transport from
"TRACK_A_TRACK_B_DUAL_MODE_RUNBOOK_v2.md".

──────────────────────────────────────────────────────────────────────────────
NSP v1.2 Wire Format (from spec, Section 8-9)
──────────────────────────────────────────────────────────────────────────────

  Packet = Length Prefix (4B) + Header (48B) + Payload (variable)
  Total  = 4 + 48 + payload_bytes

  Header layout (Little-Endian, 48 bytes):
    0x00  magic_bytes    char[4]    4   b"NSP\\x00"
    0x04  version        uint16_t   2   1
    0x06  message_type   uint8_t    1   0x01=DATA, 0x02=EOF/CLOSE
    0x07  dtype          uint8_t    1   0x03=FLOAT32
    0x08  sequence_no    uint64_t   8   monotonic counter
    0x10  timestamp      uint64_t   8   microseconds since epoch
    0x18  session_id     uint64_t   8   unique per connection lifetime
    0x20  payload_bytes  uint32_t   4   frame_length * sizeof(dtype)
    0x24  sample_rate    uint32_t   4   16000 Hz
    0x28  frame_length   uint32_t   4   samples per frame (N)
    0x2C  reserved       uint32_t   4   0

  Struct format : "<4sHBBQQQIIII"  → 48 bytes (little-endian)
  Length prefix : "<I"             → 4 bytes  (little-endian uint32)

  Validator constraint: payload_bytes == frame_length * sizeof(dtype)
  TCP constraint      : TCP_NODELAY must be set (Nagle's algorithm disabled)
  Max packet size     : 16 MB (16,777,216 bytes)

──────────────────────────────────────────────────────────────────────────────
Transport Modes (from Runbook, Sections 2-5)
──────────────────────────────────────────────────────────────────────────────

  Server Mode (default, port 7654):
    bind() → listen() → accept() with 100ms select() timeout loop
    → stream → client disconnects → loop back to accept()
    Graceful shutdown: stop flag checked every 100ms

  Client Mode:
    connect() non-blocking with 2000ms select() timeout
    → stream → remote disconnects
    → linear backoff: initial=1000ms, max=5000ms, +1000ms per retry
    Backoff is broken into 100ms chunks (checks stop flag every chunk)

──────────────────────────────────────────────────────────────────────────────
What is sent per packet:
  Raw float32 audio samples — NEVER mel, NEVER normalised tensors.
  Track B's TensorBuilder receives the raw samples and does mel preprocessing.
  Each packet carries exactly frame_length (default 16000) float32 samples.
──────────────────────────────────────────────────────────────────────────────

Usage:
  # Server mode (Track A listens, Track B connects)
  python nsp_sender.py

  # Client mode (Track A connects to Track B's listener)
  python nsp_sender.py --mode client --host 192.168.1.50 --port 7654

  # Custom configuration
  python nsp_sender.py --mode server --port 8888 --frame-len 16000 --n-frames 100
  python nsp_sender.py --sources project    # only data/raw
  python nsp_sender.py --sources unknown   # only unknown_data
  python nsp_sender.py --sources both      # both (default)
"""

import os
import sys
import time
import json
import struct
import select
import socket
import signal
import argparse
import threading
import datetime
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stream_simulator  import StreamSimulator, DEFAULT_DATA_DIRS

# ── Environment / paths ────────────────────────────────────────────────────────
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = Path(__file__).resolve().parent / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ── NSP v1.2 constants (spec Section 9, 11) ────────────────────────────────────
NSP_MAGIC          = b"NSP\x00"
NSP_VERSION        = 1
NSP_MSG_DATA       = 0x01
NSP_MSG_EOF        = 0x02
NSP_DTYPE_INT16    = 0x01
NSP_DTYPE_INT32    = 0x02
NSP_DTYPE_FLOAT32  = 0x03
NSP_DTYPE_FLOAT64  = 0x04

NSP_DTYPE_SIZE = {
    NSP_DTYPE_INT16   : 2,
    NSP_DTYPE_INT32   : 4,
    NSP_DTYPE_FLOAT32 : 4,
    NSP_DTYPE_FLOAT64 : 8,
}

NSP_HEADER_FMT    = "<4sHBBQQQIIII"    # little-endian, 48 bytes
NSP_HEADER_SIZE   = struct.calcsize(NSP_HEADER_FMT)   # must be 48
NSP_LENGTH_FMT    = "<I"               # little-endian uint32 length prefix
NSP_MAX_PACKET    = 16_777_216         # 16 MB hard limit (spec Section 7)

assert NSP_HEADER_SIZE == 48, f"Header size mismatch: {NSP_HEADER_SIZE}"

# ── Global stop flag (set by SIGINT/SIGTERM) ───────────────────────────────────
_STOP = threading.Event()


def _signal_handler(signum, frame):
    print("\n[NspSender] Shutdown signal received — stopping gracefully...")
    _STOP.set()


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── NSP packet builder ─────────────────────────────────────────────────────────

def build_header(
    msg_type   : int,
    dtype_code : int,
    seq_no     : int,
    ts_us      : int,
    session_id : int,
    payload_byt: int,
    sample_rate: int,
    frame_len  : int,
) -> bytes:
    """Serialise 48-byte NSP v1.2 header (little-endian)."""
    return struct.pack(
        NSP_HEADER_FMT,
        NSP_MAGIC,
        NSP_VERSION,
        msg_type,
        dtype_code,
        seq_no,
        ts_us,
        session_id,
        payload_byt,
        sample_rate,
        frame_len,
        0,           # reserved
    )


def build_data_packet(
    raw_frame  : np.ndarray,   # [N] float32 1D
    seq_no     : int,
    session_id : int,
    sample_rate: int = 16000,
) -> bytes:
    """
    Build a complete NSP DATA packet:
      Length prefix (4B) + Header (48B) + Payload (N × 4B)

    Validates payload_bytes == frame_length * sizeof(dtype) before sending.
    """
    frame  = raw_frame.astype(np.float32).ravel()
    N      = len(frame)
    p_bytes= N * NSP_DTYPE_SIZE[NSP_DTYPE_FLOAT32]   # N * 4

    assert p_bytes <= NSP_MAX_PACKET - NSP_HEADER_SIZE, \
        f"Payload exceeds 16MB limit: {p_bytes}"

    ts_us  = time.time_ns() // 1_000               # microseconds
    header = build_header(
        NSP_MSG_DATA, NSP_DTYPE_FLOAT32,
        seq_no, ts_us, session_id,
        p_bytes, sample_rate, N,
    )
    payload= frame.tobytes()                        # little-endian float32 bytes
    length = struct.pack(NSP_LENGTH_FMT, NSP_HEADER_SIZE + p_bytes)
    return length + header + payload


def build_eof_packet(seq_no: int, session_id: int, sample_rate: int = 16000) -> bytes:
    """Build NSP EOF/CLOSE packet (payload_bytes = 0, frame_length = 0)."""
    ts_us  = time.time_ns() // 1_000
    header = build_header(
        NSP_MSG_EOF, NSP_DTYPE_FLOAT32,
        seq_no, ts_us, session_id,
        0, sample_rate, 0,
    )
    length = struct.pack(NSP_LENGTH_FMT, NSP_HEADER_SIZE)
    return length + header


# ── Session statistics ─────────────────────────────────────────────────────────

class SessionStats:
    """Accumulates per-session statistics for JSON report."""

    def __init__(self, host: str, port: int, mode: str):
        self.host        = host
        self.port        = port
        self.mode        = mode
        self.session_id  = 0
        self.start_dt    = datetime.datetime.now()
        self.end_dt      = None
        self.packets_sent= 0
        self.bytes_sent  = 0
        self.frames_total= 0
        self.dropped     = 0
        self.welford_sum_mean = 0.0   # for aggregate Welford (Chan merge)
        self.welford_sum_std  = 0.0
        self.welford_n_frames = 0
        self.sources_project  = 0
        self.sources_unknown  = 0

    def finalise(self):
        self.end_dt = datetime.datetime.now()

    def to_dict(self) -> dict:
        duration = (
            (self.end_dt - self.start_dt).total_seconds()
            if self.end_dt else 0.0
        )
        return {
            "session_id"        : self.session_id,
            "host"              : self.host,
            "port"              : self.port,
            "mode"              : self.mode,
            "start_time"        : self.start_dt.isoformat(),
            "end_time"          : self.end_dt.isoformat() if self.end_dt else None,
            "duration_sec"      : round(duration, 3),
            "packets_sent"      : self.packets_sent,
            "bytes_sent"        : self.bytes_sent,
            "frames_total"      : self.frames_total,
            "dropped_frames"    : self.dropped,
            "audio_duration_sec": self.frames_total * 1.0,  # 1 frame = 1 sec
            "sources_project"   : self.sources_project,
            "sources_unknown"   : self.sources_unknown,
        }

    def save(self) -> Path:
        ts   = self.start_dt.strftime("%Y%m%d_%H%M%S")
        path = LOGS_DIR / f"nsp_session_{ts}.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ── Runtime statistics writer ──────────────────────────────────────────────────

def write_runtime_stats(stats: "SessionStats") -> None:
    """
    Write live send-side statistics to stats/nsp_sender_runtime.json.
    Called after every packet send. Uses atomic write (tmp → rename) to avoid
    partial reads by an external consumer.

    Fields:
      frames      — total DATA frames sent this session
      data_bytes  — total payload bytes sent (header + payload)
      data_rate   — bytes/s since session start
      send_rate   — frames/s since session start
    """
    elapsed = (datetime.datetime.now() - stats.start_dt).total_seconds()
    elapsed = elapsed if elapsed > 0 else 1e-9   # guard against div-by-zero

    payload = {
        "host"       : stats.host,
        "port"       : stats.port,
        "mode"       : stats.mode,
        "frames"     : stats.frames_total,
        "data_bytes" : stats.bytes_sent,
        "data_rate"  : round(stats.bytes_sent / elapsed, 2),   # bytes/s
        "send_rate"  : round(stats.frames_total / elapsed, 4), # frames/s
        "elapsed_sec": round(elapsed, 3),
        "updated_at" : datetime.datetime.now().isoformat(),
    }

    target = STATS_DIR / "nsp_sender_runtime.json"
    tmp    = STATS_DIR / "nsp_sender_runtime.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(target)


# ── Frame generator (StreamSimulator → fixed-size frames) ─────────────────────

def _frame_generator(data_dirs, frame_len: int = 16000):
    """
    Wraps StreamSimulator (validation config: 16kHz mono float32) and
    accumulates samples into fixed frame_len-sample frames.

    Yields np.ndarray [frame_len] float32 — the raw audio payload per NSP packet.
    """
    import torch
    sim = StreamSimulator(data_dirs=data_dirs, random_config=False,
                          chunk_min=512, chunk_max=4096)
    gen = sim.generator()

    buf = np.empty(0, dtype=np.float32)
    while not _STOP.is_set():
        chunk = next(gen)
        # Simulator yields [1, N] planar float32 in validation mode
        arr   = chunk.numpy().ravel().astype(np.float32)
        buf   = np.concatenate([buf, arr])
        while len(buf) >= frame_len:
            yield buf[:frame_len].copy()
            buf = buf[frame_len:]


# ── TCP send helper ────────────────────────────────────────────────────────────

def _sendall_safe(conn: socket.socket, data: bytes) -> bool:
    """
    Send all bytes. Returns False if the connection was broken.
    Respects _STOP flag for large sends.
    """
    try:
        conn.sendall(data)
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


# ── Session runner ─────────────────────────────────────────────────────────────

def _run_session(
    conn       : socket.socket,
    stats      : SessionStats,
    data_dirs,
    frame_len  : int,
    sample_rate: int,
    n_frames   : int | None,
):
    """
    Stream NSP DATA packets over an established TCP connection.
    Sends EOF packet on clean exit or disconnection.
    """
    seq        = 0
    session_id = int(time.time() * 1_000_000)   # unique per connection
    stats.session_id = session_id

    print(f"[NspSender] Session started | session_id={session_id} | "
          f"frame_len={frame_len} | sample_rate={sample_rate}")

    gen = _frame_generator(data_dirs, frame_len)

    try:
        for raw_frame in gen:
            if _STOP.is_set():
                break
            if n_frames is not None and stats.frames_total >= n_frames:
                break

            packet = build_data_packet(raw_frame, seq, session_id, sample_rate)

            if not _sendall_safe(conn, packet):
                print("[NspSender] Connection broken mid-stream.")
                stats.dropped += 1
                break

            seq                  += 1
            stats.packets_sent   += 1
            stats.bytes_sent     += len(packet)
            stats.frames_total   += 1

            write_runtime_stats(stats)

            if stats.frames_total % 100 == 0:
                print(f"[NspSender] Sent {stats.frames_total} frames | "
                      f"{stats.bytes_sent/1024:.1f} KB total")

    finally:
        # Always attempt to send EOF (best-effort)
        try:
            eof = build_eof_packet(seq, session_id, sample_rate)
            conn.sendall(eof)
            print(f"[NspSender] EOF sent | total_frames={stats.frames_total}")
        except OSError:
            pass


# ── Server mode loop ───────────────────────────────────────────────────────────

def run_server(
    host       : str,
    port       : int,
    data_dirs,
    frame_len  : int,
    sample_rate: int,
    n_frames   : int | None,
    stats      : SessionStats,
):
    """
    Server mode: bind/listen/accept loop.
    Per runbook Section 2: accept() uses 100ms select() timeout to check stop flag.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.setblocking(False)

    print(f"[NspSender] Server mode | listening on {host}:{port}")

    try:
        while not _STOP.is_set():
            # 100ms select() timeout — allows graceful shutdown check
            readable, _, _ = select.select([srv], [], [], 0.1)
            if not readable:
                continue

            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(True)
            print(f"[NspSender] Client connected: {addr}")

            try:
                _run_session(conn, stats, data_dirs, frame_len, sample_rate, n_frames)
            finally:
                conn.close()
                print(f"[NspSender] Client disconnected: {addr}")

            if _STOP.is_set():
                break
            if n_frames is not None and stats.frames_total >= n_frames:
                break

    finally:
        srv.close()


# ── Client mode loop ───────────────────────────────────────────────────────────

def run_client(
    host       : str,
    port       : int,
    data_dirs,
    frame_len  : int,
    sample_rate: int,
    n_frames   : int | None,
    stats      : SessionStats,
    reconnect_initial_ms: int = 1000,
    reconnect_max_ms    : int = 5000,
):
    """
    Client mode: connect with 2000ms timeout, linear backoff on failure.
    Per runbook Section 3 and Section 8:
      - Non-blocking connect() + select() with 2000ms timeout
      - Linear backoff: initial=1000ms, max=5000ms, +1000ms per retry
      - Backoff sleep broken into 100ms chunks to honour stop flag
    """
    print(f"[NspSender] Client mode | connecting to {host}:{port}")
    backoff_ms = reconnect_initial_ms

    while not _STOP.is_set():
        if n_frames is not None and stats.frames_total >= n_frames:
            break

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)

        try:
            sock.connect_ex((host, port))
        except OSError:
            pass

        # select() wait for writeability (2000ms timeout per runbook Section 12.C)
        _, writable, _ = select.select([], [sock], [], 2.0)

        if writable:
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err == 0:
                sock.setblocking(True)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[NspSender] Connected to {host}:{port}")

                try:
                    _run_session(sock, stats, data_dirs, frame_len, sample_rate, n_frames)
                finally:
                    sock.close()

                backoff_ms = reconnect_initial_ms   # reset on clean disconnect
                continue

        # Connection failed — apply linear backoff
        sock.close()
        if _STOP.is_set():
            break

        print(f"[NspSender] Connection failed. Retrying in {backoff_ms}ms...")
        # Sleep in 100ms chunks (runbook Section 12.E)
        slept = 0
        while slept < backoff_ms and not _STOP.is_set():
            time.sleep(0.1)
            slept += 100

        backoff_ms = min(backoff_ms + 1000, reconnect_max_ms)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "NSP v1.2 Sender — streams raw float32 audio frames over TCP.\n"
            "Implements dual-mode (server/client) per STREAMSENSE runbook."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["server", "client"], default="server",
                        help="Transport mode (default: server)")
    parser.add_argument("--host", default=os.environ.get("STREAMSENSE_HOST", "127.0.0.1"),
                        help="Bind/connect host (default: 127.0.0.1 or $STREAMSENSE_HOST)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("STREAMSENSE_PORT", "7654")),
                        help="Bind/connect port (default: 7654 or $STREAMSENSE_PORT)")
    parser.add_argument("--frame-len", type=int, default=16000,
                        help="Samples per NSP frame (default: 16000 = 1s @ 16kHz)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Stop after N frames (default: run indefinitely)")
    parser.add_argument("--sources", choices=["project", "unknown", "both"], default="both",
                        help="Audio source pool (default: both)")
    parser.add_argument("--reconnect-initial-ms", type=int, default=1000,
                        help="Initial reconnect backoff in ms (client mode, default: 1000)")
    parser.add_argument("--reconnect-max-ms", type=int, default=5000,
                        help="Max reconnect backoff in ms (client mode, default: 5000)")
    args = parser.parse_args()

    # ── Select data directories per --sources ─────────────────────────────────
    if args.sources == "project":
        data_dirs = [ROOT / "data" / "raw"]
    elif args.sources == "unknown":
        data_dirs = [ROOT / "unknown_data"]
    else:
        data_dirs = list(DEFAULT_DATA_DIRS)

    print("=" * 72)
    print(f"NSP v1.2 Sender | mode={args.mode} | {args.host}:{args.port}")
    print(f"  frame_len={args.frame_len} | sample_rate={args.sample_rate} Hz")
    print(f"  n_frames={args.n_frames or 'infinite'} | sources={args.sources}")
    print(f"  Packet size: 4 + 48 + {args.frame_len * 4} = {4 + 48 + args.frame_len * 4} bytes")
    print(f"  Logs  → {LOGS_DIR}")
    print(f"  Stats → {STATS_DIR}")
    print("=" * 72)

    stats = SessionStats(args.host, args.port, args.mode)

    try:
        if args.mode == "server":
            run_server(
                args.host, args.port, data_dirs,
                args.frame_len, args.sample_rate, args.n_frames,
                stats,
            )
        else:
            run_client(
                args.host, args.port, data_dirs,
                args.frame_len, args.sample_rate, args.n_frames,
                stats,
                reconnect_initial_ms=args.reconnect_initial_ms,
                reconnect_max_ms    =args.reconnect_max_ms,
            )
    finally:
        stats.finalise()
        log_path = stats.save()
        d = stats.to_dict()
        print("\n" + "=" * 72)
        print("SESSION SUMMARY")
        print(f"  Packets sent     : {d['packets_sent']}")
        print(f"  Bytes sent       : {d['bytes_sent']:,}  "
              f"({d['bytes_sent']/1024:.1f} KB)")
        print(f"  Frames total     : {d['frames_total']}")
        print(f"  Dropped frames   : {d['dropped_frames']}")
        print(f"  Audio duration   : {d['audio_duration_sec']:.1f} s")
        print(f"  Duration         : {d['duration_sec']:.1f} s")
        print(f"  Stats saved to   : {log_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()
