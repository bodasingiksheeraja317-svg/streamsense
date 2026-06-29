"""
nsp_receiver.py
Project STREAMSENSE — Track A (WA-1, Loopback Validation)

NSP v1.2 Receiver — loopback validation counterpart to nsp_sender.py.

PURPOSE: This is Track A's loopback validation tool, NOT Track B's production
receiver. Its job is to parse incoming NSP packets from nsp_sender.py,
validate header integrity, count frames, and report statistics. It does NOT
perform mel preprocessing or inference — that is Track B's TensorBuilder.

Implements the same dual-mode TCP transport as nsp_sender.py per:
  - "Untitled document (2).md"  — NSP v1.2 spec (header validation, framing)
  - "TRACK_A_TRACK_B_DUAL_MODE_RUNBOOK_v2.md" — Server/Client modes, reconnect

──────────────────────────────────────────────────────────────────────────────
Validation checks per incoming packet:
  1. Length prefix matches actual data received
  2. magic_bytes == b"NSP\\x00"
  3. version == 1
  4. dtype == 0x03 (FLOAT32)
  5. payload_bytes == frame_length * sizeof(dtype)  [spec Section 10]
  6. Length prefix == 48 + payload_bytes            [spec Section 10]
  7. sequence_no is monotonically increasing (gap detection)
  8. session_id is stable within a session
──────────────────────────────────────────────────────────────────────────────

Usage:
  # Server mode (receiver listens, sender connects)
  python nsp_receiver.py

  # Client mode (receiver connects to a listening sender)
  python nsp_receiver.py --mode client --host 127.0.0.1 --port 7654

  # With verbose per-packet output
  python nsp_receiver.py --verbose
  python nsp_receiver.py --n-frames 10 --verbose
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
from pathlib import Path

# ── NSP v1.2 constants (mirrors nsp_sender.py) ────────────────────────────────
NSP_MAGIC          = b"NSP\x00"
NSP_VERSION        = 1
NSP_MSG_DATA       = 0x01
NSP_MSG_EOF        = 0x02
NSP_DTYPE_FLOAT32  = 0x03

NSP_DTYPE_SIZE = {
    0x01: 2,   # INT16
    0x02: 4,   # INT32
    0x03: 4,   # FLOAT32
    0x04: 8,   # FLOAT64
}

NSP_HEADER_FMT  = "<4sHBBQQQIIII"
NSP_HEADER_SIZE = struct.calcsize(NSP_HEADER_FMT)   # 48
NSP_LENGTH_FMT  = "<I"
NSP_MAX_PACKET  = 16_777_216                         # 16 MB

assert NSP_HEADER_SIZE == 48

# ── Logs directory ─────────────────────────────────────────────────────────────
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = Path(__file__).resolve().parent / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ── Global stop flag ───────────────────────────────────────────────────────────
_STOP = threading.Event()


def _signal_handler(signum, frame):
    print("\n[NspReceiver] Shutdown signal — stopping gracefully...")
    _STOP.set()


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Low-level recv helpers ─────────────────────────────────────────────────────

def _recv_exactly(conn: socket.socket, n: int) -> bytes | None:
    """
    Receive exactly n bytes from the socket.
    Returns None on connection loss or stop flag.
    Implements the length-prefixed frame assembler (spec Section 6, runbook ADR-B).
    """
    data = bytearray()
    while len(data) < n:
        if _STOP.is_set():
            return None
        try:
            chunk = conn.recv(n - len(data))
        except OSError:
            return None
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


# ── Header parser + validator ──────────────────────────────────────────────────

class NspHeader:
    """Parsed + validated NSP v1.2 header."""

    __slots__ = (
        "magic_bytes", "version", "message_type", "dtype",
        "sequence_no", "timestamp_us", "session_id",
        "payload_bytes", "sample_rate", "frame_length", "reserved",
    )

    def __init__(self, raw: bytes):
        (
            self.magic_bytes,
            self.version,
            self.message_type,
            self.dtype,
            self.sequence_no,
            self.timestamp_us,
            self.session_id,
            self.payload_bytes,
            self.sample_rate,
            self.frame_length,
            self.reserved,
        ) = struct.unpack(NSP_HEADER_FMT, raw)

    def validate(self) -> list[str]:
        """
        Return list of validation errors (empty = OK).
        Checks: magic, version, dtype, payload_bytes == frame_length * sizeof(dtype).
        """
        errors = []
        if self.magic_bytes != NSP_MAGIC:
            errors.append(f"Bad magic: {self.magic_bytes!r} (expected {NSP_MAGIC!r})")
        if self.version != NSP_VERSION:
            errors.append(f"Bad version: {self.version} (expected {NSP_VERSION})")
        if self.dtype not in NSP_DTYPE_SIZE:
            errors.append(f"Unknown dtype: 0x{self.dtype:02X}")
        elif self.message_type == NSP_MSG_DATA:
            expected_bytes = self.frame_length * NSP_DTYPE_SIZE[self.dtype]
            if self.payload_bytes != expected_bytes:
                errors.append(
                    f"payload_bytes={self.payload_bytes} != "
                    f"frame_length({self.frame_length}) * sizeof(dtype)({NSP_DTYPE_SIZE[self.dtype]}) "
                    f"= {expected_bytes}"
                )
        return errors


# ── Session statistics ─────────────────────────────────────────────────────────

class RecvStats:
    def __init__(self, host: str, port: int, mode: str):
        self.host            = host
        self.port            = port
        self.mode            = mode
        self.start_dt        = datetime.datetime.now()
        self.end_dt          = None
        self.session_id      = None
        self.packets_recv    = 0
        self.bytes_recv      = 0
        self.frames_data     = 0
        self.frames_eof      = 0
        self.errors          = 0
        self.seq_gaps        = 0
        self.last_seq        = None
        # Latency tracking (sender ts_us vs receiver arrival)
        self.latency_sum_us  = 0
        self.latency_count   = 0

    def finalise(self):
        self.end_dt = datetime.datetime.now()

    def to_dict(self) -> dict:
        duration = (self.end_dt - self.start_dt).total_seconds() if self.end_dt else 0
        mean_lat = (self.latency_sum_us / self.latency_count) if self.latency_count else 0
        return {
            "role"              : "receiver",
            "host"              : self.host,
            "port"              : self.port,
            "mode"              : self.mode,
            "session_id"        : self.session_id,
            "start_time"        : self.start_dt.isoformat(),
            "end_time"          : self.end_dt.isoformat() if self.end_dt else None,
            "duration_sec"      : round(duration, 3),
            "packets_received"  : self.packets_recv,
            "bytes_received"    : self.bytes_recv,
            "data_frames"       : self.frames_data,
            "eof_frames"        : self.frames_eof,
            "validation_errors" : self.errors,
            "sequence_gaps"     : self.seq_gaps,
            "mean_latency_us"   : round(mean_lat, 2),
            "audio_duration_sec": self.frames_data * 1.0,
        }

    def save(self) -> Path:
        ts   = self.start_dt.strftime("%Y%m%d_%H%M%S")
        path = LOGS_DIR / f"nsp_recv_session_{ts}.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ── Runtime statistics writer ──────────────────────────────────────────────────

def write_runtime_stats(stats: "RecvStats") -> None:
    """
    Write live receive-side statistics to stats/nsp_receiver_runtime.json.
    Called after every packet is received. Uses atomic write (tmp → rename) to
    avoid partial reads by an external consumer.

    Fields:
      frames       — total DATA frames received this session
      data_bytes   — total bytes received (length-prefix + header + payload)
      data_rate    — bytes/s since session start
      recv_rate    — frames/s since session start
    """
    elapsed = (datetime.datetime.now() - stats.start_dt).total_seconds()
    elapsed = elapsed if elapsed > 0 else 1e-9   # guard against div-by-zero

    payload = {
        "host"       : stats.host,
        "port"       : stats.port,
        "mode"       : stats.mode,
        "frames"     : stats.frames_data,
        "data_bytes" : stats.bytes_recv,
        "data_rate"  : round(stats.bytes_recv / elapsed, 2),   # bytes/s
        "recv_rate"  : round(stats.frames_data / elapsed, 4),  # frames/s
        "elapsed_sec": round(elapsed, 3),
        "updated_at" : datetime.datetime.now().isoformat(),
    }

    target = STATS_DIR / "nsp_receiver_runtime.json"
    tmp    = STATS_DIR / "nsp_receiver_runtime.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(target)


# ── Session runner ─────────────────────────────────────────────────────────────

def _run_session(
    conn    : socket.socket,
    stats   : RecvStats,
    verbose : bool,
    n_frames: int | None,
):
    """
    Receive and validate NSP packets from an established connection.
    Implements length-prefixed framing (spec Section 6).
    """
    print(f"[NspReceiver] Session active | peer={conn.getpeername()}")
    last_seq   = None
    session_id = None

    while not _STOP.is_set():
        if n_frames is not None and stats.frames_data >= n_frames:
            break

        # ── Read 4-byte length prefix ────────────────────────────────────────
        raw_len = _recv_exactly(conn, 4)
        if raw_len is None:
            break
        (msg_len,) = struct.unpack(NSP_LENGTH_FMT, raw_len)

        if msg_len < NSP_HEADER_SIZE or msg_len > NSP_MAX_PACKET:
            print(f"[NspReceiver] [ERROR] Invalid msg_len={msg_len} — "
                  f"expected {NSP_HEADER_SIZE} ≤ len ≤ {NSP_MAX_PACKET}")
            stats.errors += 1
            break

        # ── Read header + payload ────────────────────────────────────────────
        raw_msg = _recv_exactly(conn, msg_len)
        if raw_msg is None:
            break

        raw_header  = raw_msg[:NSP_HEADER_SIZE]
        raw_payload = raw_msg[NSP_HEADER_SIZE:]

        stats.packets_recv += 1
        stats.bytes_recv   += 4 + msg_len

        write_runtime_stats(stats)

        # ── Parse header ─────────────────────────────────────────────────────
        hdr    = NspHeader(raw_header)
        errors = hdr.validate()

        # ── Validate length prefix == 48 + payload_bytes (spec Section 10) ──
        if msg_len != NSP_HEADER_SIZE + hdr.payload_bytes:
            errors.append(
                f"Length prefix {msg_len} != 48 + payload_bytes {hdr.payload_bytes} "
                f"= {NSP_HEADER_SIZE + hdr.payload_bytes}"
            )

        if errors:
            for e in errors:
                print(f"[NspReceiver] [VALIDATION ERROR] {e}")
            stats.errors += len(errors)

        # ── Session ID tracking ───────────────────────────────────────────────
        if session_id is None:
            session_id        = hdr.session_id
            stats.session_id  = session_id
            print(f"[NspReceiver] New session: session_id={session_id} | "
                  f"sample_rate={hdr.sample_rate} Hz | frame_length={hdr.frame_length}")
        elif hdr.session_id != session_id:
            print(f"[NspReceiver] [WARN] session_id changed: "
                  f"{session_id} → {hdr.session_id}")
            session_id       = hdr.session_id
            last_seq         = None

        # ── Sequence gap detection ────────────────────────────────────────────
        if last_seq is not None and hdr.message_type == NSP_MSG_DATA:
            expected = last_seq + 1
            if hdr.sequence_no != expected:
                gap = hdr.sequence_no - expected
                print(f"[NspReceiver] [WARN] Sequence gap: "
                      f"expected {expected}, got {hdr.sequence_no} (gap={gap})")
                stats.seq_gaps += 1

        # ── Latency tracking ──────────────────────────────────────────────────
        now_us = time.time_ns() // 1_000
        lat_us = now_us - hdr.timestamp_us
        if 0 < lat_us < 10_000_000:   # ignore unrealistic values (>10s)
            stats.latency_sum_us  += lat_us
            stats.latency_count   += 1

        # ── Message type handling ─────────────────────────────────────────────
        if hdr.message_type == NSP_MSG_DATA:
            stats.frames_data += 1
            last_seq           = hdr.sequence_no

            if verbose:
                lat_str = f"{lat_us:>8} µs" if stats.latency_count else "N/A"
                print(
                    f"[NspReceiver] DATA  "
                    f"seq={hdr.sequence_no:>6} | "
                    f"session={hdr.session_id} | "
                    f"len={hdr.frame_length:>6} smp | "
                    f"payload={hdr.payload_bytes:>7} B | "
                    f"lat={lat_str}"
                )
            elif stats.frames_data % 100 == 0:
                print(f"[NspReceiver] Received {stats.frames_data} DATA frames | "
                      f"{stats.bytes_recv/1024:.1f} KB total")

        elif hdr.message_type == NSP_MSG_EOF:
            stats.frames_eof += 1
            print(f"[NspReceiver] EOF received | seq={hdr.sequence_no} | "
                  f"total_data_frames={stats.frames_data}")
            break

        else:
            print(f"[NspReceiver] [WARN] Unknown message_type=0x{hdr.message_type:02X}")
            stats.errors += 1


# ── Server mode ────────────────────────────────────────────────────────────────

def run_server(
    host    : str,
    port    : int,
    stats   : RecvStats,
    verbose : bool,
    n_frames: int | None,
):
    """
    Server mode: bind/listen, accept() with 100ms select() timeout loop.
    Per runbook Section 2, Section 12.D.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.setblocking(False)

    print(f"[NspReceiver] Server mode | listening on {host}:{port}")

    try:
        while not _STOP.is_set():
            readable, _, _ = select.select([srv], [], [], 0.1)
            if not readable:
                continue

            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(True)
            print(f"[NspReceiver] Sender connected: {addr}")

            try:
                _run_session(conn, stats, verbose, n_frames)
            finally:
                conn.close()
                print(f"[NspReceiver] Sender disconnected: {addr}")

            if n_frames is not None and stats.frames_data >= n_frames:
                break
    finally:
        srv.close()


# ── Client mode ────────────────────────────────────────────────────────────────

def run_client(
    host                : str,
    port                : int,
    stats               : RecvStats,
    verbose             : bool,
    n_frames            : int | None,
    reconnect_initial_ms: int = 1000,
    reconnect_max_ms    : int = 5000,
):
    """
    Client mode: connect to a listening sender with 2000ms timeout.
    Linear backoff on failure (runbook Section 3, Section 12.C, 12.E).
    """
    print(f"[NspReceiver] Client mode | connecting to {host}:{port}")
    backoff_ms = reconnect_initial_ms

    while not _STOP.is_set():
        if n_frames is not None and stats.frames_data >= n_frames:
            break

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)

        try:
            sock.connect_ex((host, port))
        except OSError:
            pass

        _, writable, _ = select.select([], [sock], [], 2.0)

        if writable:
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err == 0:
                sock.setblocking(True)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[NspReceiver] Connected to {host}:{port}")

                try:
                    _run_session(sock, stats, verbose, n_frames)
                finally:
                    sock.close()

                backoff_ms = reconnect_initial_ms
                continue

        sock.close()
        if _STOP.is_set():
            break

        print(f"[NspReceiver] Connection failed. Retrying in {backoff_ms}ms...")
        slept = 0
        while slept < backoff_ms and not _STOP.is_set():
            time.sleep(0.1)
            slept += 100

        backoff_ms = min(backoff_ms + 1000, reconnect_max_ms)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "NSP v1.2 Receiver — loopback validator for nsp_sender.py.\n"
            "Parses and validates NSP frames. Does NOT perform mel/inference."
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
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Stop after N DATA frames (default: run until EOF/disconnect)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-packet details")
    parser.add_argument("--reconnect-initial-ms", type=int, default=1000)
    parser.add_argument("--reconnect-max-ms",     type=int, default=5000)
    args = parser.parse_args()

    print("=" * 72)
    print(f"NSP v1.2 Receiver | mode={args.mode} | {args.host}:{args.port}")
    print(f"  n_frames={args.n_frames or 'infinite'} | verbose={args.verbose}")
    print(f"  Logs  → {LOGS_DIR}")
    print(f"  Stats → {STATS_DIR}")
    print("=" * 72)

    stats = RecvStats(args.host, args.port, args.mode)

    try:
        if args.mode == "server":
            run_server(args.host, args.port, stats, args.verbose, args.n_frames)
        else:
            run_client(
                args.host, args.port, stats, args.verbose, args.n_frames,
                reconnect_initial_ms=args.reconnect_initial_ms,
                reconnect_max_ms    =args.reconnect_max_ms,
            )
    finally:
        stats.finalise()
        log_path = stats.save()
        d = stats.to_dict()

        print("\n" + "=" * 72)
        print("RECEIVER SESSION SUMMARY")
        print(f"  Packets received    : {d['packets_received']}")
        print(f"  Bytes received      : {d['bytes_received']:,}  "
              f"({d['bytes_received']/1024:.1f} KB)")
        print(f"  DATA frames         : {d['data_frames']}")
        print(f"  EOF frames          : {d['eof_frames']}")
        print(f"  Validation errors   : {d['validation_errors']}")
        print(f"  Sequence gaps       : {d['sequence_gaps']}")
        print(f"  Mean latency        : {d['mean_latency_us']:.1f} µs")
        print(f"  Audio duration      : {d['audio_duration_sec']:.1f} s")
        print(f"  Duration            : {d['duration_sec']:.1f} s")
        if d['validation_errors'] == 0 and d['sequence_gaps'] == 0:
            print("\n  [PASS] All frames valid. Zero errors. Zero sequence gaps.")
        else:
            print(f"\n  [WARN] {d['validation_errors']} errors, "
                  f"{d['sequence_gaps']} sequence gaps.")
        print(f"  Stats saved to      : {log_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()
