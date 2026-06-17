"""
nsp_protocol.py
Project STREAMSENSE — Track A
NSP Protocol v1.2 — shared header pack/unpack utilities.

Frame on the wire = [4-byte LE length prefix][48-byte header][payload]
    length_prefix = 48 + payload_bytes   (LE uint32)

Header layout (48 bytes, ALL little-endian):
    magic         4s   = b"NSP\\0"
    version       H    = 1
    msg_type      B    = 0x01 DATA / 0x02 EOF
    dtype         B    = 0x03 FLOAT32
    sequence_no   Q    starts at 0, +1 per frame
    timestamp     Q    microseconds
    session_id    Q    constant per TCP session
    payload_bytes I    = 64000  (16000 * 4)
    sample_rate   I    = 16000
    frame_length  I    = 16000
    reserved      I    = 0
"""

import struct
import socket
import time

# ── Constants ─────────────────────────────────────────────────────────────────
MAGIC          = b"NSP\x00"
VERSION        = 1

MSG_TYPE_DATA  = 0x01
MSG_TYPE_EOF   = 0x02

DTYPE_FLOAT32  = 0x03

SAMPLE_RATE    = 16000
FRAME_LENGTH   = 16000
PAYLOAD_BYTES  = FRAME_LENGTH * 4   # 64000  (float32)

HEADER_SIZE    = 48
LENGTH_PREFIX_SIZE = 4

# struct format: < = little-endian, no padding
#   4s H B B Q Q Q I I I I
#   4 + 2 + 1 + 1 + 8 + 8 + 8 + 4 + 4 + 4 + 4 = 48
HEADER_FORMAT  = "<4sHBBQQQIIII"

assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE, \
    f"Header format size mismatch: {struct.calcsize(HEADER_FORMAT)} != {HEADER_SIZE}"


def pack_header(msg_type, sequence_no, session_id,
                 payload_bytes=PAYLOAD_BYTES,
                 sample_rate=SAMPLE_RATE,
                 frame_length=FRAME_LENGTH,
                 timestamp_us=None,
                 reserved=0):
    """Pack a 48-byte NSP v1.2 header. Returns bytes."""
    if timestamp_us is None:
        timestamp_us = int(time.time() * 1_000_000)

    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        msg_type,
        DTYPE_FLOAT32,
        sequence_no,
        timestamp_us,
        session_id,
        payload_bytes,
        sample_rate,
        frame_length,
        reserved,
    )


def unpack_header(header_bytes):
    """Unpack a 48-byte NSP v1.2 header into a dict. Raises on bad magic/version."""
    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"Header must be {HEADER_SIZE} bytes, got {len(header_bytes)}")

    (magic, version, msg_type, dtype, sequence_no, timestamp_us,
     session_id, payload_bytes, sample_rate, frame_length, reserved) = \
        struct.unpack(HEADER_FORMAT, header_bytes)

    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r} (expected {MAGIC!r})")
    if version != VERSION:
        raise ValueError(f"Unsupported NSP version: {version} (expected {VERSION})")
    if dtype != DTYPE_FLOAT32:
        raise ValueError(f"Unsupported dtype: 0x{dtype:02x} (expected 0x{DTYPE_FLOAT32:02x})")

    return {
        "magic":         magic,
        "version":       version,
        "msg_type":      msg_type,
        "dtype":         dtype,
        "sequence_no":   sequence_no,
        "timestamp_us":  timestamp_us,
        "session_id":    session_id,
        "payload_bytes": payload_bytes,
        "sample_rate":   sample_rate,
        "frame_length":  frame_length,
        "reserved":      reserved,
    }


def build_frame(msg_type, sequence_no, session_id,
                 payload=b"", timestamp_us=None):
    """
    Build a full on-wire frame: [4-byte LE length prefix][48-byte header][payload].

    For msg_type == MSG_TYPE_EOF, payload should be b"" and payload_bytes=0.
    """
    payload_bytes = len(payload)
    header = pack_header(
        msg_type=msg_type,
        sequence_no=sequence_no,
        session_id=session_id,
        payload_bytes=payload_bytes,
        timestamp_us=timestamp_us,
    )
    length_prefix = struct.pack("<I", HEADER_SIZE + payload_bytes)
    return length_prefix + header + payload


# ── Socket helpers ───────────────────────────────────────────────────────────
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from sock or raise ConnectionError on EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(
                f"Connection closed by peer while expecting {remaining} more bytes "
                f"(received {n - remaining}/{n})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket):
    """
    Read one full NSP v1.2 frame from sock.

    Returns:
        (header_dict, payload_bytes)

    Raises ConnectionError on clean disconnect, ValueError on malformed frame.
    """
    length_prefix_bytes = recv_exact(sock, LENGTH_PREFIX_SIZE)
    (frame_len,) = struct.unpack("<I", length_prefix_bytes)

    if frame_len < HEADER_SIZE:
        raise ValueError(f"Frame length {frame_len} smaller than header size {HEADER_SIZE}")

    header_bytes = recv_exact(sock, HEADER_SIZE)
    header = unpack_header(header_bytes)

    payload_len = frame_len - HEADER_SIZE
    if header["payload_bytes"] != payload_len:
        raise ValueError(
            f"payload_bytes field ({header['payload_bytes']}) does not match "
            f"frame length minus header ({payload_len})"
        )

    payload = recv_exact(sock, payload_len) if payload_len > 0 else b""
    return header, payload


def configure_socket(sock: socket.socket):
    """Apply NSP v1.2 socket options: TCP_NODELAY = 1 (Nagle disabled)."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
