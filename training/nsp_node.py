"""
nsp_node.py
Project STREAMSENSE — Track A
NSP Protocol v1.2 — DUAL-ROLE node (SENDER + RECEIVER together).

WHY THIS FILE EXISTS
---------------------
The spec says: "Both Track A and Track B must be client AND server (NSP)."
nsp_sender.py (client-out) and nsp_receiver.py (server-in) already implement
each role separately. This file just RUNS BOTH AT THE SAME TIME from one
process, using two independent TCP connections on two different ports:

    RECEIVER role (server) : this node LISTENS on --listen-port,
                              waits for Kavish's sender to connect in,
                              runs MPIC + ONNX inference on incoming frames.

    SENDER role (client)   : this node CONNECTS OUT to Kavish's receiver
                              at --peer-host:--peer-port and streams WAV
                              frames to it.

These two roles are independent NSP sessions (different sockets, different
session_ids) and run concurrently:
    - the RECEIVER role runs in a background thread (it has to sit and
      wait for an inbound connection, so it can't block the main thread)
    - the SENDER role runs in the main thread (it actively does work —
      read WAV files, send frames, then exits)

If you ONLY want to send, or ONLY want to receive, you can still use
nsp_sender.py / nsp_receiver.py directly — this file is for running both
at once on the same machine/process.

HOW THE TWO HALVES REUSE EXISTING CODE
----------------------------------------
- All wire-format logic (header packing, framing, TCP_NODELAY) comes from
  nsp_protocol.py — unchanged, single source of truth.
- The receiver half is nsp_receiver.run_receiver(), imported directly.
- The sender half is nsp_sender.run_sender(), imported directly.
This file adds NO new protocol logic — it is purely a concurrency wrapper.

USAGE
-----
    python nsp_node.py ^
        --listen-port 50007 ^
        --model ..\\onnx_models\\streamsense_model_fp32.onnx ^
        --peer-host 192.168.1.50 --peer-port 50008 ^
        --wav-dir ..\\recordings

If you omit --peer-host, only the RECEIVER role starts (it will run forever
until Ctrl+C). If you omit --wav-dir/--wav-file, only the RECEIVER role
starts as well (sender role needs something to send).
"""

import argparse
import sys
import threading
from pathlib import Path

import nsp_receiver
import nsp_sender


PROJECT_ROOT      = Path(__file__).resolve().parent.parent
CLASS_LABELS_FILE = PROJECT_ROOT / "class_labels.json"


def receiver_thread_target(listen_host, listen_port, model_path, labels_path, stop_flag):
    """
    Wrapper around nsp_receiver.run_receiver() so it can run in a daemon
    thread. run_receiver() itself loops forever (accepting new connections)
    until KeyboardInterrupt — in a background thread it will simply keep
    running until the whole process exits, which is what we want: the
    receiver role should stay "always on" while the sender role does its
    one-shot job in the main thread.
    """
    try:
        nsp_receiver.run_receiver(listen_host, listen_port, model_path, labels_path)
    except Exception as e:
        print(f"[node][receiver-thread] FATAL: {e}", file=sys.stderr)
        stop_flag.set()


def run_dual_role(listen_host, listen_port, model_path, labels_path,
                   peer_host, peer_port, wav_paths, session_id=None):

    stop_flag = threading.Event()

    # ── Start RECEIVER role in background thread ─────────────────────────
    # daemon=True -> this thread is killed automatically when main thread
    # exits, so we don't need explicit shutdown handling for it here.
    recv_thread = threading.Thread(
        target=receiver_thread_target,
        args=(listen_host, listen_port, model_path, labels_path, stop_flag),
        daemon=True,
        name="nsp-receiver",
    )
    recv_thread.start()
    print(f"[node] RECEIVER role started in background "
          f"(listening on {listen_host}:{listen_port})")

    # ── Run SENDER role in main thread (if a peer + WAVs were given) ─────
    if peer_host is not None and wav_paths:
        print(f"[node] SENDER role starting -> connecting to "
              f"{peer_host}:{peer_port} ...")
        try:
            nsp_sender.run_sender(peer_host, peer_port, wav_paths, session_id=session_id)
        except Exception as e:
            print(f"[node][sender] ERROR: {e}", file=sys.stderr)
        print("[node] SENDER role finished.")
    else:
        print("[node] No --peer-host / WAV input given — "
              "running RECEIVER role only.")

    # ── Keep process alive so the RECEIVER thread keeps serving ──────────
    print("[node] RECEIVER role is still running. Press Ctrl+C to stop.")
    try:
        while recv_thread.is_alive() and not stop_flag.is_set():
            recv_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[node] shutting down (Ctrl+C)")


def main():
    parser = argparse.ArgumentParser(
        description="STREAMSENSE NSP v1.2 dual-role node "
                     "(RECEIVER server thread + SENDER client in main thread)."
    )

    # Receiver-role args (always required — every node listens)
    parser.add_argument("--listen-host", default="0.0.0.0",
                         help="Address this node's RECEIVER binds to (default: 0.0.0.0)")
    parser.add_argument("--listen-port", type=int, required=True,
                         help="Port this node's RECEIVER listens on")
    parser.add_argument("--model", type=Path, required=True,
                         help="Path to ONNX model used by the RECEIVER role")
    parser.add_argument("--labels", type=Path, default=CLASS_LABELS_FILE,
                         help=f"Path to class_labels.json (default: {CLASS_LABELS_FILE})")

    # Sender-role args (optional — only used if a peer is given)
    parser.add_argument("--peer-host", default=None,
                         help="Host/IP of the peer's RECEIVER (enables SENDER role)")
    parser.add_argument("--peer-port", type=int, default=None,
                         help="Port of the peer's RECEIVER")
    parser.add_argument("--wav-dir", type=Path, default=None,
                         help="Directory of .wav files to send (recursive)")
    parser.add_argument("--wav-file", type=Path, default=None,
                         help="Single .wav file to send")
    parser.add_argument("--session-id", type=int, default=None,
                         help="Override SENDER session_id (default: derived from time)")

    args = parser.parse_args()

    wav_paths = []
    if args.peer_host is not None:
        if args.peer_port is None:
            print("[node] ERROR: --peer-port is required when --peer-host is given",
                  file=sys.stderr)
            sys.exit(1)
        try:
            wav_paths = nsp_sender.gather_wav_files(args.wav_dir, args.wav_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"[node] ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        run_dual_role(
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            model_path=args.model,
            labels_path=args.labels,
            peer_host=args.peer_host,
            peer_port=args.peer_port,
            wav_paths=wav_paths,
            session_id=args.session_id,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[node] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
