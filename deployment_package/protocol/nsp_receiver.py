"""
nsp_receiver.py
Project STREAMSENSE — Track A
NSP Protocol v1.2 — TCP server/receiver + inference.

Listens for an incoming NSP v1.2 connection (e.g. from Kavish's sender, or
from nsp_sender.py on this machine for self-test), receives DATA frames
(64000-byte float32 LE payloads = 16000 samples @ 16 kHz), runs each frame
through:

    mel_pipeline.preprocess()  ->  [1,1,64,97] float32
    ONNX Runtime inference     ->  [1,10] logits
    argmax -> class label (class_labels.json)

and prints the prediction for each frame. Stops cleanly on an EOF frame or
client disconnect.

Usage:
    python nsp_receiver.py --port 50007 --model ..\\onnx_models\\streamsense_model_fp32.onnx

Per NSP v1.2 / project rules:
    - TCP_NODELAY = 1 (Nagle disabled)
    - 4-byte LE length prefix + 48-byte header + payload, all little-endian
    - dtype must be FLOAT32 (0x03), payload_bytes must be 64000
    - This script implements RECEIVER (server role). A corresponding SENDER
      mode of this node (acting as client toward Kavish's receiver) is
      implemented separately in nsp_sender.py — both Track A and Track B
      run both roles.
"""

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

import nsp_protocol as nsp
from mel_pipeline import preprocess, OUTPUT_SHAPE


PROJECT_ROOT      = Path(__file__).resolve().parent.parent
CLASS_LABELS_FILE = PROJECT_ROOT / "class_labels.json"


def load_class_labels(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"class_labels.json not found: {path}")
    with open(path, "r") as f:
        raw = json.load(f)
    # keys are strings "0".."9" -> ensure int-indexable list of length 10
    labels = [None] * len(raw)
    for k, v in raw.items():
        labels[int(k)] = v
    return labels


def build_session(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


def run_inference(sess, input_name, output_name, mel_tensor, class_labels):
    """
    mel_tensor: torch.Tensor [1,1,64,97] float32 (from mel_pipeline.preprocess)
    Returns: (predicted_label: str, top1_index: int, logits: np.ndarray [10])
    """
    input_array = mel_tensor.numpy().astype(np.float32)

    if tuple(input_array.shape) != OUTPUT_SHAPE:
        raise ValueError(
            f"Inference input shape mismatch: expected {OUTPUT_SHAPE}, "
            f"got {tuple(input_array.shape)}"
        )

    outputs = sess.run([output_name], {input_name: input_array})
    logits = outputs[0][0]  # [10]

    top1_index = int(np.argmax(logits))
    predicted_label = class_labels[top1_index]
    return predicted_label, top1_index, logits


def handle_connection(conn: socket.socket, addr, sess, input_name, output_name, class_labels):
    print(f"[receiver] connection from {addr}")
    nsp.configure_socket(conn)

    frame_count = 0
    session_id = None
    expected_seq = 0

    try:
        while True:
            try:
                header, payload = nsp.recv_frame(conn)
            except ConnectionError as e:
                print(f"[receiver] connection closed: {e}")
                break

            if session_id is None:
                session_id = header["session_id"]
                print(f"[receiver] session_id={session_id}")

            if header["session_id"] != session_id:
                print(f"[receiver] WARNING: session_id changed mid-stream "
                      f"({session_id} -> {header['session_id']})")
                session_id = header["session_id"]

            if header["sequence_no"] != expected_seq:
                print(f"[receiver] WARNING: sequence gap — expected "
                      f"{expected_seq}, got {header['sequence_no']}")
            expected_seq = header["sequence_no"] + 1

            if header["msg_type"] == nsp.MSG_TYPE_EOF:
                print(f"[receiver] received EOF  seq={header['sequence_no']}")
                break

            if header["msg_type"] != nsp.MSG_TYPE_DATA:
                print(f"[receiver] WARNING: unknown msg_type "
                      f"0x{header['msg_type']:02x}, skipping")
                continue

            if header["payload_bytes"] != nsp.PAYLOAD_BYTES:
                print(f"[receiver] WARNING: unexpected payload_bytes "
                      f"{header['payload_bytes']} (expected {nsp.PAYLOAD_BYTES}), skipping")
                continue

            samples = np.frombuffer(payload, dtype="<f4").copy()  # [16000] float32

            mel_tensor = preprocess(samples)  # [1,1,64,97]
            predicted_label, top1_index, logits = run_inference(
                sess, input_name, output_name, mel_tensor, class_labels
            )

            frame_count += 1
            print(f"[receiver] frame seq={header['sequence_no']:6d}  "
                  f"-> predicted='{predicted_label}' (class {top1_index})  "
                  f"top1_logit={logits[top1_index]:.4f}")

    finally:
        conn.close()
        print(f"[receiver] connection from {addr} closed. "
              f"frames processed: {frame_count}")


def run_receiver(host: str, port: int, model_path: Path, labels_path: Path):
    class_labels = load_class_labels(labels_path)
    print(f"[receiver] loaded {len(class_labels)} class labels: {class_labels}")

    sess, input_name, output_name = build_session(model_path)
    print(f"[receiver] loaded ONNX model: {model_path}")
    print(f"[receiver] input='{input_name}'  output='{output_name}'")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)
    print(f"[receiver] listening on {host}:{port} ...")

    try:
        while True:
            conn, addr = server_sock.accept()
            handle_connection(conn, addr, sess, input_name, output_name, class_labels)
            print("[receiver] waiting for next connection ... (Ctrl+C to stop)")
    except KeyboardInterrupt:
        print("\n[receiver] shutting down (Ctrl+C)")
    finally:
        server_sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="STREAMSENSE NSP v1.2 receiver — receives frames, runs MPIC + ONNX inference."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--model", type=Path, required=True,
                         help="Path to ONNX model (fp32 or int8)")
    parser.add_argument("--labels", type=Path, default=CLASS_LABELS_FILE,
                         help=f"Path to class_labels.json (default: {CLASS_LABELS_FILE})")

    args = parser.parse_args()

    try:
        run_receiver(args.host, args.port, args.model, args.labels)
    except (FileNotFoundError, ValueError) as e:
        print(f"[receiver] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
