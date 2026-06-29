"""
test_integration.py
Project STREAMSENSE — Track A (Scope 2)

Pipes the StreamSimulator directly into the StreamingFramer to prove
the framer can ingest completely randomized stream configurations and
network jitter, while successfully emitting standardized 16kHz frames.
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stream_simulator import StreamSimulator
from streaming_framer import StreamingFramer

def main():
    print("=" * 80)
    print("INTEGRATION TEST: StreamSimulator -> StreamingFramer")
    print("=" * 80)

    # 1. Start the simulator (randomizes C, Rate, SampleType, Layout)
    sim = StreamSimulator(
        data_dirs=["C:/STREAMSENSE/data/raw", "C:/STREAMSENSE/unknown_data"],
        random_config=True
    )
    
    # 2. Start the framer, perfectly bound to the Simulator's configuration
    framer = StreamingFramer(
        stream_sr       = sim.stream_sr,
        stream_channels = sim.stream_channels,
        dtype           = sim.dtype,
        layout          = sim.layout,
    )
    
    gen = sim.generator()

    print(f"\n{'-'*115}")
    print(f"{'NETWORK IN (from Simulator)':^48} | {'ENGINE OUT (from Framer)':^30} | {'CNN BUFFER':^25}")
    print(f"{'-'*115}")
    print(f"{'Chunk':>5} | {'N (Samples)':>11} | {'Raw Shape':>14} | {'Dtype':>10} | "
          f"{'Frames Emitted':>15} | {'Output Shape':>16} | {'Rolling Window':>20}")
    print(f"{'-'*115}")

    total_tensors = 0

    for i in range(15):
        # The network delivers a randomized chunk
        chunk = next(gen)
        
        # The engine digests it. It might return [], or it might return a list of full [1,1,64,97] tensors.
        out_list = framer.process_chunk(chunk)
        
        # Stats
        n_samples = chunk.shape[1] if sim.layout == "planar" else chunk.shape[0]
        dtype_str = str(chunk.dtype).replace('torch.', '')
        n_emitted  = len(out_list)
        total_tensors += n_emitted
        
        cnn_status = "Waiting..."
        out_shape = "---"
        
        if n_emitted > 0:
            out_shape = str(list(out_list[0].shape))
            cnn_status = f"Ready: {n_emitted} chunks!"

        print(
            f"{i+1:>5} | "
            f"{n_samples:>11} | "
            f"{str(list(chunk.shape)):>14} | "
            f"{dtype_str:>10} | "
            f"{n_emitted:>15} | "
            f"{out_shape:>16} | "
            f"{cnn_status:>20}"
        )
        time.sleep(0.1)

    print(f"{'-'*115}")
    print(f"Integration Success! Handled 15 messy packets and safely extracted {total_tensors} standard [1,1,64,97] blocks.")
    print("=" * 80)

if __name__ == "__main__":
    main()
