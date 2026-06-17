import os
import json
import torch
import torch.nn as nn
import numpy as np
from two_stage_detector import TwoStageDetector
from secure_logger import verify_chain

# Simple mock 4-input model for demo purposes
class MockAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        # Initialize weights to something non-zero
        with torch.no_grad():
            self.linear.weight.fill_(0.5)
            self.linear.bias.fill_(0.1)

    def forward(self, x):
        return self.linear(x)

def run_demo():
    log_path = "outputs/demo_event_log.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)

    print("--- STEP 1: Initializing Two-Stage Detector with Secure Logger ---")
    ae = MockAutoencoder()
    ae.eval()
    
    # Instantiate detector to write to demo log file
    detector = TwoStageDetector(
        autoencoder=ae,
        scaler=None,
        mse_threshold=0.01,
        plausibility_idx=4,
        log_path=log_path
    )

    print("\n--- STEP 2: Logging 10 detection events ---")
    # We simulate 10 CAN frames with varying features.
    # We alternate between physical violations (plausibility=1) and statistical features.
    for i in range(10):
        # Feature vector: [delta_t, can_id_freq, payload_entropy, dlc_consistency, physical_plausibility_score]
        if i % 2 == 0:
            # Stage 1 physical anomaly
            feat = np.array([0.0003, 15.0, 1.2, 0.0, 1.0])
        else:
            # Stage 2 statistical anomaly (force MSE high by using large inputs and a low threshold)
            feat = np.array([10.0, 100.0, 8.0, 5.0, 0.0])
            
        res = detector.predict(feat)
        print(f"Frame {i+1:2d} -> Anomaly: {res['is_anomaly']}, Stage: {res['stage']}, Confidence: {res['confidence']}")

    print("\n--- STEP 3: Verifying the integrity of the log file ---")
    is_valid = verify_chain(log_path)
    print(f"Log integrity check result: {is_valid} (Expected: True)")

    print("\n--- STEP 4: Simulating manual corruption / tampering ---")
    # Read the log file, corrupt one line, and write back
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"Original Line 5: {lines[4].strip()}")
    # Corrupt a value in the 5th log entry (index 4)
    entry = json.loads(lines[4])
    entry["confidence"] = "Low"  # Tampered! (was "High" because it was physical_gate)
    lines[4] = json.dumps(entry) + "\n"
    print(f"Tampered Line 5: {lines[4].strip()}")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\n--- STEP 5: Re-verifying the integrity of the tampered log file ---")
    is_valid_after_tamper = verify_chain(log_path)
    print(f"Log integrity check result post-tamper: {is_valid_after_tamper} (Expected: False)")

    # Cleanup
    if os.path.exists(log_path):
        os.remove(log_path)

if __name__ == "__main__":
    run_demo()
