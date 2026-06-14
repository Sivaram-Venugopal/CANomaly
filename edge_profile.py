import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import time
import tracemalloc

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"

# Features to use
FEATURES = ['delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency']

# PyTorch Autoencoder definition
class AnomalyAutoencoder(nn.Module):
    def __init__(self):
        super(AnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

def get_if_operations(clf):
    # For Isolation Forest, each tree requires comparison operations.
    # We estimate operations as n_estimators * average max depth of trees.
    try:
        depths = [e.tree_.max_depth for e in clf.estimators_]
        avg_depth = np.mean(depths)
        n_trees = len(clf.estimators_)
        return int(n_trees * avg_depth)
    except Exception:
        # Fallback if scikit-learn structure differs
        return 100 * 15

def get_ae_operations():
    # Architecture: Input(4) -> Dense(8) -> Dense(4) -> Dense(8) -> Output(4)
    # Layer 1: Linear(4, 8) -> 4*8 mult + 8 add = 40 ops
    # Layer 2: Linear(8, 4) -> 8*4 mult + 4 add = 36 ops
    # Layer 3: Linear(4, 8) -> 4*8 mult + 8 add = 40 ops
    # Layer 4: Linear(8, 4) -> 8*4 mult + 4 add = 36 ops
    # Plus ReLUs: 8 + 4 + 8 = 20 comparison operations
    # Total = 40 + 36 + 40 + 36 + 20 = 172 operations
    return 172

def main():
    print("Loading models and scaler...")
    # Load Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    # Load Isolation Forest
    if_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    clf = joblib.load(if_path)
    
    # Load Autoencoder
    ae_path = os.path.join(MODELS_DIR, "autoencoder.pt")
    model = AnomalyAutoencoder()
    model.load_state_dict(torch.load(ae_path))
    model.eval()
    
    # Load Threshold
    threshold_path = os.path.join(MODELS_DIR, "threshold.txt")
    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())
        
    print("Loading test data...")
    df = pd.read_csv(FEATURES_CSV, nrows=10000)
    X = df[FEATURES].values
    X_scaled = scaler.transform(X)
    
    # ==========================================
    # Profile Model A: Isolation Forest
    # ==========================================
    print("\nProfiling Isolation Forest...")
    # Reset and start tracemalloc
    tracemalloc.stop()
    tracemalloc.start()
    
    t0 = time.time()
    predictions_if = []
    # Classify one frame at a time
    for i in range(10000):
        frame = X_scaled[i].reshape(1, -1)
        pred = clf.predict(frame)
        predictions_if.append(pred)
        
    t_if = time.time() - t0
    current, peak_if = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    avg_latency_if_us = (t_if / 10000) * 1e6
    peak_mem_if_kb = peak_if / 1024
    throughput_if = 10000 / t_if
    
    # ==========================================
    # Profile Model B: Autoencoder
    # ==========================================
    print("Profiling Autoencoder...")
    tracemalloc.start()
    
    t0 = time.time()
    predictions_ae = []
    # Classify one frame at a time
    with torch.no_grad():
        for i in range(10000):
            frame = torch.FloatTensor(X_scaled[i]).reshape(1, -1)
            output = model(frame)
            mse = torch.mean((frame - output) ** 2).item()
            pred = 1 if mse > threshold else 0
            predictions_ae.append(pred)
            
    t_ae = time.time() - t0
    current, peak_ae = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    avg_latency_ae_us = (t_ae / 10000) * 1e6
    peak_mem_ae_kb = peak_ae / 1024
    throughput_ae = 10000 / t_ae
    
    # ==========================================
    # Print Performance Table
    # ==========================================
    print("\n=================================================================================")
    print(f"{'Model':<20} | {'Avg Latency (us)':<18} | {'Peak Memory (KB)':<18} | {'Throughput (fps)':<18}")
    print("---------------------------------------------------------------------------------")
    print(f"{'Isolation Forest':<20} | {avg_latency_if_us:<18.2f} | {peak_mem_if_kb:<18.2f} | {throughput_if:<18.2f}")
    print(f"{'Autoencoder':<20} | {avg_latency_ae_us:<18.2f} | {peak_mem_ae_kb:<18.2f} | {throughput_ae:<18.2f}")
    print("=================================================================================\n")
    
    # ==========================================
    # MIPS Estimation
    # ==========================================
    # MIPS = (Operations / Latency_in_seconds) / 1,000,000
    ops_if = get_if_operations(clf)
    mips_if = (ops_if / (avg_latency_if_us / 1e6)) / 1e6
    
    ops_ae = get_ae_operations()
    mips_ae = (ops_ae / (avg_latency_ae_us / 1e6)) / 1e6
    
    print(f"STM32F4 @ 168MHz has ~21 MIPS (Million Instructions Per Second).")
    print(f"Estimated operations per prediction:")
    print(f"  - Isolation Forest: {ops_if} comparisons")
    print(f"  - Autoencoder: {ops_ae} multiply-accumulates & comparisons")
    print(f"Our model requires:")
    print(f"  - Isolation Forest: {mips_if:.4f} MIPS (estimated)")
    print(f"  - Autoencoder: {mips_ae:.4f} MIPS (estimated)")
    print(f"\nSTEP 3 Completed successfully!")

if __name__ == "__main__":
    main()
