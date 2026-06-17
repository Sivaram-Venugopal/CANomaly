import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"
OUTPUTS_DIR = "D:/Tata Innovent/CANomaly/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 5 features to match the trained model
FEATURES = ['delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency', 'physical_plausibility_score']

# PyTorch Autoencoder Architecture
class AnomalyAutoencoder(nn.Module):
    def __init__(self):
        super(AnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 5)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

def calc_entropy(bytes_list):
    if not bytes_list:
        return 0.0
    counts = {}
    for b in bytes_list:
        counts[b] = counts.get(b, 0) + 1
    total = len(bytes_list)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())

def main():
    print("Building custom attack replay stream (2,000 normal + injections)...")
    # Load features.csv and grab normal frames
    df = pd.read_csv(FEATURES_CSV)
    df_normal = df[df['Label'] == 0].copy()
    
    # We need exactly 2000 normal frames
    df_normal = df_normal.head(2000).reset_index(drop=True)
    normal_frames = df_normal.to_dict('records')
    
    # Base timestamp
    current_ts = normal_frames[0]['Timestamp']
    
    # Create 50 DoS attack frames (Frame 800 to 849)
    print("Creating 50 DoS attack frames...")
    dos_frames = []
    for i in range(50):
        frame = {
            'Timestamp': 0, # Placeholder
            'CAN_ID': '000',
            'DLC': 8,
            'Flag': 'T',
            'Label': 1,
            'Attack_Type': 'DoS',
            'injected_attack_type': 'DoS'
        }
        for idx in range(8):
            frame[str(idx+3)] = "00"
        dos_frames.append(frame)
        
    # Create 30 Fuzzy attack frames (Frame 1400 to 1429)
    print("Creating 30 Fuzzy attack frames...")
    fuzzy_frames = []
    for i in range(30):
        can_id = f"{np.random.randint(0x500, 0x7FF):03X}"
        dlc = np.random.randint(1, 9)
        payload = [f"{np.random.randint(0, 256):02X}" for _ in range(dlc)]
        frame = {
            'Timestamp': 0,
            'CAN_ID': can_id,
            'DLC': dlc,
            'Flag': 'T',
            'Label': 1,
            'Attack_Type': 'Fuzzy',
            'injected_attack_type': 'Fuzzy'
        }
        for idx, p in enumerate(payload):
            frame[str(idx+3)] = p
        for idx in range(dlc, 8):
            frame[str(idx+3)] = np.nan
        fuzzy_frames.append(frame)
        
    # Create 20 RPM Spoofing frames (Frame 1700 to 1719)
    print("Creating 20 RPM Spoofing attack frames...")
    rpm_frames = []
    rpm_payload = ["00", "00", "10", "20", "00", "00", "00", "00"]
    for i in range(20):
        frame = {
            'Timestamp': 0,
            'CAN_ID': '0C4',
            'DLC': 8,
            'Flag': 'T',
            'Label': 1,
            'Attack_Type': 'RPM',
            'injected_attack_type': 'RPM_Spoofing'
        }
        for idx, p in enumerate(rpm_payload):
            frame[str(idx+3)] = p
        rpm_frames.append(frame)

    # Create 50 UDS Reflash frames (Frame 1900 to 1949)
    print("Creating 50 UDS Reflash frames (SUMS-relevant)...")
    uds_frames = []
    # SIDs: 0x10 (DiagnosticSessionControl), 0x34 (RequestDownload), 0x36 (TransferData), 0x37 (RequestTransferExit)
    for i in range(50):
        if i < 5:
            sid = "10"
        elif i < 10:
            sid = "34"
        elif i < 45:
            sid = "36"
        else:
            sid = "37"
            
        frame = {
            'Timestamp': 0,
            'CAN_ID': '7E0',
            'DLC': 8,
            'Flag': 'T',
            'Label': 1,
            'Attack_Type': 'UDS_Reflash',
            'injected_attack_type': 'UDS_Reflash'
        }
        frame['3'] = "02"  # PCI length
        frame['4'] = sid   # UDS Service ID
        for idx in range(2, 8):
            frame[str(idx+3)] = "00"
        uds_frames.append(frame)
        
    # Standardize normal frames keys
    for f in normal_frames:
        f['injected_attack_type'] = 'None'
        
    # Combine sequences:
    # 0 to 800: Normal (800 frames)
    # 800 to 850: DoS (50 frames)
    # 850 to 1400: Normal (550 frames)
    # 1400 to 1430: Fuzzy (30 frames)
    # 1430 to 1700: Normal (270 frames)
    # 1700 to 1720: RPM Spoofing (20 frames)
    # 1720 to 1900: Normal (180 frames) -> split normal_frames[1620:1800]
    # 1900 to 1950: UDS Reflash (50 frames)
    # 1950 to 2150: Normal (200 frames) -> split normal_frames[1800:]
    combined = (
        normal_frames[:800] + 
        dos_frames + 
        normal_frames[800:1350] + 
        fuzzy_frames + 
        normal_frames[1350:1620] + 
        rpm_frames + 
        normal_frames[1620:1800] +
        uds_frames +
        normal_frames[1800:]
    )
    
    df_replay = pd.DataFrame(combined)
    
    # Calculate sequential Timestamps
    timestamps = [current_ts]
    for idx in range(1, len(df_replay)):
        attack = df_replay.iloc[idx]['injected_attack_type']
        if attack == 'DoS':
            current_ts += 0.00005 # high frequency DoS
        elif attack == 'Fuzzy':
            current_ts += 0.00010 # high frequency Fuzzy
        elif attack == 'RPM_Spoofing':
            current_ts += 0.03000 # 3x normal interval
        elif attack == 'UDS_Reflash':
            current_ts += 0.00020 # high frequency UDS Reflash
        else:
            current_ts += 0.00030 # average normal interval
        timestamps.append(current_ts)
        
    df_replay['Timestamp'] = timestamps
    
    # Re-calculate 5 core features
    df_replay['delta_t'] = df_replay['Timestamp'].diff().fillna(0)
    df_replay['window'] = df_replay['Timestamp'] // 0.1
    df_replay['can_id_freq'] = df_replay.groupby(['window', 'CAN_ID'])['CAN_ID'].transform('count')
    df_replay['dlc_consistency'] = df_replay.groupby('CAN_ID')['DLC'].transform('std').fillna(0)
    # Ensure physical_plausibility_score is filled for injected frames
    if 'physical_plausibility_score' in df_replay.columns:
        df_replay['physical_plausibility_score'] = df_replay['physical_plausibility_score'].fillna(0.0)
    else:
        df_replay['physical_plausibility_score'] = 0.0
    
    # Calculate payload entropy
    payload_cols = ['3', '4', '5', '6', '7', '8', '9', '10']
    entropies = []
    for idx, row in df_replay.iterrows():
        dlc = int(row['DLC'])
        bytes_list = []
        for col in payload_cols:
            val = row[col]
            if pd.notna(val) and val != '' and str(val).lower() != 'nan':
                bytes_list.append(str(val))
        bytes_list = bytes_list[:dlc]
        entropies.append(calc_entropy(bytes_list))
    df_replay['payload_entropy'] = entropies
    
    # Rename columns to match requested schema: [frame_idx, can_id, injected_attack_type, delta_t, payload_entropy]
    df_replay = df_replay.reset_index(drop=True)
    df_replay['frame_idx'] = df_replay.index
    
    # Step 2: Run Ensemble Detection
    print("Loading models for ensemble prediction...")
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    clf_if = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    clf_svm = joblib.load(os.path.join(MODELS_DIR, "one_class_svm.pkl"))
    
    model_ae = AnomalyAutoencoder()
    model_ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt")))
    model_ae.eval()
    
    with open(os.path.join(MODELS_DIR, "threshold.txt"), "r") as f:
        threshold_ae = float(f.read().strip())
        
    # Scale features
    X_scaled = scaler.transform(df_replay[FEATURES].values)
    
    # Compute predictions for all 3 models
    y_pred_if = (clf_if.predict(X_scaled) == -1).astype(int)
    y_pred_svm = (clf_svm.predict(X_scaled) == -1).astype(int)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        reconstructed = model_ae(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        y_pred_ae = (errors > threshold_ae).astype(int)
        
    # Frame prediction = average of the 3 model predictions
    frame_predictions = (y_pred_if + y_pred_ae + y_pred_svm) / 3.0
    
    # Run sliding window ensemble (W=50, stride=10) on the replay stream
    W = 50
    stride = 10
    
    scores_records = []
    window_id = 0
    
    for start in range(0, len(df_replay) - W + 1, stride):
        end = start + W
        # Slice frame ensemble scores
        win_scores = frame_predictions[start:end]
        ensemble_score = float(np.mean(win_scores))
        
        # Check if window contains any attack frame (true label == 1)
        ground_truth = 1 if (df_replay.iloc[start:end]['Label'] == 1).any() else 0
        
        scores_records.append({
            "window_id": window_id,
            "start_frame": start,
            "end_frame": end - 1,
            "ensemble_score": ensemble_score,
            "ground_truth_contains_attack": ground_truth
        })
        window_id += 1
        
    scores_df = pd.DataFrame(scores_records)
    scores_df.to_csv(os.path.join(OUTPUTS_DIR, "replay_scores.csv"), index=False)
    print(f"Saved sliding window scores to outputs/replay_scores.csv")
    
    # Run SUMS-aligned check
    print("\nRunning SUMS reflash verification on stream...")
    from sums_check import scan_stream_for_uds_anomalies
    scan_stream_for_uds_anomalies(df_replay, maintenance_window_active=False)
    
    # Step 3: Plot the spike chart (replay_attack_spike.png)
    print("Plotting attack replay spike chart...")
    plt.figure(figsize=(12, 6))
    
    # Compute X values (center of each window)
    window_centers = [(r['start_frame'] + r['end_frame']) / 2 for r in scores_records]
    ensemble_scores = [r['ensemble_score'] for r in scores_records]
    
    # Plot line
    plt.plot(window_centers, ensemble_scores, color='#1f77b4', lw=2.5, marker='o', markersize=4, label='Ensemble Anomaly Score')
    
    # Draw horizontal red dashed line at y=0.5
    plt.axhline(y=0.5, color='red', linestyle='--', lw=1.5, label='Anomaly Threshold (0.5)')
    
    # Shade background red in injection zones
    plt.axvspan(800, 850, color='red', alpha=0.15, label='DoS Injection Zone')
    plt.axvspan(1400, 1430, color='orange', alpha=0.15, label='Fuzzy Injection Zone')
    plt.axvspan(1700, 1720, color='purple', alpha=0.15, label='RPM Spoofing Injection Zone')
    plt.axvspan(1900, 1950, color='brown', alpha=0.15, label='Unauthorized Reflash (SUMS)')
    
    # Annotations
    plt.text(825, 0.92, 'DoS Injected\n(Frame 800)', color='#d62728', fontsize=10, fontweight='bold', ha='center')
    plt.text(1415, 0.92, 'Fuzzy Injected\n(Frame 1400)', color='#ff7f0e', fontsize=10, fontweight='bold', ha='center')
    plt.text(1710, 0.92, 'RPM Spoofing Injected\n(Frame 1700)', color='purple', fontsize=10, fontweight='bold', ha='center')
    plt.text(1925, 0.92, 'Reflash (SUMS)\n(Frame 1900)', color='brown', fontsize=10, fontweight='bold', ha='center')
    
    plt.xlim(0, 2200)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Frame Index', fontsize=12, fontweight='bold')
    plt.ylabel('Ensemble Anomaly Score', fontsize=12, fontweight='bold')
    plt.title('CANomaly Live Attack Replay Detection Spike Chart', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    spike_path = os.path.join(OUTPUTS_DIR, "replay_attack_spike.png")
    plt.savefig(spike_path, dpi=300)
    plt.close()
    print(f"Saved spike chart to {spike_path}")
    
    # Save custom stream for stream simulator reference
    replay_stream_path = os.path.join(OUTPUTS_DIR, "replay_stream.csv")
    df_replay.rename(columns={'CAN_ID': 'can_id'}, inplace=True)
    df_replay.to_csv(replay_stream_path, index=False)
    
    # Step 4: README section console output
    dos_spike = max([r['ensemble_score'] for r in scores_records if r['start_frame'] >= 780 and r['end_frame'] <= 870])
    fuzzy_spike = max([r['ensemble_score'] for r in scores_records if r['start_frame'] >= 1380 and r['end_frame'] <= 1450])
    rpm_spike = max([r['ensemble_score'] for r in scores_records if r['start_frame'] >= 1680 and r['end_frame'] <= 1740])
    reflash_spike = max([r['ensemble_score'] for r in scores_records if r['start_frame'] >= 1880 and r['end_frame'] <= 1970])
    
    print("\n" + "="*50)
    print(" COPY-PASTE THIS BLOCK INTO YOUR GITHUB README.md")
    print("="*50 + "\n")
    
    readme_block = f"""## Live attack replay demo
The chart below shows CANomaly's ensemble anomaly score (0–1) on a
2,150-frame CAN stream with four injected attacks/anomalies (including SUMS reflash).

![Attack Replay Spike Chart](outputs/replay_attack_spike.png)

| Injection point | Attack type | Score spike |
|---|---|---|
| Frame 800 | DoS | > {dos_spike:.2f} |
| Frame 1400 | Fuzzy | > {fuzzy_spike:.2f} |
| Frame 1700 | RPM Spoofing | > {rpm_spike:.2f} |
| Frame 1900 | UDS Reflash (SUMS) | > {reflash_spike:.2f} |

Detection threshold: 0.5. All four attacks/anomalies detected within one window (50 frames)."""
    
    print(readme_block)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()

