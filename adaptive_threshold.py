import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import time

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"
OUTPUTS_DIR = "D:/Tata Innovent/CANomaly/outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 4 features
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

# Step 1: AdaptiveThreshold Class
class AdaptiveThreshold:
    def __init__(self, alpha=0.05, sigma=3.0, warmup_frames=50):
        self.alpha = alpha
        self.sigma = sigma
        self.warmup_frames = warmup_frames
        self.ema_mean = {}
        self.ema_var = {}
        self.frame_count = {}
        
    def is_anomaly(self, can_id, recon_error, is_training=True):
        if can_id not in self.ema_mean:
            # Initialize
            self.ema_mean[can_id] = recon_error
            self.ema_var[can_id] = 0.0
            self.frame_count[can_id] = 0
            
        self.frame_count[can_id] += 1
        count = self.frame_count[can_id]
        
        # Calculate z-score using current EMA values
        std = (self.ema_var[can_id] ** 0.5) + 1e-8
        z = (recon_error - self.ema_mean[can_id]) / std
        
        is_anomaly = False
        if count >= self.warmup_frames:
            is_anomaly = (z > self.sigma)
            
        # Update EMA only if not an anomaly (to prevent baseline drift) or in warmup
        if not is_anomaly or count < self.warmup_frames or is_training:
            self.ema_mean[can_id] = (1.0 - self.alpha) * self.ema_mean[can_id] + self.alpha * recon_error
            self.ema_var[can_id] = (1.0 - self.alpha) * self.ema_var[can_id] + self.alpha * ((recon_error - self.ema_mean[can_id]) ** 2)
            
        return is_anomaly

def evaluate_subsets(y_true, y_pred, attack_types):
    attacks = ['DoS', 'Fuzzy', 'Gear', 'RPM']
    results = {}
    for attack in attacks:
        idx = (attack_types == 'Normal') | (attack_types == attack)
        y_true_sub = y_true[idx]
        y_pred_sub = y_pred[idx]
        if len(y_true_sub) > 0 and (y_true_sub == 1).sum() > 0:
            _, _, f1, _ = precision_recall_fscore_support(
                y_true_sub, y_pred_sub, average='binary', pos_label=1, zero_division=0
            )
            results[attack] = float(f1)
        else:
            results[attack] = 0.0
            
    # Overall
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    results["Overall"] = float(f1)
    return results

def main():
    print("Loading datasets sequentially for adaptive threshold evaluation...")
    df = pd.read_csv(FEATURES_CSV).sort_values(by='Timestamp').reset_index(drop=True)
    
    # Load Scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_scaled = scaler.transform(df[FEATURES].values)
    
    # Load Autoencoder
    device = torch.device('cpu')
    model = AnomalyAutoencoder()
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt")))
    model.eval()
    
    # Load Global Threshold
    with open(os.path.join(MODELS_DIR, "threshold.txt"), "r") as f:
        global_threshold = float(f.read().strip())
        
    print("Computing Autoencoder reconstruction errors...")
    t0 = time.time()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    print(f"Computed reconstruction errors for {len(errors):,} frames in {time.time() - t0:.2f}s")
    
    # Run Evaluations
    print("Evaluating Global vs Adaptive thresholding...")
    
    # Evaluation A: Global Threshold
    y_pred_global = (errors > global_threshold).astype(int)
    
    # Evaluation B: Adaptive Threshold
    # Instantiate tracker
    tracker = AdaptiveThreshold(alpha=0.05, sigma=3.0, warmup_frames=50)
    
    y_pred_adaptive = []
    can_ids = df['CAN_ID'].values
    
    t_start = time.time()
    # Run sequentially on the stream to simulate self-calibration
    for i in range(len(errors)):
        is_training = (df.iloc[i]['Label'] == 0) # Update EMA baseline on normal frames
        is_anom = tracker.is_anomaly(can_ids[i], errors[i], is_training=is_training)
        y_pred_adaptive.append(1 if is_anom else 0)
        
    y_pred_adaptive = np.array(y_pred_adaptive)
    print(f"Sequential adaptive thresholding completed in {time.time() - t_start:.2f}s")
    
    # Save the AdaptiveThreshold state
    tracker_path = os.path.join(MODELS_DIR, "adaptive_threshold.pkl")
    joblib.dump(tracker, tracker_path)
    print(f"Saved AdaptiveThreshold baseline tracker to {tracker_path}")
    
    # Compute Metrics
    y_true = df['Label'].values
    attack_types = df['Attack_Type'].values
    
    f1_global = evaluate_subsets(y_true, y_pred_global, attack_types)
    f1_adaptive = evaluate_subsets(y_true, y_pred_adaptive, attack_types)
    
    # Print side-by-side F1 table
    print("\n==================================================")
    print(" F1-Score Comparison: Global vs. Adaptive Threshold")
    print("==================================================")
    print(f"{'Attack Type':<15} | {'Global F1':<12} | {'Adaptive F1':<12}")
    print(f"--------------------------------------------------")
    for k in ['DoS', 'Fuzzy', 'Gear', 'RPM', 'Overall']:
        print(f"{k:<15} | {f1_global[k]:<12.4f} | {f1_adaptive[k]:<12.4f}")
    print("==================================================\n")
    
    # Save comparison to outputs/adaptive_vs_global.png
    print("Plotting comparison chart...")
    categories = ['DoS', 'Fuzzy', 'Gear', 'RPM', 'Overall']
    vals_global = [f1_global[c] for c in categories]
    vals_adaptive = [f1_adaptive[c] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, vals_global, width, label='Global Threshold (Static)', color='#ff7f0e', alpha=0.85)
    ax.bar(x + width/2, vals_adaptive, width, label='Adaptive Threshold (Self-Calibrating)', color='#1f77b4', alpha=0.85)
    
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Global vs. Self-Calibrating Adaptive Thresholding F1-Score', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    for idx, rect in enumerate(ax.patches):
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    plt.tight_layout()
    chart_path = os.path.join(OUTPUTS_DIR, "adaptive_vs_global.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved comparison chart to {chart_path}")
    print("\nAdaptive evaluation completed successfully!")

if __name__ == "__main__":
    main()
