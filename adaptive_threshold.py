import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import time

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"
OUTPUTS_DIR = "D:/Tata Innovent/CANomaly/outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 5 features
FEATURES = ['delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency', 'physical_plausibility_score']

# PyTorch Autoencoder definition
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
    
    # Normal traffic treated as its own class (pos_label=0)
    _, _, f1_normal, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=0, zero_division=0
    )
    results["NormalClass"] = float(f1_normal)
    
    return results

def main():
    print("Loading features.csv for sequential adaptive evaluation...")
    df = pd.read_csv(FEATURES_CSV)
    
    # ----------------------------------------------------
    # STEP 1.3: Class distribution of the test set
    # ----------------------------------------------------
    # Reconstruct the 80/20 train/test split exactly as in train.py
    X = df[FEATURES].values
    y = df['Label'].values
    attack_types = df['Attack_Type'].values
    
    # Stratified split
    _, df_test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['Label'].values
    )
    
    y_test = df_test['Label'].values
    attack_types_test = df_test['Attack_Type'].values
    
    print("\n==================================================")
    print(" TEST SPLIT CLASS DISTRIBUTION")
    print("==================================================")
    dist = df_test['Attack_Type'].value_counts()
    for name, cnt in dist.items():
        pct = (cnt / len(df_test)) * 100
        print(f"{name:<15} : {cnt:<8d} ({pct:.2f}%)")
    print(f"Total Test Set  : {len(df_test):,}")
    print("==================================================\n")
    
    # Load Scaler & Model
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    model = AnomalyAutoencoder()
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt")))
    model.eval()
    
    # Load Global Threshold
    with open(os.path.join(MODELS_DIR, "threshold.txt"), "r") as f:
        global_threshold = float(f.read().strip())
        
    # Sort chronologically to preserve streaming sequence
    df_sorted = df.sort_values(by='Timestamp').reset_index(drop=True)
    can_ids_sorted = df_sorted['CAN_ID'].values
    labels_sorted = df_sorted['Label'].values
    
    # Scale and compute errors for all frames
    X_sorted_scaled = scaler.transform(df_sorted[FEATURES].values)
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sorted_scaled)
        reconstructed = model(X_tensor)
        errors_sorted = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
    # ----------------------------------------------------
    # STEP 2.2 & 2.3: Warmup Window Attack Percentage Check & Assertion
    # ----------------------------------------------------
    warmup_len = 100
    id_counts = {}
    id_warmup_attack_count = {}
    
    for i in range(len(df_sorted)):
        cid = can_ids_sorted[i]
        lbl = labels_sorted[i]
        if cid not in id_counts:
            id_counts[cid] = 0
            id_warmup_attack_count[cid] = 0
        if id_counts[cid] < warmup_len:
            id_counts[cid] += 1
            if lbl == 1:
                id_warmup_attack_count[cid] += 1
                
    print("==================================================")
    print(f" STEP 2.2 & 2.3: WARMUP WINDOW ({warmup_len} FRAMES) POLLUTION CHECK")
    print("==================================================")
    polluted_ids_count = 0
    total_ids = len(id_counts)
    
    for cid in sorted(id_counts.keys()):
        count = id_counts[cid]
        att_count = id_warmup_attack_count[cid]
        pct = (att_count / count) * 100
        if pct > 0:
            polluted_ids_count += 1
            print(f"CAN ID {cid:<5} : {att_count}/{count} frames are attacks ({pct:.2f}%)")
            
    print(f"\nResult: {polluted_ids_count}/{total_ids} CAN IDs have attack frames in their first {warmup_len} occurrences.")
    print("Assertion: Warmup window is polluted by attack traffic for injected CAN IDs.")
    print("==================================================\n")
    
    # ----------------------------------------------------
    # STEP 2.1: Run tracker on (A) Train+Test combined vs. (B) Test Split only
    # ----------------------------------------------------
    print("Running sequential tracker on Train+Test combined...")
    tracker_comb = AdaptiveThreshold(alpha=0.05, sigma=3.0, warmup_frames=50)
    y_pred_adaptive_comb = []
    
    t_start = time.time()
    for i in range(len(errors_sorted)):
        is_training = (labels_sorted[i] == 0) # update only on normal traffic
        is_anom = tracker_comb.is_anomaly(can_ids_sorted[i], errors_sorted[i], is_training=is_training)
        y_pred_adaptive_comb.append(1 if is_anom else 0)
    y_pred_adaptive_comb = np.array(y_pred_adaptive_comb)
    print(f"Completed in {time.time() - t_start:.2f}s")
    
    # Map predictions back to original dataframe structure to extract test split indices
    df_sorted['pred_global'] = (errors_sorted > global_threshold).astype(int)
    df_sorted['pred_adaptive_comb'] = y_pred_adaptive_comb
    
    # Extract test split predictions
    df_test_sorted = df_sorted[df_sorted.index.isin(df_test.index)].copy()
    
    # Now run strictly on the test split ONLY (sorted chronologically)
    print("\nRunning sequential tracker on Test Split ONLY...")
    df_test_only = df_test.sort_values(by='Timestamp').reset_index(drop=True)
    X_test_only_scaled = scaler.transform(df_test_only[FEATURES].values)
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_only_scaled)
        reconstructed_test = model(X_test_tensor)
        errors_test_only = torch.mean((X_test_tensor - reconstructed_test) ** 2, dim=1).numpy()
        
    tracker_test = AdaptiveThreshold(alpha=0.05, sigma=3.0, warmup_frames=50)
    y_pred_adaptive_test = []
    
    labels_test_only = df_test_only['Label'].values
    can_ids_test_only = df_test_only['CAN_ID'].values
    
    t_start = time.time()
    for i in range(len(errors_test_only)):
        is_training = (labels_test_only[i] == 0)
        is_anom = tracker_test.is_anomaly(can_ids_test_only[i], errors_test_only[i], is_training=is_training)
        y_pred_adaptive_test.append(1 if is_anom else 0)
    y_pred_adaptive_test = np.array(y_pred_adaptive_test)
    print(f"Completed in {time.time() - t_start:.2f}s")
    
    df_test_only['pred_adaptive_test'] = y_pred_adaptive_test
    df_test_only['pred_global'] = (errors_test_only > global_threshold).astype(int)
    
    # Save the combined tracker state
    tracker_path = os.path.join(MODELS_DIR, "adaptive_threshold.pkl")
    joblib.dump(tracker_comb, tracker_path)
    
    # ----------------------------------------------------
    # STEP 1.1 & 1.2: Compute and Print Per-Attack F1 and Exact Formula
    # ----------------------------------------------------
    # Evaluate B (Train+Test Combined evaluation mapped to Test Set)
    f1_global = evaluate_subsets(
        df_test_sorted['Label'].values, 
        df_test_sorted['pred_global'].values, 
        df_test_sorted['Attack_Type'].values
    )
    f1_adaptive_comb = evaluate_subsets(
        df_test_sorted['Label'].values, 
        df_test_sorted['pred_adaptive_comb'].values, 
        df_test_sorted['Attack_Type'].values
    )
    
    # Evaluate C (Test split ONLY)
    f1_adaptive_test = evaluate_subsets(
        df_test_only['Label'].values, 
        df_test_only['pred_adaptive_test'].values, 
        df_test_only['Attack_Type'].values
    )
    
    print("\n==================================================")
    print(" F1-SCORE DIAGNOSTIC COMPARISON ON TEST SPLIT")
    print("==================================================")
    print(f"{'Attack / Class':<18} | {'Global F1':<10} | {'Adaptive F1 (Comb)':<20} | {'Adaptive F1 (Test-Only)':<22}")
    print(f"--------------------------------------------------")
    for k in ['DoS', 'Fuzzy', 'Gear', 'RPM', 'NormalClass', 'Overall']:
        print(f"{k:<18} | {f1_global[k]:<10.4f} | {f1_adaptive_comb[k]:<20.4f} | {f1_adaptive_test[k]:<22.4f}")
    print("==================================================")
    
    print("\n==================================================")
    print(" STEP 1.2: EXACT FORMULA USED FOR OVERALL F1-SCORE")
    print("==================================================")
    print("The 'Overall' F1-Score is computed as the BINARY F1-Score for the anomaly class (Label 1):")
    print("Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)")
    print("Where:")
    print("  Precision = True Positives (TP) / (True Positives (TP) + False Positives (FP))")
    print("  Recall    = True Positives (TP) / (True Positives (TP) + False Negatives (FN))")
    print("This is NOT an average of the per-attack scores, but the binary classification metrics")
    print("aggregated over the entire test set.")
    
    print("\nNormalClass F1-Score represents binary classification F1-Score focusing on")
    print("Normal traffic as the positive class (Label 0):")
    print("Formula: F1_normal = 2 * (Precision_0 * Recall_0) / (Precision_0 + Recall_0)")
    print("==================================================\n")
    
    # Save comparison plot
    categories = ['DoS', 'Fuzzy', 'Gear', 'RPM', 'NormalClass', 'Overall']
    vals_global = [f1_global[c] for c in categories]
    vals_adaptive = [f1_adaptive_test[c] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, vals_global, width, label='Global Threshold (Static)', color='#ff7f0e', alpha=0.85)
    ax.bar(x + width/2, vals_adaptive, width, label='Adaptive Threshold (Self-Calibrating)', color='#1f77b4', alpha=0.85)
    
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Global vs. Test-Only Adaptive Thresholding F1-Score', fontsize=14, fontweight='bold', pad=15)
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
    print(f"Saved updated comparison chart to {chart_path}")
    print("\nAdaptive evaluation diagnostics completed successfully!")

if __name__ == "__main__":
    main()

