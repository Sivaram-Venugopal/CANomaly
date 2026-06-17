import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import json
import time

# Import TwoStageDetector
from two_stage_detector import TwoStageDetector

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
OUTPUTS_DIR = "D:/Tata Innovent/CANomaly/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 4 features for the Autoencoder
FEATURES_4 = ['delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency']
# All 5 features
FEATURES_5 = ['delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency', 'physical_plausibility_score']

# Autoencoder with input size 4
class AnomalyAutoencoder4(nn.Module):
    def __init__(self):
        super(AnomalyAutoencoder4, self).__init__()
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

def evaluate_subsets(y_true, y_pred, attack_types):
    attacks = ['DoS', 'Fuzzy', 'Gear', 'RPM']
    results = {}
    for attack in attacks:
        idx = (attack_types == 'Normal') | (attack_types == attack)
        y_true_sub = y_true[idx]
        y_pred_sub = y_pred[idx]
        if len(y_true_sub) > 0 and (y_true_sub == 1).sum() > 0:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true_sub, y_pred_sub, average='binary', pos_label=1, zero_division=0
            )
            results[attack] = {"precision": float(p), "recall": float(r), "f1_score": float(f1)}
        else:
            results[attack] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
            
    # Overall (binary class 1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    results["Overall"] = {"precision": float(p), "recall": float(r), "f1_score": float(f1)}
    
    # Normal treated as its own class (pos_label=0)
    p_norm, r_norm, f1_norm, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=0, zero_division=0
    )
    results["NormalClass"] = {"precision": float(p_norm), "recall": float(r_norm), "f1_score": float(f1_norm)}
    
    return results

def main():
    print("Loading features.csv...")
    df = pd.read_csv(FEATURES_CSV)
    
    # Stratified Train/Test split (same as train.py)
    df_train, df_test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['Label'].values
    )
    
    print("\nTraining local 4-feature Autoencoder for Stage 2...")
    # Get Normal training data
    df_train_normal = df_train[df_train['Label'] == 0]
    X_train_normal_4 = df_train_normal[FEATURES_4].values
    
    scaler_4 = StandardScaler()
    X_train_normal_scaled_4 = scaler_4.fit_transform(X_train_normal_4)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_normal_scaled_4))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    ae_4 = AnomalyAutoencoder4()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_4.parameters(), lr=0.001)
    
    ae_4.train()
    for epoch in range(15):
        for batch in train_loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            outputs = ae_4(x_batch)
            loss = criterion(outputs, x_batch)
            loss.backward()
            optimizer.step()
            
    ae_4.eval()
    print("Stage 2 Autoencoder training completed.")
    
    # Compute normal training MSE to determine percentiles
    with torch.no_grad():
        X_train_normal_tensor = torch.FloatTensor(X_train_normal_scaled_4)
        recon_train = ae_4(X_train_normal_tensor)
        mse_train_normal = torch.mean((X_train_normal_tensor - recon_train) ** 2, dim=1).numpy()
        
    threshold_ae_95 = np.percentile(mse_train_normal, 95)
    
    # ----------------------------------------------------
    # STEP 2: Evaluate the Two-Stage Detector
    # ----------------------------------------------------
    print("\nEvaluating Two-Stage Detector (Global 95th Percentile)...")
    detector = TwoStageDetector(
        autoencoder=ae_4, 
        scaler=scaler_4, 
        mse_threshold=threshold_ae_95, 
        plausibility_idx=4
    )
    
    X_test_5 = df_test[FEATURES_5].values
    y_test = df_test['Label'].values
    attack_types_test = df_test['Attack_Type'].values
    can_ids_test = df_test['CAN_ID'].values
    
    y_pred_two_stage = []
    stage_breakdown = {
        "Gear": {"physical_gate": 0, "statistical": 0, "missed": 0},
        "RPM": {"physical_gate": 0, "statistical": 0, "missed": 0}
    }
    
    for i in range(len(X_test_5)):
        feat = X_test_5[i]
        res = detector.predict(feat)
        y_pred_two_stage.append(1 if res["is_anomaly"] else 0)
        
        attack = attack_types_test[i]
        if attack in ['Gear', 'RPM']:
            if res["is_anomaly"]:
                stage_breakdown[attack][res["stage"]] += 1
            else:
                stage_breakdown[attack]["missed"] += 1
                
    y_pred_two_stage = np.array(y_pred_two_stage)
    
    # Print Stage Breakdown
    print("\n--- Detection Stage Breakdown ---")
    for attack in ["Gear", "RPM"]:
        total = sum(stage_breakdown[attack].values())
        print(f"{attack} Spoof Detections (Total={total:,}):")
        for stage, count in stage_breakdown[attack].items():
            pct = (count / total) * 100 if total > 0 else 0
            print(f"  {stage:<15} : {count:<6d} ({pct:.2f}%)")
            
    metrics_two_stage = evaluate_subsets(y_test, y_pred_two_stage, attack_types_test)
    
    # ----------------------------------------------------
    # STEP 3: Precision-Recall Curve Sweep for Stage 2 (DoS, Fuzzy)
    # ----------------------------------------------------
    print("\nSweeping MSE Threshold for Stage 2 (DoS & Fuzzy)...")
    # Filter test set for DoS, Fuzzy, and Normal only
    idx_stage2 = (attack_types_test == 'Normal') | (attack_types_test == 'DoS') | (attack_types_test == 'Fuzzy')
    X_test_stage2 = df_test.loc[idx_stage2, FEATURES_4].values
    y_test_stage2 = y_test[idx_stage2]
    
    # Scale features
    X_test_stage2_scaled = scaler_4.transform(X_test_stage2)
    with torch.no_grad():
        X_test_stage2_tensor = torch.FloatTensor(X_test_stage2_scaled)
        recon_stage2 = ae_4(X_test_stage2_tensor)
        mse_stage2 = torch.mean((X_test_stage2_tensor - recon_stage2) ** 2, dim=1).numpy()
        
    percentiles = np.linspace(50, 99.9, 50)
    precisions = []
    recalls = []
    f1s = []
    
    for p in percentiles:
        thresh = np.percentile(mse_train_normal, p)
        y_pred_p = (mse_stage2 > thresh).astype(int)
        precision_p, recall_p, f1_p, _ = precision_recall_fscore_support(
            y_test_stage2, y_pred_p, average='binary', pos_label=1, zero_division=0
        )
        precisions.append(precision_p)
        recalls.append(recall_p)
        f1s.append(f1_p)
        
    # Plot PR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='#1f77b4', lw=2.5, marker='o', markersize=3, label='Stage 2 (DoS/Fuzzy) PR Curve')
    
    # Annotate specific percentiles
    annotate_p = [50, 80, 90, 95, 98, 99, 99.9]
    for p in annotate_p:
        idx = np.argmin(np.abs(percentiles - p))
        plt.annotate(f"{p}%", 
                     xy=(recalls[idx], precisions[idx]),
                     xytext=(10, -5),
                     textcoords='offset points',
                     fontsize=9, fontweight='bold', color='red')
                     
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve for Stage 2 (Statistical Anomaly Detection)', fontsize=13, fontweight='bold', pad=15)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(OUTPUTS_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Saved Precision-Recall Curve to {pr_path}")
    
    # ----------------------------------------------------
    # STEP 4: Class-Conditional Thresholds for Stage 2
    # ----------------------------------------------------
    print("\nEvaluating Class-Conditional Thresholds...")
    threshold_ae_90 = np.percentile(mse_train_normal, 90)
    threshold_ae_98 = np.percentile(mse_train_normal, 98)
    
    # Tier A critical CAN IDs in KAIST
    critical_ids = {'0316', '043f', '0002', '00a0', '00a1', '0350', '0370'}
    
    y_pred_conditional = []
    for i in range(len(X_test_5)):
        feat = X_test_5[i]
        cid = can_ids_test[i]
        
        # Stage 1 check
        if feat[4] >= 0.5:
            y_pred_conditional.append(1)
            continue
            
        # Stage 2 check
        features_for_mse = np.delete(feat, 4)
        features_scaled = scaler_4.transform(features_for_mse.reshape(1, -1))[0]
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            recon = ae_4(x_tensor).squeeze(0).numpy()
            
        mse = ((recon - features_scaled) ** 2).mean()
        
        # Decide threshold based on CAN ID criticality tier
        thresh = threshold_ae_90 if cid in critical_ids else threshold_ae_98
        y_pred_conditional.append(1 if mse > thresh else 0)
        
    y_pred_conditional = np.array(y_pred_conditional)
    metrics_conditional = evaluate_subsets(y_test, y_pred_conditional, attack_types_test)
    
    # ----------------------------------------------------
    # STEP 5: Regenerate the Before/After Table
    # ----------------------------------------------------
    # Static Global 5-Feature single-stage baseline F1 scores (from task-750 run)
    f1_single_stage = {
        "DoS": 0.6512,
        "Fuzzy": 0.0016,
        "Gear": 0.6078,
        "RPM": 0.6273,
        "NormalClass": 0.9562
    }
    
    # F1 scores using class-conditional two-stage detector
    f1_two_stage = {
        "DoS": metrics_conditional["DoS"]["f1_score"],
        "Fuzzy": metrics_conditional["Fuzzy"]["f1_score"],
        "Gear": metrics_conditional["Gear"]["f1_score"],
        "RPM": metrics_conditional["RPM"]["f1_score"],
        "NormalClass": metrics_conditional["NormalClass"]["f1_score"]
    }
    
    # Comparative printout
    print("\n==================================================")
    print(" COMPARATIVE RESULTS: SINGLE-STAGE VS TWO-STAGE")
    print("==================================================")
    print(f"{'Attack Type':<15} | {'Single F1 (5-feat)':<20} | {'Two-Stage F1':<15} | {'Improvement':<12}")
    print(f"--------------------------------------------------")
    
    comparison_records = {}
    for k in ["DoS", "Fuzzy", "Gear", "RPM", "NormalClass"]:
        single = f1_single_stage[k]
        two = f1_two_stage[k]
        diff = two - single
        print(f"{k:<15} | {single:<20.4f} | {two:<15.4f} | {diff:<+12.4f}")
        comparison_records[k] = {
            "single_stage_f1": single,
            "two_stage_f1": two,
            "improvement": diff
        }
    print("==================================================")
    
    # Save JSON results
    json_path = os.path.join(OUTPUTS_DIR, "two_stage_results.json")
    with open(json_path, "w") as f:
        json.dump(comparison_records, f, indent=4)
    print(f"Saved numerical comparison to {json_path}")
    
    # Plot Comparison Bar Chart
    categories = ['DoS', 'Fuzzy', 'Gear Spoof', 'RPM Spoof', 'Normal']
    vals_single = [f1_single_stage["DoS"], f1_single_stage["Fuzzy"], f1_single_stage["Gear"], f1_single_stage["RPM"], f1_single_stage["NormalClass"]]
    vals_two = [f1_two_stage["DoS"], f1_two_stage["Fuzzy"], f1_two_stage["Gear"], f1_two_stage["RPM"], f1_two_stage["NormalClass"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, vals_single, width, label='Single-Stage Baseline', color='#ff7f0e', alpha=0.85)
    ax.bar(x + width/2, vals_two, width, label='Two-Stage (Class-Conditional)', color='#1f77b4', alpha=0.85)
    
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Single-Stage vs. Two-Stage Detection Architecture F1-Score', fontsize=14, fontweight='bold', pad=15)
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
    comp_path = os.path.join(OUTPUTS_DIR, "two_stage_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved comparison bar chart to {comp_path}")
    print("\nTwo-Stage evaluation completed successfully!")

if __name__ == '__main__':
    main()
