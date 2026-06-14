import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
import time

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"
OUTPUTS_DIR = "D:/Tata Innovent/CANomaly/outputs"
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

def get_rule_based_predictions(df_test):
    # Rule-based baseline: anomaly if delta_t < 0.0001 (high frequency) OR dlc_consistency > 0.5 (variable DLC)
    # This represents standard rule-based automotive CAN filters
    pred_rule = ((df_test['delta_t'] < 0.0001) | (df_test['dlc_consistency'] > 0.5)).astype(int)
    return pred_rule.values

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
            
    # Overall
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    results["Overall"] = {"precision": float(p), "recall": float(r), "f1_score": float(f1)}
    
    return results

def main():
    print("Loading test data...")
    df = pd.read_csv(FEATURES_CSV)
    
    # Split using the exact same random state to reconstruct the same test set
    X = df[FEATURES].values
    y = df['Label'].values
    attack_types = df['Attack_Type'].values
    
    _, X_test, _, y_test, _, attack_types_test = train_test_split(
        X, y, attack_types, test_size=0.20, random_state=42, stratify=y
    )
    
    # Get test slice of df for rule-based evaluation (needs original column values)
    _, df_test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['Label'].values
    )
    
    # Load scaler and models
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    clf_if = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    
    model_ae = AnomalyAutoencoder()
    model_ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt")))
    model_ae.eval()
    
    with open(os.path.join(MODELS_DIR, "threshold.txt"), "r") as f:
        threshold_ae = float(f.read().strip())
        
    X_test_scaled = scaler.transform(X_test)
    
    # ==========================================
    # Get Predictions and Anomaly Scores
    # ==========================================
    print("Evaluating models...")
    
    # 1. Isolation Forest
    scores_if = -clf_if.score_samples(X_test_scaled) # Higher score = more anomalous
    y_pred_if = (clf_if.predict(X_test_scaled) == -1).astype(int)
    
    # 2. Autoencoder
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        reconstructed_test = model_ae(X_test_tensor)
        scores_ae = torch.mean((X_test_tensor - reconstructed_test) ** 2, dim=1).numpy()
        y_pred_ae = (scores_ae > threshold_ae).astype(int)
        
    # 3. Rule-based Baseline
    y_pred_rule = get_rule_based_predictions(df_test)
    
    # Evaluate each
    metrics_if = evaluate_subsets(y_test, y_pred_if, attack_types_test)
    metrics_ae = evaluate_subsets(y_test, y_pred_ae, attack_types_test)
    metrics_rule = evaluate_subsets(y_test, y_pred_rule, attack_types_test)
    
    # Save results to json
    results = {
        "IsolationForest": metrics_if,
        "Autoencoder": metrics_ae,
        "RuleBasedBaseline": metrics_rule
    }
    
    results_path = os.path.join(OUTPUTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved numerical metrics to {results_path}")
    
    # ==========================================
    # 1. Save comparison table as comparison.png
    # ==========================================
    print("Generating comparison chart...")
    categories = ['DoS', 'Fuzzy', 'Gear', 'RPM', 'Overall']
    f1_rule = [metrics_rule[c]['f1_score'] for c in categories]
    f1_if = [metrics_if[c]['f1_score'] for c in categories]
    f1_ae = [metrics_ae[c]['f1_score'] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, f1_rule, width, label='Rule-based Baseline', color='#d62728', alpha=0.85)
    ax.bar(x, f1_if, width, label='Isolation Forest (Model A)', color='#ff7f0e', alpha=0.85)
    ax.bar(x + width, f1_ae, width, label='Autoencoder (Model B)', color='#1f77b4', alpha=0.85)
    
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Detection F1-Score Comparison Across Attack Types', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add values on top of bars
    for idx, rect in enumerate(ax.patches):
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUTS_DIR, "comparison.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"Saved comparison chart to {comparison_path}")
    
    # ==========================================
    # 2. Save ROC curve plot as roc_curve.png
    # ==========================================
    print("Generating ROC curve plot...")
    fpr_if, tpr_if, _ = roc_curve(y_test, scores_if)
    auc_if = auc(fpr_if, tpr_if)
    
    fpr_ae, tpr_ae, _ = roc_curve(y_test, scores_ae)
    auc_ae = auc(fpr_ae, tpr_ae)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_if, tpr_if, color='#ff7f0e', lw=2.5, label=f'Isolation Forest (AUC = {auc_if:.4f})')
    plt.plot(fpr_ae, tpr_ae, color='#1f77b4', lw=2.5, label=f'Autoencoder (AUC = {auc_ae:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    roc_path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve plot to {roc_path}")
    
    # ==========================================
    # 3. Save confusion matrix heatmap as confusion_matrix.png
    # ==========================================
    print("Generating Confusion Matrix plot...")
    cm_if = confusion_matrix(y_test, y_pred_if)
    cm_ae = confusion_matrix(y_test, y_pred_ae)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Isolation Forest CM
    im0 = axes[0].imshow(cm_if, cmap='Oranges', interpolation='nearest')
    axes[0].set_title('Confusion Matrix - Isolation Forest', fontsize=12, fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Normal', 'Anomaly'])
    axes[0].set_yticklabels(['Normal', 'Anomaly'])
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)
    
    # Annotate cell counts
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm_if[i, j]:,}", ha="center", va="center", 
                         color="white" if cm_if[i, j] > cm_if.max()/2 else "black", fontsize=12, fontweight='bold')
            
    # Plot Autoencoder CM
    im1 = axes[1].imshow(cm_ae, cmap='Blues', interpolation='nearest')
    axes[1].set_title('Confusion Matrix - Autoencoder', fontsize=12, fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Normal', 'Anomaly'])
    axes[1].set_yticklabels(['Normal', 'Anomaly'])
    axes[1].set_xlabel('Predicted Label', fontsize=10)
    axes[1].set_ylabel('True Label', fontsize=10)
    
    # Annotate cell counts
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm_ae[i, j]:,}", ha="center", va="center", 
                         color="white" if cm_ae[i, j] > cm_ae.max()/2 else "black", fontsize=12, fontweight='bold')
            
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    print("\nSTEP 5 Completed successfully!")

if __name__ == "__main__":
    main()
