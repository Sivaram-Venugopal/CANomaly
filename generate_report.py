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

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"
OUTPUTS_DIR = "D:/Tata Innovent/CANomaly/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

FEATURES = ['delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency']
FEATURE_NAMES = [
    'delta_t', 
    'can_id_freq', 
    'payload_entropy', 
    'dlc_consistency', 
    'payload_byte_variance', 
    'inter_id_interval_mean', 
    'can_id_transition_entropy'
]

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
    
    # Reconstruct train/test split
    _, X_test, _, y_test, _, attack_types_test = train_test_split(
        X, y, attack_types, test_size=0.20, random_state=42, stratify=y
    )
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
    # Get Single-Frame Predictions
    # ==========================================
    print("Computing single-frame predictions...")
    scores_if = -clf_if.score_samples(X_test_scaled)
    y_pred_if = (clf_if.predict(X_test_scaled) == -1).astype(int)
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        reconstructed_test = model_ae(X_test_tensor)
        scores_ae = torch.mean((X_test_tensor - reconstructed_test) ** 2, dim=1).numpy()
        y_pred_ae = (scores_ae > threshold_ae).astype(int)
        
    y_pred_rule = get_rule_based_predictions(df_test)
    
    metrics_if = evaluate_subsets(y_test, y_pred_if, attack_types_test)
    metrics_ae = evaluate_subsets(y_test, y_pred_ae, attack_types_test)
    metrics_rule = evaluate_subsets(y_test, y_pred_rule, attack_types_test)
    
    # ==========================================
    # Compute Window Ensemble Metrics
    # ==========================================
    print("Computing window-level ensemble metrics...")
    # Load window stats computed in explain.py
    window_df_path = os.path.join(MODELS_DIR, "test_windows.csv")
    if os.path.exists(window_df_path):
        win_df = pd.read_csv(window_df_path)
        
        # Ground truth window label (any frame in window is an attack)
        win_df['true_label'] = (win_df['label_sum'] > 0).astype(int)
        
        # Ensemble prediction (score > 0.5)
        win_df['pred_ensemble'] = (win_df['ensemble_score'] > 0.5).astype(int)
        
        # To compute per-attack metrics, we must map window to its attack type
        # Group df_test by window and extract the dominant non-Normal Attack_Type
        df_test_copy = df_test.copy()
        df_test_copy['window'] = df_test_copy['Timestamp'] // 0.1
        
        window_attack_map = df_test_copy.groupby('window')['Attack_Type'].apply(
            lambda x: 'Normal' if (x == 'Normal').all() else x[x != 'Normal'].iloc[0]
        ).to_dict()
        
        win_df['Attack_Type'] = win_df['window'].map(window_attack_map)
        
        metrics_ensemble = evaluate_subsets(
            win_df['true_label'].values, 
            win_df['pred_ensemble'].values, 
            win_df['Attack_Type'].values
        )
    else:
        print("Warning: test_windows.csv not found. Run explain.py first.")
        metrics_ensemble = {}

    # Save results to json (before/after comparison)
    results = {
        "SingleFrame_RuleBased": metrics_rule,
        "SingleFrame_IsolationForest": metrics_if,
        "SingleFrame_Autoencoder": metrics_ae,
        "WindowEnsemble_Consensus": metrics_ensemble
    }
    
    results_path = os.path.join(OUTPUTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved numerical metrics to {results_path}")
    
    # ==========================================
    # 1. Plot comparison.png (Original plot updated)
    # ==========================================
    print("Updating comparison chart...")
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
    ax.set_title('Detection F1-Score Comparison (Single-Frame)', fontsize=14, fontweight='bold', pad=15)
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
    comparison_path = os.path.join(OUTPUTS_DIR, "comparison.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    
    # ==========================================
    # 2. Plot ROC curves (roc_curve.png)
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
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (Single-Frame Anomaly Scores)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "roc_curve.png"), dpi=300)
    plt.close()
    
    # ==========================================
    # 3. Plot Confusion Matrices (confusion_matrix.png)
    # ==========================================
    print("Generating Confusion Matrix plot...")
    cm_if = confusion_matrix(y_test, y_pred_if)
    cm_ae = confusion_matrix(y_test, y_pred_ae)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm, title, cmap in zip(axes, [cm_if, cm_ae], ['Isolation Forest', 'Autoencoder'], ['Oranges', 'Blues']):
        im = ax.imshow(cm, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Confusion Matrix - {title}', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", 
                        color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # ==========================================
    # NEW ARTIFACT 1: feature_importance.png
    # ==========================================
    print("Generating feature importance chart...")
    explanations_path = os.path.join(OUTPUTS_DIR, "attack_explanations.json")
    if os.path.exists(explanations_path):
        with open(explanations_path, "r") as f:
            exps = json.load(f)
        
        # Calculate mean absolute z-score per feature across all flagged attack windows
        if len(exps) > 0:
            z_abs_sums = {feat: 0.0 for feat in FEATURE_NAMES}
            for exp in exps:
                for feat in FEATURE_NAMES:
                    z_abs_sums[feat] += abs(exp["z_scores"][feat])
            
            mean_z_abs = {feat: z_abs_sums[feat]/len(exps) for feat in FEATURE_NAMES}
            
            # Sort features by importance
            sorted_feats = sorted(mean_z_abs.items(), key=lambda x: x[1], reverse=True)
            sorted_names, sorted_vals = zip(*sorted_feats)
            
            plt.figure(figsize=(10, 6))
            colors = plt.cm.plasma(np.linspace(0.8, 0.3, len(sorted_names)))
            plt.barh(sorted_names[::-1], sorted_vals[::-1], color=colors)
            plt.xlabel('Mean Absolute Z-Score (Deviance from Normal)', fontsize=12, fontweight='bold')
            plt.title('Feature Deviancy Importance Across Flagged Windows', fontsize=14, fontweight='bold', pad=15)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            feat_imp_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
            plt.savefig(feat_imp_path, dpi=300)
            plt.close()
            print(f"Saved feature importance to {feat_imp_path}")
            
    # ==========================================
    # NEW ARTIFACT 2: ensemble_vs_single_comparison.png
    # ==========================================
    if os.path.exists(window_df_path):
        print("Generating ensemble vs single F1 comparison chart...")
        f1_single_ae = [metrics_ae[c]['f1_score'] for c in categories]
        f1_ensemble_consensus = [metrics_ensemble[c]['f1_score'] for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, f1_single_ae, width, label='Single-Frame (Autoencoder)', color='#ff7f0e', alpha=0.85)
        ax.bar(x + width/2, f1_ensemble_consensus, width, label='Window Ensemble (Consensus)', color='#1f77b4', alpha=0.85)
        
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('Single-Frame vs. Window Ensemble F1-Score Improvement', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add labels
        for idx, rect in enumerate(ax.patches):
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
                
        plt.tight_layout()
        ens_comp_path = os.path.join(OUTPUTS_DIR, "ensemble_vs_single_comparison.png")
        plt.savefig(ens_comp_path, dpi=300)
        plt.close()
        print(f"Saved ensemble comparison to {ens_comp_path}")
        
    # ==========================================
    # NEW ARTIFACT 3: attack_type_distribution.png
    # ==========================================
    if os.path.exists(explanations_path):
        print("Generating attack type distribution pie chart...")
        fingerprints = [exp["fingerprint"] for exp in exps]
        unique_fingerprints, counts = np.unique(fingerprints, return_counts=True)
        
        # Clean labels
        labels = []
        for fp in unique_fingerprints:
            if fp == "Suspected DoS":
                labels.append("DoS Attack")
            elif fp == "Suspected Fuzzy":
                labels.append("Fuzzy Attack")
            elif fp == "Suspected Spoofing":
                labels.append("Spoofing Attack")
            else:
                labels.append("Unknown/Unclassified")
                
        # Premium color palette for the pie chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        colors = colors[:len(labels)]
        
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, 
                colors=colors, textprops={'fontsize': 11, 'weight': 'bold'})
        plt.title('Fingerprint Distribution of Detected Anomalies', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        dist_path = os.path.join(OUTPUTS_DIR, "attack_type_distribution.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()
        print(f"Saved attack type distribution to {dist_path}")
        
    print("\nSTEP 5 Completed successfully!")

if __name__ == "__main__":
    main()
