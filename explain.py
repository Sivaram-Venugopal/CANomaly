import pandas as pd
import numpy as np
import os
import json
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"
OUTPUTS_DIR = "D:/Tata Innovent/CANomaly/outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

FEATURE_NAMES = [
    'delta_t', 
    'can_id_freq', 
    'payload_entropy', 
    'dlc_consistency', 
    'payload_byte_variance', 
    'inter_id_interval_mean', 
    'can_id_transition_entropy'
]

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

def get_payload_variance_vectorized(df_temp):
    cols = ['3', '4', '5', '6', '7', '8', '9', '10']
    num_df = pd.DataFrame(index=df_temp.index)
    
    # Precompute hex lookup dictionary for maximum speed
    hex_dict = {}
    for i in range(256):
        hex_dict[f"{i:02x}"] = i
        hex_dict[f"{i:02X}"] = i
        hex_dict[f"{i:x}"] = i
        hex_dict[f"{i:X}"] = i
        # Also support integers if already parsed
        hex_dict[i] = i
        hex_dict[float(i)] = i
        
    for c in cols:
        if c in df_temp.columns:
            s = df_temp[c]
            num_df[c] = s.map(hex_dict)
            
    return num_df.var(axis=1).fillna(0.0)

def calc_transition_entropy(can_ids):
    if len(can_ids) < 2:
        return 0.0
    transitions = [f"{can_ids[i]}_{can_ids[i+1]}" for i in range(len(can_ids)-1)]
    counts = {}
    for t in transitions:
        counts[t] = counts.get(t, 0) + 1
    total = len(transitions)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())

def main():
    print("Loading features.csv for explanation profiling...")
    t0 = time.time()
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    
    # Calculate additional message-level features in a fast, vectorized way
    print("Calculating payload byte variance...")
    df['payload_byte_variance'] = get_payload_variance_vectorized(df)
    
    print("Calculating inter-ID interval...")
    df['inter_id_interval'] = df.groupby('CAN_ID')['Timestamp'].diff().fillna(0)
    
    # Define 100ms time windows
    df['window'] = df['Timestamp'] // 0.1
    
    # Split into train/test using the exact same random state as train.py
    X_cols = ['delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency']
    X = df[X_cols].values
    y = df['Label'].values
    
    # Get the train/test split of dataframes
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42, stratify=y)
    
    # ==========================================
    # STEP 3A: Compute and Save Normal Stats on Training Normal Data
    # ==========================================
    print("Computing normal baseline statistics from training set...")
    # Filter normal-only training data
    df_train_normal = df_train[df_train['Attack_Type'] == 'Normal']
    
    # Group by window
    train_normal_grouped = df_train_normal.groupby('window')
    train_normal_windows = train_normal_grouped.agg(
        delta_t=('delta_t', 'mean'),
        can_id_freq=('can_id_freq', 'mean'),
        payload_entropy=('payload_entropy', 'mean'),
        dlc_consistency=('dlc_consistency', 'mean'),
        payload_byte_variance=('payload_byte_variance', 'mean'),
        inter_id_interval_mean=('inter_id_interval', 'mean')
    )
    
    # Transition entropy per window
    train_normal_trans_entropy = {}
    for win, grp in train_normal_grouped:
        train_normal_trans_entropy[win] = calc_transition_entropy(grp['CAN_ID'].tolist())
    train_normal_windows['can_id_transition_entropy'] = pd.Series(train_normal_trans_entropy)
    
    # Compute mean and std for each feature
    normal_stats = {}
    for feat in FEATURE_NAMES:
        mean_val = float(train_normal_windows[feat].mean())
        std_val = float(train_normal_windows[feat].std())
        if std_val == 0:
            std_val = 1e-6
        normal_stats[feat] = {"mean": mean_val, "std": std_val}
        
    stats_path = os.path.join(MODELS_DIR, "normal_stats.json")
    with open(stats_path, "w") as f:
        json.dump(normal_stats, f, indent=4)
    print(f"Saved normal statistics baseline to {stats_path}")
    
    # ==========================================
    # Compute Ensemble predictions on Test Set
    # ==========================================
    print("Loading models to predict test set ensemble scores...")
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    clf_if = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    
    model_ae = AnomalyAutoencoder()
    model_ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt")))
    model_ae.eval()
    
    with open(os.path.join(MODELS_DIR, "threshold.txt"), "r") as f:
        threshold_ae = float(f.read().strip())
        
    # Scale test features
    X_test_scaled = scaler.transform(df_test[X_cols].values)
    
    # Predictions
    y_pred_if = (clf_if.predict(X_test_scaled) == -1).astype(int)
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        reconstructed_test = model_ae(X_test_tensor)
        scores_ae = torch.mean((X_test_tensor - reconstructed_test) ** 2, dim=1).numpy()
        y_pred_ae = (scores_ae > threshold_ae).astype(int)
        
    # Ensemble prediction per frame: average of the two predictions
    # This represents the consensus score
    df_test = df_test.copy()
    df_test['pred_if'] = y_pred_if
    df_test['pred_ae'] = y_pred_ae
    df_test['ensemble_pred'] = (y_pred_if + y_pred_ae) / 2.0
    
    # ==========================================
    # STEP 3B & 3C: Anomaly Window Explanation on Test Set
    # ==========================================
    print("Grouping test set by 100ms windows...")
    test_grouped = df_test.groupby('window')
    
    test_windows = test_grouped.agg(
        delta_t=('delta_t', 'mean'),
        can_id_freq=('can_id_freq', 'mean'),
        payload_entropy=('payload_entropy', 'mean'),
        dlc_consistency=('dlc_consistency', 'mean'),
        payload_byte_variance=('payload_byte_variance', 'mean'),
        inter_id_interval_mean=('inter_id_interval', 'mean'),
        ensemble_score=('ensemble_pred', 'mean'),
        label_sum=('Label', 'sum')
    )
    
    # Compute transition entropy per test window
    test_trans_entropy = {}
    for win, grp in test_grouped:
        test_trans_entropy[win] = calc_transition_entropy(grp['CAN_ID'].tolist())
    test_windows['can_id_transition_entropy'] = pd.Series(test_trans_entropy)
    
    # A window is flagged if the ensemble score > 0.5 (meaning the models consistently flag it as anomalous)
    # Or if it contains real attacks (so we can explain it!)
    # Let's flag windows where ensemble_score > 0.5 (or where label_sum > 0, to make sure we capture all attacks for the report)
    flagged_windows = test_windows[(test_windows['ensemble_score'] > 0.5) | (test_windows['label_sum'] > 0)].copy()
    print(f"Flagged anomaly windows in test set: {len(flagged_windows)}")
    
    explanations = []
    
    for win_id, row in flagged_windows.iterrows():
        # Get start and end frame indices in original dataframe (sorted chronologically)
        grp = test_grouped.get_group(win_id).sort_index()
        start_frame = int(grp.index[0])
        end_frame = int(grp.index[-1])
        
        # Compute z-scores
        z_scores = {}
        deviant_features = []
        for feat in FEATURE_NAMES:
            mean_ref = normal_stats[feat]["mean"]
            std_ref = normal_stats[feat]["std"]
            z = (row[feat] - mean_ref) / std_ref
            z_scores[feat] = z
            if abs(z) > 2.0:
                deviant_features.append(feat)
                
        # Rule-based fingerprinting (Step 3B)
        fingerprint = "Unknown"
        explanation_msg = "Unknown anomaly pattern: manual inspection recommended"
        
        # 1. DoS Rule
        if z_scores['delta_t'] < -2.0 and z_scores['can_id_freq'] > 2.0:
            fingerprint = "Suspected DoS"
            explanation_msg = "Suspected DoS: abnormally high message frequency detected"
        # 2. Fuzzy Rule
        elif z_scores['payload_entropy'] > 2.0 and z_scores['payload_byte_variance'] > 2.0:
            fingerprint = "Suspected Fuzzy"
            explanation_msg = "Suspected Fuzzy Attack: payload randomisation pattern detected"
        # 3. Spoofing Rule
        elif z_scores['inter_id_interval_mean'] > 2.0 and z_scores['can_id_transition_entropy'] < -2.0:
            fingerprint = "Suspected Spoofing"
            explanation_msg = "Suspected Spoofing: periodic ID injection with low CAN sequence entropy"
            
        # Confidence calculation
        num_deviant = len(deviant_features)
        if num_deviant >= 4:
            confidence = "High"
        elif num_deviant >= 2:
            confidence = "Medium"
        else:
            confidence = "Low"
            
        explanations.append({
            "window_id": int(win_id),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "ensemble_score": float(row['ensemble_score']),
            "deviant_features": deviant_features,
            "z_scores": {k: float(v) for k, v in z_scores.items()},
            "fingerprint": fingerprint,
            "explanation": explanation_msg,
            "confidence": confidence
        })
        
    # Save all explanations
    explanations_path = os.path.join(OUTPUTS_DIR, "attack_explanations.json")
    with open(explanations_path, "w") as f:
        json.dump(explanations, f, indent=4)
    print(f"Saved explanations report to {explanations_path}")
    
    # Print first 3 explanations to console
    print("\n==============================================")
    print(" FIRST 3 FLAGGED WINDOW EXPLANATIONS")
    print("==============================================")
    for i, item in enumerate(explanations[:3]):
        print(f"\n[{i+1}] Window ID: {item['window_id']}")
        print(f"    Frame Range: {item['start_frame']} - {item['end_frame']}")
        print(f"    Ensemble Score: {item['ensemble_score']:.4f}")
        print(f"    Deviant Features: {item['deviant_features']}")
        print(f"    Fingerprint: {item['fingerprint']}")
        print(f"    Explanation: {item['explanation']}")
        print(f"    Confidence: {item['confidence']}")
    print("==============================================\n")
    
    # Also save window stats for generate_report.py
    window_df_path = os.path.join(MODELS_DIR, "test_windows.csv")
    test_windows.to_csv(window_df_path)
    
    print("STEP 3 Completed successfully!")

if __name__ == "__main__":
    main()
