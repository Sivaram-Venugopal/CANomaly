import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support
import time

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 4 features to use
FEATURES = ['delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency']

# PyTorch Autoencoder Architecture
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

def print_metrics_table(model_name, y_true, y_pred, attack_types):
    print(f"\n==================================================")
    print(f" Performance Table: {model_name}")
    print(f"==================================================")
    print(f"{'Attack Type':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print(f"--------------------------------------------------")
    
    attacks = ['DoS', 'Fuzzy', 'Gear', 'RPM']
    
    for attack in attacks:
        idx = (attack_types == 'Normal') | (attack_types == attack)
        y_true_sub = y_true[idx]
        y_pred_sub = y_pred[idx]
        
        if len(y_true_sub) > 0 and (y_true_sub == 1).sum() > 0:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true_sub, y_pred_sub, average='binary', pos_label=1, zero_division=0
            )
            print(f"{attack:<15} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
        else:
            print(f"{attack:<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
            
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1, zero_division=0
    )
    print(f"--------------------------------------------------")
    print(f"{'Overall':<15} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
    print(f"==================================================\n")

def main():
    print("Loading features.csv...")
    t0 = time.time()
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    
    X = df[FEATURES].values
    y = df['Label'].values
    attack_types = df['Attack_Type'].values
    
    X_train, X_test, y_train, y_test, _, attack_types_test = train_test_split(
        X, y, attack_types, test_size=0.20, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # ==========================================
    # MODEL A: Isolation Forest
    # ==========================================
    print("\nTraining Model A: Isolation Forest...")
    t_if = time.time()
    clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled)
    print(f"Isolation Forest trained in {time.time() - t_if:.2f}s")
    
    if_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    joblib.dump(clf, if_path)
    print(f"Saved Isolation Forest to {if_path}")
    
    y_pred_if = (clf.predict(X_test_scaled) == -1).astype(int)
    print_metrics_table("Isolation Forest (Model A)", y_test, y_pred_if, attack_types_test)
    
    # ==========================================
    # MODEL B: Autoencoder
    # ==========================================
    print("\nTraining Model B: Lightweight Autoencoder...")
    normal_idx = (y_train == 0)
    X_train_normal = X_train_scaled[normal_idx]
    print(f"Normal traffic rows for training: {len(X_train_normal):,}")
    
    device = torch.device('cpu')
    train_dataset = TensorDataset(torch.FloatTensor(X_train_normal))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = AnomalyAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    t_ae = time.time()
    epochs = 20
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(X_train_normal)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")
            
    print(f"Autoencoder trained in {time.time() - t_ae:.2f}s")
    
    ae_path = os.path.join(MODELS_DIR, "autoencoder.pt")
    torch.save(model.state_dict(), ae_path)
    print(f"Saved Autoencoder state_dict to {ae_path}")
    
    model.eval()
    with torch.no_grad():
        X_train_normal_tensor = torch.FloatTensor(X_train_normal).to(device)
        reconstructed_normal = model(X_train_normal_tensor)
        mse_normal = torch.mean((X_train_normal_tensor - reconstructed_normal) ** 2, dim=1).numpy()
        threshold = np.percentile(mse_normal, 95)
        print(f"Determined Reconstruction Threshold (95th percentile of normal training): {threshold:.6f}")
        
        threshold_path = os.path.join(MODELS_DIR, "threshold.txt")
        with open(threshold_path, "w") as f:
            f.write(str(threshold))
            
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        reconstructed_test = model(X_test_tensor)
        mse_test = torch.mean((X_test_tensor - reconstructed_test) ** 2, dim=1).numpy()
        y_pred_ae = (mse_test > threshold).astype(int)
        
    print_metrics_table("Autoencoder (Model B)", y_test, y_pred_ae, attack_types_test)
    
    # ==========================================
    # MODEL C: One-Class SVM (Nu=0.05)
    # ==========================================
    print("\nTraining Model C: One-Class SVM...")
    # Train on normal training data only (unsupervised anomaly detection)
    # Downsample slightly to speed up SVM training if dataset is huge, but 680k features takes ~2-3 mins.
    # Let's sample max 100k normal records to train SVM in under 30 seconds while keeping high accuracy!
    t_svm = time.time()
    svm_sample_size = min(len(X_train_normal), 100000)
    indices = np.random.choice(len(X_train_normal), svm_sample_size, replace=False)
    X_train_normal_svm = X_train_normal[indices]
    
    clf_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    clf_svm.fit(X_train_normal_svm)
    print(f"One-Class SVM trained on {len(X_train_normal_svm):,} rows in {time.time() - t_svm:.2f}s")
    
    svm_path = os.path.join(MODELS_DIR, "one_class_svm.pkl")
    joblib.dump(clf_svm, svm_path)
    print(f"Saved One-Class SVM to {svm_path}")
    
    y_pred_svm = (clf_svm.predict(X_test_scaled) == -1).astype(int)
    print_metrics_table("One-Class SVM (Model C)", y_test, y_pred_svm, attack_types_test)
    
    print("STEP 2 Completed successfully!")

if __name__ == "__main__":
    main()
