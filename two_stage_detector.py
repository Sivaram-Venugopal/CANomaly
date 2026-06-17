import numpy as np
import torch
from secure_logger import HashChainLogger

class TwoStageDetector:
    def __init__(self, autoencoder, scaler, mse_threshold, plausibility_idx, log_path=None):
        self.autoencoder = autoencoder
        self.scaler = scaler
        self.mse_threshold = mse_threshold
        self.plausibility_idx = plausibility_idx  # index of physical_plausibility_score in feature vector
        self.logger = HashChainLogger(log_path=log_path) if log_path else None

    def predict(self, feature_vector):
        # STAGE 1: deterministic physical check (bypasses statistics entirely)
        # Check if the plausibility score indicates a violation (1.0 or close to 1.0 if scaled)
        # If the input feature_vector is scaled, 1.0 represents the active anomaly state.
        # We check for >= 0.5 to handle floating point scaling robustly.
        if feature_vector[self.plausibility_idx] >= 0.5:
            res = {"is_anomaly": True, "stage": "physical_gate", "confidence": "High", "mse": 0.0}
            if self.logger:
                self.logger.log_event({
                    "is_anomaly": True,
                    "stage": res["stage"],
                    "confidence": res["confidence"],
                    "mse": res["mse"],
                    "features": [float(x) for x in feature_vector]
                })
            return res
        
        # STAGE 2: statistical check on remaining features (drop plausibility 
        # column before computing MSE, since stage 1 already used it)
        features_for_mse = np.delete(feature_vector, self.plausibility_idx)
        
        # Ensure correct batch dimensions for the PyTorch Autoencoder (adds batch dimension if 1D)
        is_1d = features_for_mse.ndim == 1
        if is_1d:
            features_tensor = torch.FloatTensor(features_for_mse).unsqueeze(0)
        else:
            features_tensor = torch.FloatTensor(features_for_mse)
            
        with torch.no_grad():
            recon_tensor = self.autoencoder(features_tensor)
            if is_1d:
                recon = recon_tensor.squeeze(0).numpy()
            else:
                recon = recon_tensor.numpy()
                
        mse = ((recon - features_for_mse) ** 2).mean()
        is_anomaly = mse > self.mse_threshold
        res = {
            "is_anomaly": is_anomaly, 
            "stage": "statistical", 
            "confidence": "Medium" if is_anomaly else "Low", 
            "mse": float(mse)
        }
        if self.logger and is_anomaly:
            self.logger.log_event({
                "is_anomaly": True,
                "stage": res["stage"],
                "confidence": res["confidence"],
                "mse": res["mse"],
                "features": [float(x) for x in feature_vector]
            })
        return res

