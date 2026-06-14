import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import time

# Set Streamlit page config
st.set_page_config(
    page_title="CANomaly — CAN Bus Intrusion Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Dark Mode glassmorphism look)
st.markdown("""
    <style>
        .reportview-container {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        .main {
            background-color: #0d1117;
        }
        .stButton>button {
            background: linear-gradient(135deg, #1f77b4 0%, #00d2ff 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.2);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 210, 255, 0.4);
        }
        .card {
            background-color: #161b22;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #30363d;
            margin-bottom: 20px;
        }
        h1, h2, h3 {
            font-family: 'Outfit', 'Inter', sans-serif;
            font-weight: 700;
        }
        .title-text {
            background: linear-gradient(to right, #00d2ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin-bottom: 5px;
        }
        .subtitle-text {
            color: #8b949e;
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #8b949e;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

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

# Load assets and models
MODELS_DIR = "D:/Tata Innovent/CANomaly/models"
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"

@st.cache_resource
def load_models_and_scaler():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    clf = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    
    ae_model = AnomalyAutoencoder()
    ae_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "autoencoder.pt")))
    ae_model.eval()
    
    with open(os.path.join(MODELS_DIR, "threshold.txt"), "r") as f:
        ae_threshold = float(f.read().strip())
        
    return scaler, clf, ae_model, ae_threshold

@st.cache_data
def load_simulation_data():
    df = pd.read_csv(FEATURES_CSV)
    # We want a sequence of 500 frames containing both normal and attacks
    # Take a sample of 500 frames and sort them by Timestamp to keep chronological order
    df_sim = df.sample(n=500, random_state=42).sort_values(by='Timestamp').copy()
    
    # Calculate some stats for manual classification
    # average frequency per window for each CAN ID
    id_freqs = df.groupby('CAN_ID')['can_id_freq'].mean().to_dict()
    # standard dev of DLC for each CAN ID
    id_dlc_stds = df.groupby('CAN_ID')['dlc_consistency'].mean().to_dict()
    # average delta_t
    avg_delta_t = df['delta_t'].mean()
    
    return df_sim, id_freqs, id_dlc_stds, avg_delta_t

try:
    scaler, clf, ae_model, ae_threshold = load_models_and_scaler()
    df_sim, id_freqs, id_dlc_stds, avg_delta_t = load_simulation_data()
except Exception as e:
    st.error(f"Error loading models or dataset: {e}. Make sure step 1 and step 2 have run successfully.")
    st.stop()

# Header block
st.markdown('<div class="title-text">CANomaly</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Self-Calibrating CAN Bus Intrusion Detector — Edge AI Prototype</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://www.tatatechnologies.com/wp-content/themes/tatatechnologies/assets/images/logo.png", width=180)
st.sidebar.markdown("### Model Configuration")
selected_model_name = st.sidebar.radio(
    "Select Model for Detection:",
    ("Lightweight Autoencoder", "Isolation Forest")
)

st.sidebar.markdown("""
---
### About the Prototype
This prototype detects malicious injections on the Controller Area Network (CAN) bus.
It runs entirely on CPU and is designed for edge microcontrollers like the STM32F4.

**Features Extracted:**
1. Inter-message arrival time (delta_t)
2. Message frequency per 100ms
3. Payload byte entropy
4. DLC consistency per CAN ID
""")

# Layout splits
col_main, col_stats = st.columns([3, 1])

with col_main:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live CAN Stream Simulator")
    
    # Simulate controls
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        start_sim = st.button("🚀 Run Live Simulation")
    with col_btn2:
        stop_sim = st.button("⏹ Stop")
        
    # Anomaly chart placeholder
    chart_placeholder = st.empty()
    # Console output placeholder
    console_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col_stats:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("System Status")
    
    # Running counters
    total_placeholder = st.metric("Total Frames Processed", "0")
    attacks_placeholder = st.metric("Attacks Detected", "0", delta_color="inverse")
    fp_placeholder = st.metric("False Positives", "0", delta_color="inverse")
    
    st.markdown("---")
    st.markdown("**Thresholds & Settings:**")
    if selected_model_name == "Lightweight Autoencoder":
        st.info(f"Model: Unsupervised AE\nThreshold: {ae_threshold:.6f}")
    else:
        st.info(f"Model: Isolation Forest\nContamination: 0.05")
    st.markdown('</div>', unsafe_allow_html=True)

# Manual Injection section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Manual CAN Message Injection Test")
col_input1, col_input2, col_input3 = st.columns([1, 2, 1])

with col_input1:
    manual_id = st.text_input("CAN ID (Hex)", "0316")
with col_input2:
    manual_payload = st.text_input("Payload (8 Bytes Space-Separated Hex)", "05 21 68 09 21 21 00 6f")
with col_input3:
    st.markdown("<br>", unsafe_allow_html=True)
    classify_manual = st.button("🔍 Classify Frame")

if classify_manual:
    # Process manual input
    try:
        # 1. Parse CAN ID
        can_id = manual_id.strip().zfill(4)
        
        # 2. Parse payload bytes
        bytes_list = [b.strip() for b in manual_payload.split() if b.strip()]
        dlc = len(bytes_list)
        
        if dlc > 8:
            st.error("Payload cannot exceed 8 bytes.")
        else:
            # 3. Calculate features
            # delta_t: use dataset average
            dt = avg_delta_t
            
            # frequency: look up in stats, fallback to 1 if new ID (which is an anomaly indicator)
            freq = id_freqs.get(can_id, 1.0)
            
            # dlc consistency: look up in stats, fallback to 1.0 if new
            dlc_std = id_dlc_stds.get(can_id, 1.0)
            
            # entropy: compute from bytes
            counts = {}
            for b in bytes_list:
                counts[b] = counts.get(b, 0) + 1
            entropy = -sum((c / dlc) * np.log2(c / dlc) for c in counts.values()) if dlc > 0 else 0.0
            
            # Scale features
            feat_vector = np.array([[dt, freq, entropy, dlc_std]])
            feat_scaled = scaler.transform(feat_vector)
            
            # Classify
            is_anomaly = False
            score = 0.0
            thresh = 0.0
            
            if selected_model_name == "Lightweight Autoencoder":
                with torch.no_grad():
                    frame_tensor = torch.FloatTensor(feat_scaled)
                    output = ae_model(frame_tensor)
                    score = torch.mean((frame_tensor - output) ** 2).item()
                thresh = ae_threshold
                is_anomaly = score > thresh
            else:
                score = -clf.score_samples(feat_scaled)[0]
                is_anomaly = clf.predict(feat_scaled)[0] == -1
                thresh = 0.5 # Approximate standard threshold for score_samples
                
            # Output result beautifully
            st.markdown("### Detection Result:")
            if is_anomaly:
                st.error(f"🚨 **ANOMALY DETECTED!** \n\n Anomaly Score: **{score:.6f}** (Threshold: {thresh:.6f})")
            else:
                st.success(f"✅ **NORMAL FRAME** \n\n Anomaly Score: **{score:.6f}** (Threshold: {thresh:.6f})")
                
            # Show feature vector details
            st.markdown("**Computed Features:**")
            feat_df = pd.DataFrame({
                "Inter-arrival (delta_t)": [f"{dt:.6f} s"],
                "ID Freq (per 100ms)": [f"{freq:.2f}"],
                "Payload Entropy": [f"{entropy:.4f}"],
                "DLC Consistency (std)": [f"{dlc_std:.4f}"]
            })
            st.table(feat_df)
            
    except Exception as ex:
        st.error(f"Error parsing manual input: {ex}")
st.markdown('</div>', unsafe_allow_html=True)

# Run Live Simulation Logic
if start_sim:
    # Initialize variables for live simulation
    scores = []
    labels = []
    colors = []
    timestamps = []
    
    total_frames = 0
    attacks_detected = 0
    false_positives = 0
    
    # We will display the stream log in a rolling list
    stream_logs = []
    
    # Extract features for simulation
    X_sim = df_sim[FEATURES].values
    X_sim_scaled = scaler.transform(X_sim)
    y_sim_true = df_sim['Label'].values
    attack_types_sim = df_sim['Attack_Type'].values
    can_ids_sim = df_sim['CAN_ID'].values
    dlcs_sim = df_sim['DLC'].values
    
    # Run the loop
    for i in range(500):
        # Frame information
        frame_scaled = X_sim_scaled[i].reshape(1, -1)
        true_label = y_sim_true[i]
        attack_type = attack_types_sim[i]
        can_id = can_ids_sim[i]
        dlc = dlcs_sim[i]
        
        # Classification
        score = 0.0
        is_anomaly = False
        
        if selected_model_name == "Lightweight Autoencoder":
            with torch.no_grad():
                frame_tensor = torch.FloatTensor(frame_scaled)
                output = ae_model(frame_tensor)
                score = torch.mean((frame_tensor - output) ** 2).item()
            is_anomaly = score > ae_threshold
        else:
            score = -clf.score_samples(frame_scaled)[0]
            is_anomaly = clf.predict(frame_scaled)[0] == -1
            
        # Update metrics
        total_frames += 1
        
        # Attack detected: model predicts anomaly (is_anomaly == True)
        if is_anomaly:
            attacks_detected += 1
            # False positive: predicted anomaly, but true label is Normal (true_label == 0)
            if true_label == 0:
                false_positives += 1
                
        # Store for plotting
        scores.append(score)
        timestamps.append(i)
        
        # Update metric displays
        total_placeholder.metric("Total Frames Processed", f"{total_frames}")
        attacks_placeholder.metric("Attacks Detected", f"{attacks_detected}")
        fp_placeholder.metric("False Positives", f"{false_positives}")
        
        # Update Live Anomaly Score Line Chart
        chart_data = pd.DataFrame({
            "Anomaly Score": scores
        }, index=timestamps)
        
        chart_placeholder.line_chart(chart_data)
        
        # Update Stream Log
        status_symbol = "🚨" if is_anomaly else "✅"
        log_color = "red" if is_anomaly else "green"
        log_text = f"<span style='color: {log_color}; font-family: monospace;'>{status_symbol} Frame #{i+1:03d} | CAN_ID: 0x{can_id} | DLC: {dlc} | True Label: {attack_type} | Score: {score:.6f}</span>"
        stream_logs.insert(0, log_text) # insert at beginning
        
        # Keep logs list bounded to 10 lines
        if len(stream_logs) > 10:
            stream_logs.pop()
            
        console_html = "<br>".join(stream_logs)
        console_placeholder.markdown(f"**Live CAN Frames Log (latest on top):**<br>{console_html}", unsafe_allow_html=True)
        
        # 10ms delay as requested
        time.sleep(0.01)
        
    st.success("Simulation completed successfully!")
