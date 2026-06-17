import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import time
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

# Paths
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
        
    with open(os.path.join(MODELS_DIR, "normal_stats.json"), "r") as f:
        normal_stats = json.load(f)
        
    return scaler, clf, ae_model, ae_threshold, normal_stats

@st.cache_data
def load_simulation_data():
    replay_path = "D:/Tata Innovent/CANomaly/outputs/replay_stream.csv"
    if os.path.exists(replay_path):
        df_sim = pd.read_csv(replay_path)
        # Standardize column names for simulator compatibility
        df_sim.rename(columns={'can_id': 'CAN_ID', 'injected_attack_type': 'Attack_Type'}, inplace=True)
    else:
        df = pd.read_csv(FEATURES_CSV)
        df_sim = df.sample(n=500, random_state=42).sort_values(by='Timestamp').copy()
        
    df = pd.read_csv(FEATURES_CSV)
    # Stats for manual mode
    id_freqs = df.groupby('CAN_ID')['can_id_freq'].mean().to_dict()
    id_dlc_stds = df.groupby('CAN_ID')['dlc_consistency'].mean().to_dict()
    avg_delta_t = df['delta_t'].mean()
    
    return df_sim, id_freqs, id_dlc_stds, avg_delta_t

try:
    scaler, clf, ae_model, ae_threshold, normal_stats = load_models_and_scaler()
    df_sim, id_freqs, id_dlc_stds, avg_delta_t = load_simulation_data()
    
    # Import AdaptiveThreshold
    from adaptive_threshold import AdaptiveThreshold
    
    # Load tracker state into session state
    if 'tracker' not in st.session_state:
        tracker_path = os.path.join(MODELS_DIR, "adaptive_threshold.pkl")
        if os.path.exists(tracker_path):
            try:
                st.session_state.tracker = joblib.load(tracker_path)
            except Exception:
                st.session_state.tracker = AdaptiveThreshold(alpha=0.05, sigma=3.0, warmup_frames=50)
        else:
            st.session_state.tracker = AdaptiveThreshold(alpha=0.05, sigma=3.0, warmup_frames=50)
except Exception as e:
    st.error(f"Error loading models or data: {e}. Run step 1, 2, and 3 first.")
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

detection_mode = st.sidebar.radio(
    "Detection Mode:",
    ("Single-frame", "Window ensemble")
)

use_adaptive_threshold = False
if selected_model_name == "Lightweight Autoencoder":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Adaptive Baselines")
    use_adaptive_threshold = st.sidebar.toggle(
        "Adaptive threshold mode", 
        value=False,
        help="Self-calibrates the Autoencoder anomaly threshold per CAN ID in real time."
    )

st.sidebar.markdown("""
---
### Explainable AI (XAI)
In **Window ensemble** mode, CANomaly groups messages into 100ms windows and calculates 7 statistical features. 
Any window with an ensemble score $> 0.5$ triggers a diagnosis using statistical z-scores.
""")

# Initialize Session State for XAI results
if 'xai_data' not in st.session_state:
    st.session_state.xai_data = None

# Layout splits
col_main, col_stats = st.columns([3, 1])

with col_main:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live CAN Stream Simulator")
    
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        start_sim = st.button("🚀 Run Live Simulation")
    with col_btn2:
        stop_sim = st.button("⏹ Reset")
        if stop_sim:
            st.session_state.xai_data = None
            if 'tracker' in st.session_state:
                del st.session_state.tracker
            st.rerun()
        
    chart_placeholder = st.empty()
    console_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # XAI Panel Section below chart
    xai_placeholder = st.container()

with col_stats:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("System Status")
    
    total_placeholder = st.metric("Total Frames Processed", "0")
    attacks_placeholder = st.metric("Attacks Detected", "0", delta_color="inverse")
    fp_placeholder = st.metric("False Positives", "0", delta_color="inverse")
    
    # Adaptive baselines learned card
    baselines_placeholder = st.empty()
    if use_adaptive_threshold:
        num_baselines = len(st.session_state.tracker.ema_mean)
        baselines_placeholder.metric("Adaptive Baselines Learned", f"{num_baselines}")
    
    st.markdown("---")
    st.markdown("**Thresholds & Settings:**")
    if selected_model_name == "Lightweight Autoencoder":
        if use_adaptive_threshold:
            st.info(f"Model: Unsupervised AE\nThreshold: Adaptive (Self-Calibrated)")
        else:
            st.info(f"Model: Unsupervised AE\nThreshold: {ae_threshold:.6f}")
    else:
        st.info(f"Model: Isolation Forest\nContamination: 0.05")
    st.markdown('</div>', unsafe_allow_html=True)

# Helper function to compute transition entropy
def calc_transition_entropy(can_ids):
    if len(can_ids) < 2:
        return 0.0
    transitions = [f"{can_ids[i]}_{can_ids[i+1]}" for i in range(len(can_ids)-1)]
    counts = {}
    for t in transitions:
        counts[t] = counts.get(t, 0) + 1
    total = len(transitions)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())

# Helper function to compute payload byte variance
def calc_byte_variance(row_dict):
    dlc = int(row_dict['DLC'])
    vals = []
    # Columns in features.csv for payload are '3' to '10'
    for c in ['3', '4', '5', '6', '7', '8', '9', '10']:
        val = row_dict.get(c)
        if pd.notna(val) and val != '' and str(val).lower() != 'nan':
            try:
                if isinstance(val, str):
                    vals.append(int(val, 16))
                else:
                    vals.append(int(val))
            except ValueError:
                pass
    vals = vals[:dlc]
    return np.var(vals) if vals else 0.0

# Live Simulation Execution
if start_sim:
    # Set up data slices
    X_sim = df_sim[FEATURES].values
    X_sim_scaled = scaler.transform(X_sim)
    y_sim_true = df_sim['Label'].values
    attack_types_sim = df_sim['Attack_Type'].values
    can_ids_sim = df_sim['CAN_ID'].values
    dlcs_sim = df_sim['DLC'].values
    timestamps_sim = df_sim['Timestamp'].values
    
    # Store original row payloads
    rows_dict = df_sim.to_dict('records')
    
    # Track statistics
    total_frames = 0
    attacks_detected = 0
    false_positives = 0
    
    # For plot
    plot_x = []
    plot_y = []
    plot_colors = []
    plot_thresh = []
    
    # Stream logs
    stream_logs = []
    
    # For Window Ensemble processing
    current_window_id = None
    window_buffer = []
    
    for i in range(len(df_sim)):
        # Frame metadata
        frame_scaled = X_sim_scaled[i].reshape(1, -1)
        true_label = y_sim_true[i]
        attack_type = attack_types_sim[i]
        can_id = can_ids_sim[i]
        dlc = dlcs_sim[i]
        ts = timestamps_sim[i]
        row_data = rows_dict[i]
        
        # Predict anomaly
        is_frame_anomaly = False
        frame_score = 0.0
        
        # Autoencoder
        with torch.no_grad():
            frame_tensor = torch.FloatTensor(frame_scaled)
            output = ae_model(frame_tensor)
            ae_score = torch.mean((frame_tensor - output) ** 2).item()
            
            if use_adaptive_threshold:
                # Update EMA only if normal frame (true_label == 0) or during warmup
                is_training = (true_label == 0)
                ae_pred = 1 if st.session_state.tracker.is_anomaly(can_id, ae_score, is_training=is_training) else 0
                
                # Dynamically track threshold for plotting
                mean_val = st.session_state.tracker.ema_mean.get(can_id, ae_score)
                std_val = (st.session_state.tracker.ema_var.get(can_id, 0.0) ** 0.5)
                thresh_val = mean_val + 3.0 * std_val
                plot_thresh.append(thresh_val)
            else:
                ae_pred = 1 if ae_score > ae_threshold else 0
            
        # Isolation Forest
        if_score = -clf.score_samples(frame_scaled)[0]
        if_pred = 1 if clf.predict(frame_scaled)[0] == -1 else 0
        
        # Decide frame-level parameters
        if selected_model_name == "Lightweight Autoencoder":
            frame_score = ae_score
            is_frame_anomaly = (ae_pred == 1)
        else:
            frame_score = if_score
            is_frame_anomaly = (if_pred == 1)
            
        # Increment counters
        total_frames += 1
        
        # Calculate window ID (100ms time blocks)
        win_id = ts // 0.1
        
        # Pack data for this frame to buffer
        frame_meta = {
            'Timestamp': ts,
            'CAN_ID': can_id,
            'DLC': dlc,
            'delta_t': row_data['delta_t'],
            'can_id_freq': row_data['can_id_freq'],
            'payload_entropy': row_data['payload_entropy'],
            'dlc_consistency': row_data['dlc_consistency'],
            'payload_byte_variance': calc_byte_variance(row_data),
            'ae_pred': ae_pred,
            'if_pred': if_pred,
            'true_label': true_label,
            'attack_type': attack_type
        }
        
        if current_window_id is None:
            current_window_id = win_id
            
        # If we transition to a new window, process the completed window
        if win_id != current_window_id and len(window_buffer) > 0:
            # PROCESS COMPLETED WINDOW
            # 1. Compute means of the 7 features
            delta_t_mean = np.mean([f['delta_t'] for f in window_buffer])
            can_id_freq_mean = np.mean([f['can_id_freq'] for f in window_buffer])
            payload_entropy_mean = np.mean([f['payload_entropy'] for f in window_buffer])
            dlc_consistency_mean = np.mean([f['dlc_consistency'] for f in window_buffer])
            payload_byte_variance_mean = np.mean([f['payload_byte_variance'] for f in window_buffer])
            
            # Inter-ID interval
            # Calculated as time since last timestamp of the same ID in this window
            # Fallback to delta_t_mean if only one message
            intervals = []
            id_last_ts = {}
            for f in window_buffer:
                cid = f['CAN_ID']
                t_val = f['Timestamp']
                if cid in id_last_ts:
                    intervals.append(t_val - id_last_ts[cid])
                id_last_ts[cid] = t_val
            inter_id_interval_mean = np.mean(intervals) if intervals else delta_t_mean
            
            # CAN ID Transition Entropy
            can_id_sequence = [f['CAN_ID'] for f in window_buffer]
            can_id_transition_entropy = calc_transition_entropy(can_id_sequence)
            
            # 2. Ensemble Score (consensual prediction of both models)
            # score = mean of (ae_pred + if_pred) / 2.0
            win_ensemble_score = np.mean([(f['ae_pred'] + f['if_pred']) / 2.0 for f in window_buffer])
            
            # 3. Z-scores calculation
            z_scores = {}
            win_features = {
                'delta_t': delta_t_mean,
                'can_id_freq': can_id_freq_mean,
                'payload_entropy': payload_entropy_mean,
                'dlc_consistency': dlc_consistency_mean,
                'payload_byte_variance': payload_byte_variance_mean,
                'inter_id_interval_mean': inter_id_interval_mean,
                'can_id_transition_entropy': can_id_transition_entropy
            }
            
            deviant_features = []
            for feat in FEATURE_NAMES:
                mean_ref = normal_stats[feat]["mean"]
                std_ref = normal_stats[feat]["std"]
                z = (win_features[feat] - mean_ref) / std_ref
                z_scores[feat] = z
                if abs(z) > 2.0:
                    deviant_features.append(feat)
                    
            # 4. Fingerprinting
            fingerprint = "Unknown"
            explanation_msg = "Unknown anomaly pattern: manual inspection recommended"
            if z_scores['delta_t'] < -2.0 and z_scores['can_id_freq'] > 2.0:
                fingerprint = "Suspected DoS"
                explanation_msg = "Suspected DoS: abnormally high message frequency detected"
            elif z_scores['payload_entropy'] > 2.0 and z_scores['payload_byte_variance'] > 2.0:
                fingerprint = "Suspected Fuzzy"
                explanation_msg = "Suspected Fuzzy Attack: payload randomisation pattern detected"
            elif z_scores['inter_id_interval_mean'] > 2.0 and z_scores['can_id_transition_entropy'] < -2.0:
                fingerprint = "Suspected Spoofing"
                explanation_msg = "Suspected Spoofing: periodic ID injection with low CAN sequence entropy"
                
            confidence = "High" if len(deviant_features) >= 4 else ("Medium" if len(deviant_features) >= 2 else "Low")
            
            # Update metric logs and plot if in window ensemble mode
            if detection_mode == "Window ensemble":
                plot_x.append(current_window_id)
                plot_y.append(win_ensemble_score)
                plot_colors.append('red' if win_ensemble_score > 0.5 else 'green')
                
                # Check for detection counters
                # Window true label: is there any attack frame in the buffer?
                win_has_attack = any(f['true_label'] == 1 for f in window_buffer)
                
                if win_ensemble_score > 0.5:
                    attacks_detected += 1
                    if not win_has_attack:
                        false_positives += 1
                        
                    # Save XAI data to session state for real-time XAI display
                    st.session_state.xai_data = {
                        "window_id": int(current_window_id),
                        "fingerprint": fingerprint,
                        "explanation": explanation_msg,
                        "confidence": confidence,
                        "z_scores": z_scores,
                        "deviant_features": deviant_features
                    }
            
            # Reset buffer for next window
            window_buffer = []
            current_window_id = win_id
            
        # Add current frame to buffer
        window_buffer.append(frame_meta)
        
        # If in single-frame mode, update metrics immediately
        if detection_mode == "Single-frame":
            plot_x.append(i)
            plot_y.append(frame_score)
            plot_colors.append('red' if is_frame_anomaly else '#00d2ff')
            
            if is_frame_anomaly:
                attacks_detected += 1
                if true_label == 0:
                    false_positives += 1
                    
        # Update metric displays
        total_placeholder.metric("Total Processed", f"{total_frames}")
        attacks_placeholder.metric("Attacks Detected", f"{attacks_detected}")
        fp_placeholder.metric("False Positives", f"{false_positives}")
        if use_adaptive_threshold:
            baselines_placeholder.metric("Adaptive Baselines Learned", f"{len(st.session_state.tracker.ema_mean)}")
        
        # Render Plotly Chart
        fig = go.Figure()
        if detection_mode == "Single-frame":
            fig.add_trace(go.Scatter(x=plot_x, y=plot_y, mode='lines', name='Anomaly Score', line=dict(color='#00d2ff', width=1.5)))
            
            # Decide threshold to use for plotting anomaly markers
            if use_adaptive_threshold and selected_model_name == "Lightweight Autoencoder":
                fig.add_trace(go.Scatter(x=plot_x, y=plot_thresh, mode='lines', name='Adaptive Threshold', line=dict(color='orange', width=1.5, dash='dash')))
                anomaly_indices = [idx for idx, val in enumerate(plot_y) if (val > plot_thresh[idx])]
            else:
                thresh = ae_threshold if selected_model_name == "Lightweight Autoencoder" else 0.5
                fig.add_hline(y=thresh, line_dash="dash", line_color="red", annotation_text="Threshold")
                anomaly_indices = [idx for idx, val in enumerate(plot_y) if (val > thresh)]
                
            # Add anomaly markers
            if anomaly_indices:
                fig.add_trace(go.Scatter(
                    x=[plot_x[idx] for idx in anomaly_indices],
                    y=[plot_y[idx] for idx in anomaly_indices],
                    mode='markers',
                    name='Flagged Anomaly',
                    marker=dict(color='red', size=6)
                ))
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='#0d1117',
                plot_bgcolor='#161b22',
                font=dict(color='#c9d1d9'),
                height=300,
                xaxis_title="Frame ID",
                yaxis_title="Anomaly Score",
                showlegend=False
            )
        else:
            # Ensemble Mode: line chart of rolling score (0-1)
            fig.add_trace(go.Scatter(x=plot_x, y=plot_y, mode='lines+markers', name='Ensemble Score', 
                                     line=dict(color='#8b949e', width=2),
                                     marker=dict(color=plot_colors, size=6)))
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (0.5)")
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='#0d1117',
                plot_bgcolor='#161b22',
                font=dict(color='#c9d1d9'),
                height=300,
                xaxis_title="Time Window (100ms)",
                yaxis_title="Ensemble Score (0-1)",
                yaxis=dict(range=[-0.05, 1.05]),
                showlegend=False
            )
            
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Print stream logs
        status_symbol = "🚨" if (is_frame_anomaly if detection_mode == "Single-frame" else (len(plot_y)>0 and plot_y[-1] > 0.5)) else "✅"
        log_color = "red" if (is_frame_anomaly if detection_mode == "Single-frame" else (len(plot_y)>0 and plot_y[-1] > 0.5)) else "green"
        log_text = f"<span style='color: {log_color}; font-family: monospace;'>{status_symbol} Frame #{i+1:03d} | CAN_ID: 0x{can_id} | DLC: {dlc} | Label: {attack_type} | Score: {frame_score:.6f}</span>"
        stream_logs.insert(0, log_text)
        if len(stream_logs) > 10:
            stream_logs.pop()
        console_html = "<br>".join(stream_logs)
        console_placeholder.markdown(f"**Live CAN Frames Log (latest on top):**<br>{console_html}", unsafe_allow_html=True)
        
        # Render XAI Expandable Panel below chart
        if detection_mode == "Window ensemble" and st.session_state.xai_data is not None:
            with xai_placeholder:
                xai = st.session_state.xai_data
                st.markdown(f"### 🚨 Explainable AI (XAI) Diagnosis (Window {xai['window_id']})")
                col_x1, col_x2 = st.columns([1, 1])
                with col_x1:
                    st.markdown(f"**Diagnosis:** `{xai['fingerprint']}` ({xai['explanation']})")
                    st.markdown(f"**Confidence Level:** `{xai['confidence']}`")
                    st.markdown(f"**Deviant Features (> 2.0 z-score):** {', '.join(xai['deviant_features'])}")
                with col_x2:
                    # Matplotlib bar chart of z-scores
                    fig_z, ax_z = plt.subplots(figsize=(6, 3.5))
                    fig_z.patch.set_facecolor('#0d1117')
                    ax_z.set_facecolor('#161b22')
                    ax_z.spines['bottom'].set_color('#30363d')
                    ax_z.spines['top'].set_color('#30363d')
                    ax_z.spines['left'].set_color('#30363d')
                    ax_z.spines['right'].set_color('#30363d')
                    ax_z.tick_params(colors='#c9d1d9', which='both', labelsize=8)
                    ax_z.xaxis.label.set_color('#c9d1d9')
                    ax_z.yaxis.label.set_color('#c9d1d9')
                    ax_z.title.set_color('#ffffff')
                    
                    colors_z = ['#d62728' if abs(xai['z_scores'][feat]) > 2.0 else '#8b949e' for feat in FEATURE_NAMES]
                    ax_z.barh(FEATURE_NAMES, [xai['z_scores'][feat] for feat in FEATURE_NAMES], color=colors_z)
                    ax_z.axvline(2.0, color='red', linestyle='--', alpha=0.7)
                    ax_z.axvline(-2.0, color='red', linestyle='--', alpha=0.7)
                    ax_z.set_title('Feature Z-Scores Anomaly Profile', fontsize=10, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_z)
                    plt.close()
                    
        # Delay
        time.sleep(0.01)
        
    st.success("Simulation completed successfully!")

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
    try:
        can_id = manual_id.strip().zfill(4)
        bytes_list = [b.strip() for b in manual_payload.split() if b.strip()]
        dlc = len(bytes_list)
        
        if dlc > 8:
            st.error("Payload cannot exceed 8 bytes.")
        else:
            dt = avg_delta_t
            freq = id_freqs.get(can_id, 1.0)
            dlc_std = id_dlc_stds.get(can_id, 1.0)
            
            counts = {}
            for b in bytes_list:
                counts[b] = counts.get(b, 0) + 1
            entropy = -sum((c / dlc) * np.log2(c / dlc) for c in counts.values()) if dlc > 0 else 0.0
            
            feat_vector = np.array([[dt, freq, entropy, dlc_std]])
            feat_scaled = scaler.transform(feat_vector)
            
            is_anomaly = False
            score = 0.0
            thresh = 0.0
            
            if selected_model_name == "Lightweight Autoencoder":
                with torch.no_grad():
                    frame_tensor = torch.FloatTensor(feat_scaled)
                    output = ae_model(frame_tensor)
                    score = torch.mean((frame_tensor - output) ** 2).item()
                if use_adaptive_threshold:
                    is_anomaly = st.session_state.tracker.is_anomaly(can_id, score, is_training=False)
                    mean_val = st.session_state.tracker.ema_mean[can_id]
                    std_val = (st.session_state.tracker.ema_var[can_id] ** 0.5)
                    thresh = mean_val + 3.0 * std_val
                else:
                    thresh = ae_threshold
                    is_anomaly = score > thresh
            else:
                score = -clf.score_samples(feat_scaled)[0]
                is_anomaly = clf.predict(feat_scaled)[0] == -1
                thresh = 0.5
                
            st.markdown("### Detection Result:")
            if is_anomaly:
                st.error(f"🚨 **ANOMALY DETECTED!** \n\n Anomaly Score: **{score:.6f}** (Threshold: {thresh:.6f})")
            else:
                st.success(f"✅ **NORMAL FRAME** \n\n Anomaly Score: **{score:.6f}** (Threshold: {thresh:.6f})")
                
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
