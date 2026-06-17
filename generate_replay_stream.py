import pandas as pd
import numpy as np
import os
import time

# Paths
FEATURES_CSV = "D:/Tata Innovent/CANomaly/features.csv"
REPLAY_CSV = "D:/Tata Innovent/CANomaly/outputs/replay_stream.csv"

def calc_entropy(bytes_list):
    if not bytes_list:
        return 0.0
    counts = {}
    for b in bytes_list:
        counts[b] = counts.get(b, 0) + 1
    total = len(bytes_list)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())

def calc_byte_variance(bytes_list):
    if not bytes_list:
        return 0.0
    # Convert hex strings to decimal integers
    vals = []
    for b in bytes_list:
        try:
            vals.append(int(b, 16))
        except ValueError:
            pass
    return np.var(vals) if vals else 0.0

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
    print("Generating custom replay stream...")
    # Load normal frames from features.csv
    df = pd.read_csv(FEATURES_CSV)
    df_normal = df[df['Label'] == 0].copy()
    
    # We need about 2000 normal frames
    df_normal = df_normal.head(2200).reset_index(drop=True)
    
    # We will build a list of frames as dictionaries
    frames = df_normal.to_dict('records')
    
    # Add default injected_attack_type = 'None'
    for f in frames:
        f['injected_attack_type'] = 'None'
        
    # Create 30 Fuzzy attack frames
    print("Creating 30 Fuzzy attack frames...")
    fuzzy_frames = []
    base_ts = frames[1400]['Timestamp']
    for i in range(30):
        # Random CAN ID in hex
        can_id = f"{np.random.randint(0x500, 0x7FF):03X}"
        dlc = np.random.randint(1, 9)
        # Random payload bytes
        payload = [f"{np.random.randint(0, 256):02X}" for _ in range(dlc)]
        # Add to matching data byte columns
        frame_dict = {
            'Timestamp': base_ts + i * 0.0001, # high frequency
            'CAN_ID': can_id,
            'DLC': dlc,
            'Flag': 'T',
            'Label': 1,
            'Attack_Type': 'Fuzzy',
            'injected_attack_type': 'Fuzzy'
        }
        # Fill payload bytes
        for idx, p in enumerate(payload):
            frame_dict[str(idx+3)] = p
        # Fill remaining columns up to 10 with NaN
        for idx in range(dlc, 8):
            frame_dict[str(idx+3)] = np.nan
            
        fuzzy_frames.append(frame_dict)
        
    # Create 20 RPM Spoofing frames
    print("Creating 20 RPM Spoofing frames...")
    rpm_frames = []
    # Normal interval of 0x0C4 is ~10ms. 3x normal means ~30ms interval.
    base_ts = frames[1700]['Timestamp']
    rpm_payload = ["00", "00", "10", "20", "00", "00", "00", "00"]
    for i in range(20):
        frame_dict = {
            'Timestamp': base_ts + i * 0.03, # 30ms interval (3x normal)
            'CAN_ID': '0C4', # Valid RPM ID
            'DLC': 8,
            'Flag': 'T',
            'Label': 1,
            'Attack_Type': 'RPM',
            'injected_attack_type': 'RPM_Spoofing'
        }
        for idx, p in enumerate(rpm_payload):
            frame_dict[str(idx+3)] = p
            
        rpm_frames.append(frame_dict)
        
    # Inject Fuzzy frames at index 1400
    print("Injecting Fuzzy frames at frame 1400...")
    combined_frames = frames[:1400] + fuzzy_frames + frames[1400:1700]
    
    # Adjust index for RPM injection
    combined_frames = combined_frames[:1730] + rpm_frames + combined_frames[1730:]
    
    # Create dataframe
    df_replay = pd.DataFrame(combined_frames)
    
    # Re-calculate timestamps to be strictly increasing and consistent
    print("Re-calculating timestamps and features...")
    # Base timestamp
    current_ts = df_replay.iloc[0]['Timestamp']
    timestamps = [current_ts]
    
    # Loop to assign sequential timestamps based on injection intervals
    for idx in range(1, len(df_replay)):
        attack = df_replay.iloc[idx]['injected_attack_type']
        if attack == 'Fuzzy':
            current_ts += 0.0001 # high frequency
        elif attack == 'RPM_Spoofing':
            current_ts += 0.0300 # 30ms interval (3x normal)
        else:
            current_ts += 0.0003 # typical average normal interval
        timestamps.append(current_ts)
        
    df_replay['Timestamp'] = timestamps
    
    # Re-calculate features to be mathematically accurate for this stream
    df_replay['delta_t'] = df_replay['Timestamp'].diff().fillna(0)
    df_replay['window'] = df_replay['Timestamp'] // 0.1
    df_replay['can_id_freq'] = df_replay.groupby(['window', 'CAN_ID'])['CAN_ID'].transform('count')
    df_replay['dlc_consistency'] = df_replay.groupby('CAN_ID')['DLC'].transform('std').fillna(0)
    
    # Compute payload entropy
    payload_cols = ['3', '4', '5', '6', '7', '8', '9', '10']
    entropies = []
    byte_variances = []
    for idx, row in df_replay.iterrows():
        dlc = int(row['DLC'])
        bytes_list = []
        for col in payload_cols:
            val = row[col]
            if pd.notna(val) and val != '' and str(val).lower() != 'nan':
                bytes_list.append(str(val))
        bytes_list = bytes_list[:dlc]
        entropies.append(calc_entropy(bytes_list))
        byte_variances.append(calc_byte_variance(bytes_list))
        
    df_replay['payload_entropy'] = entropies
    df_replay['payload_byte_variance'] = byte_variances
    
    # Inter-ID intervals
    df_replay['inter_id_interval'] = df_replay.groupby('CAN_ID')['Timestamp'].diff().fillna(0)
    
    # Group by window and compute window transition entropy, then map back to frames
    window_groups = df_replay.groupby('window')
    window_trans_entropy = {}
    for win, grp in window_groups:
        window_trans_entropy[win] = calc_transition_entropy(grp['CAN_ID'].tolist())
    df_replay['can_id_transition_entropy'] = df_replay['window'].map(window_trans_entropy)
    
    # Add frame_idx
    df_replay = df_replay.reset_index(drop=True)
    df_replay['frame_idx'] = df_replay.index
    
    # Select columns as requested
    cols_order = ['frame_idx', 'CAN_ID', 'injected_attack_type', 'delta_t', 'payload_entropy', 
                  'dlc_consistency', 'can_id_freq', 'payload_byte_variance', 'inter_id_interval', 
                  'can_id_transition_entropy', 'Label', 'Timestamp', 'DLC', 'Flag', 
                  '3', '4', '5', '6', '7', '8', '9', '10', 'Attack_Type']
                  
    # Clean column mapping
    df_replay = df_replay[cols_order]
    df_replay.rename(columns={'CAN_ID': 'can_id'}, inplace=True)
    
    df_replay.to_csv(REPLAY_CSV, index=False)
    print(f"Replay stream saved successfully to {REPLAY_CSV} (Shape: {df_replay.shape})")

if __name__ == "__main__":
    main()
