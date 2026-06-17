import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import gc

# Set style for premium aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
DATA_DIR = "D:/Tata Innovent/CANomaly/data"
OUTPUTS_DIR = "D:/Tata Innovent/CANomaly/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Datasets files and their attack names
DATASET_FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "Gear": "gear_dataset.csv",
    "RPM": "RPM_dataset.csv"
}

def calc_entropy_str(payload_str):
    bytes_list = [b for b in payload_str.split('_') if b != '' and b != 'nan']
    if not bytes_list:
        return 0.0
    counts = {}
    for b in bytes_list:
        counts[b] = counts.get(b, 0) + 1
    total = len(bytes_list)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())

def process_file(attack_name, filename, sample_size=250000):
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\nProcessing {attack_name} dataset: {filename}...")
    
    t0 = time.time()
    # Force 12 columns using names=range(12) to avoid tokenization errors due to variable DLC
    df = pd.read_csv(filepath, header=None, names=range(12))
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    
    # Extract the Flag column in a vectorized way based on DLC
    # The Flag is always at column index DLC + 3
    t_flag = time.time()
    df['Flag'] = 'R'
    for dlc_val in df[2].unique():
        if pd.isna(dlc_val):
            continue
        dlc_int = int(dlc_val)
        idx = df[2] == dlc_val
        df.loc[idx, 'Flag'] = df.loc[idx, dlc_int + 3]
    print(f"Extracted Flag in {time.time() - t_flag:.2f}s")
    
    # Rename key columns for clarity
    df.rename(columns={0: 'Timestamp', 1: 'CAN_ID', 2: 'DLC'}, inplace=True)
    
    # Show original distribution
    dist = df['Flag'].value_counts()
    print(f"Original distribution:\n{dist}")
    
    # 1. Inter-message arrival time (delta timestamp)
    df['delta_t'] = df['Timestamp'].diff().fillna(0)
    
    # 2. CAN ID frequency per window (100ms)
    df['window'] = df['Timestamp'] // 0.1
    df['can_id_freq'] = df.groupby(['window', 'CAN_ID'])['CAN_ID'].transform('count')
    
    # 3. DLC consistency per CAN ID (standard dev of DLC)
    df['dlc_consistency'] = df.groupby('CAN_ID')['DLC'].transform('std').fillna(0)
    
    # 3b. Physical plausibility score (RPM and Gear Range Checks)
    df['physical_plausibility_score'] = 0.0
    
    # RPM Plausibility Check (CAN ID 0316)
    idx_316 = df['CAN_ID'] == '0316'
    if idx_316.any():
        b2 = df.loc[idx_316, 5].fillna('00').astype(str).apply(lambda x: int(x, 16) if x != '' and x != 'nan' else 0)
        b3 = df.loc[idx_316, 6].fillna('00').astype(str).apply(lambda x: int(x, 16) if x != '' and x != 'nan' else 0)
        rpm = b3 * 256 + b2
        delta_rpm = rpm.diff().abs().fillna(0)
        implausible_rpm = (delta_rpm > 400) | (rpm < 1500) | (rpm > 4000)
        df.loc[idx_316, 'physical_plausibility_score'] = implausible_rpm.astype(float)
        
    # Gear Plausibility Check (CAN ID 043f)
    idx_43f = df['CAN_ID'] == '043f'
    if idx_43f.any():
        b9 = df.loc[idx_43f, 9].fillna('00').astype(str).apply(lambda x: int(x, 16) if x != '' and x != 'nan' else 0)
        invalid_range = (b9 < 7) | (b9 > 13)
        delta_gear = b9.diff().abs().fillna(0)
        implausible_gear_change = delta_gear > 1
        implausible_gear = invalid_range | implausible_gear_change
        df.loc[idx_43f, 'physical_plausibility_score'] = implausible_gear.astype(float)
    
    # Stratified downsampling to preserve R/T ratio
    normal_df = df[df['Flag'] == 'R']
    attack_df = df[df['Flag'] == 'T']
    
    n_normal = int(sample_size * len(normal_df) / len(df))
    n_attack = sample_size - n_normal
    
    sampled_normal = normal_df.sample(n=n_normal, random_state=42)
    sampled_attack = attack_df.sample(n=n_attack, random_state=42)
    
    sampled_df = pd.concat([sampled_normal, sampled_attack]).sort_values(by='Timestamp')
    print(f"Sampled {len(sampled_df):,} rows (Normal: {n_normal:,}, Attack: {n_attack:,})")
    
    # Clean memory by deleting large df before computing entropy on the sample
    del df
    del normal_df
    del attack_df
    gc.collect()
    
    # 4. Compute Payload byte entropy ONLY on the sample (saves huge CPU time and RAM)
    t_ent = time.time()
    payload_cols = [3, 4, 5, 6, 7, 8, 9, 10]
    payload_df = sampled_df[payload_cols].fillna('').astype(str)
    
    sampled_df['payload_str'] = ''
    for dlc_val in sampled_df['DLC'].unique():
        dlc_int = int(dlc_val)
        idx = sampled_df['DLC'] == dlc_val
        cols = list(range(3, 3 + dlc_int))
        if cols:
            payload_series = payload_df.loc[idx, cols[0]].copy()
            for col in cols[1:]:
                payload_series = payload_series + '_' + payload_df.loc[idx, col]
            sampled_df.loc[idx, 'payload_str'] = payload_series
        else:
            sampled_df.loc[idx, 'payload_str'] = ''
            
    unique_payloads = sampled_df['payload_str'].unique()
    entropy_map = {p: calc_entropy_str(p) for p in unique_payloads}
    sampled_df['payload_entropy'] = sampled_df['payload_str'].map(entropy_map)
    
    # Drop intermediate column
    sampled_df.drop(columns=['payload_str'], inplace=True)
    print(f"Computed payload entropy for sample in {time.time() - t_ent:.2f}s")
    
    # Add labels and attack type
    sampled_df['Attack_Type'] = sampled_df['Flag'].apply(lambda x: 'Normal' if x == 'R' else attack_name)
    sampled_df['Label'] = sampled_df['Flag'].apply(lambda x: 0 if x == 'R' else 1)
    
    # Keep only the necessary columns to save features.csv disk size and memory
    # Keep Timestamp, CAN_ID, DLC and payload bytes (3..10) for the Streamlit dashboard simulation
    cols_to_keep = ['Timestamp', 'CAN_ID', 'DLC', 3, 4, 5, 6, 7, 8, 9, 10, 
                    'delta_t', 'can_id_freq', 'payload_entropy', 'dlc_consistency', 'physical_plausibility_score', 'Flag', 'Attack_Type', 'Label']
    # Filter columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in sampled_df.columns]
    
    return sampled_df[cols_to_keep]

def main():
    combined_list = []
    
    for attack_name, filename in DATASET_FILES.items():
        processed_df = process_file(attack_name, filename, sample_size=250000)
        combined_list.append(processed_df)
        
    print("\nCombining datasets...")
    combined_df = pd.concat(combined_list, ignore_index=True)
    
    # Let's save the processed features
    features_csv_path = "D:/Tata Innovent/CANomaly/features.csv"
    combined_df.to_csv(features_csv_path, index=False)
    print(f"\nSaved combined features to {features_csv_path} (Shape: {combined_df.shape})")
    
    # Print overall distribution
    print("\nOverall Class Distribution in combined dataset:")
    print(combined_df['Attack_Type'].value_counts())
    print("\nLabel Distribution (0: Normal, 1: Attack):")
    print(combined_df['Label'].value_counts(normalize=True))
    
    # Generate Plots
    print("\nGenerating plots...")
    
    # 1. Histogram of inter-arrival times (delta_t)
    plt.figure(figsize=(10, 6))
    # Filter extreme values for visualization (delta_t < 20ms)
    delta_t_clean = combined_df[combined_df['delta_t'] < 0.02]['delta_t']
    plt.hist(delta_t_clean, bins=100, color='#1f77b4', edgecolor='none', alpha=0.8)
    plt.title('Histogram of CAN Inter-Message Arrival Times (delta_t < 20ms)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Inter-Arrival Time (seconds)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    hist_path = os.path.join(OUTPUTS_DIR, "inter_arrival_histogram.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved histogram to {hist_path}")
    
    # 2. CAN ID frequency heatmap
    plt.figure(figsize=(12, 8))
    # We will compute time bins (e.g. 5-second bins) on a subset of 100k messages to make a clean heatmap
    heatmap_df = combined_df.sample(n=100000, random_state=42).copy()
    heatmap_df['time_bin'] = ((heatmap_df['Timestamp'] - heatmap_df['Timestamp'].min()) // 5).astype(int)
    
    # Pivot to get CAN ID vs Time Bin frequency count
    pivot_df = heatmap_df.pivot_table(index='CAN_ID', columns='time_bin', values='delta_t', aggfunc='count', fill_value=0)
    
    # Select top 25 CAN IDs by frequency to keep it readable
    top_can_ids = pivot_df.sum(axis=1).nlargest(25).index
    pivot_df = pivot_df.loc[top_can_ids]
    
    # Plot heatmap using matplotlib imshow
    im = plt.imshow(pivot_df.values, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, label='Message Count (per 5s)')
    plt.yticks(ticks=np.arange(len(top_can_ids)), labels=top_can_ids)
    plt.title('CAN ID Message Frequency Heatmap over Time (Top 25 CAN IDs)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time Bin (5-second intervals)', fontsize=12)
    plt.ylabel('CAN ID (Hex)', fontsize=12)
    plt.grid(False) # Turn off grid lines for heatmap
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUTS_DIR, "can_id_frequency_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved frequency heatmap to {heatmap_path}")
    
    print("\nSTEP 1 Completed successfully!")

if __name__ == "__main__":
    main()
