import pandas as pd
import numpy as np

def check_uds_frame(row, maintenance_window_active=False):
    """
    Checks if a single CAN frame represents an unauthorized UDS reprogramming/diagnostic session.
    
    Service IDs of interest (ISO 14229):
      - 0x10: DiagnosticSessionControl
      - 0x34: RequestDownload
      - 0x36: TransferData
      - 0x37: RequestTransferExit
    
    Diagnostic CAN IDs of interest:
      - 0x7DF: Functional request
      - 0x7E0 - 0x7E7: Physical request
    """
    can_id_str = str(row.get('can_id', row.get('CAN_ID', ''))).strip().upper()
    try:
        id_val = int(can_id_str, 16)
    except ValueError:
        return {"suspicious": False, "reason": None}

    # UDS request IDs
    is_uds_id = (id_val == 0x7DF) or (0x7E0 <= id_val <= 0x7E7)
    if not is_uds_id:
        return {"suspicious": False, "reason": None}

    # Extract payload bytes from columns '3' to '10'
    payload_cols = [str(i) for i in range(3, 11)]
    payload_bytes = []
    for col in payload_cols:
        val = row.get(col)
        if val is not None and val != '' and str(val).lower() != 'nan':
            try:
                if isinstance(val, str):
                    payload_bytes.append(int(val, 16))
                else:
                    payload_bytes.append(int(val))
            except ValueError:
                pass

    if not payload_bytes:
        return {"suspicious": False, "reason": "Empty payload"}

    uds_sids = {0x10, 0x34, 0x36, 0x37}
    
    # In ISO-TP, Byte 0 is usually PCI, Byte 1 is Service ID (SID)
    # Check first 2 bytes to be robust to direct framing or ISO-TP wrapping
    found_sid = None
    for idx in range(min(len(payload_bytes), 2)):
        if payload_bytes[idx] in uds_sids:
            found_sid = payload_bytes[idx]
            break

    if found_sid is None:
        return {"suspicious": False, "reason": None}

    if not maintenance_window_active:
        sid_hex = f"0x{found_sid:02X}"
        return {
            "suspicious": True,
            "reason": f"UDS service {sid_hex} detected outside maintenance window on CAN ID 0x{id_val:03X}"
        }

    return {"suspicious": False, "reason": None}

def scan_stream_for_uds_anomalies(df, maintenance_window_active=False):
    """
    Scans a stream (DataFrame) of CAN frames and flags unauthorized UDS sessions.
    """
    flagged_indices = []
    reasons = []
    
    for idx, row in df.iterrows():
        res = check_uds_frame(row, maintenance_window_active)
        if res["suspicious"]:
            flagged_indices.append(idx)
            reasons.append(res["reason"])
            
    if flagged_indices:
        start_frame = min(flagged_indices)
        end_frame = max(flagged_indices)
        print(f"SUMS-aligned check: flagged unauthorized reflash attempt at frame {start_frame}-{end_frame}")
        return {
            "detected": True,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "reasons": list(set(reasons))
        }
    else:
        return {
            "detected": False,
            "start_frame": None,
            "end_frame": None,
            "reasons": []
        }
