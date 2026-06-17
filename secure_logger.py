import hashlib
import json
import time
import os

class HashChainLogger:
    def __init__(self, log_path="outputs/event_log.jsonl"):
        self.log_path = log_path
        # Ensure parent directory exists
        dir_name = os.path.dirname(self.log_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.prev_hash = self.get_last_hash()

    def get_last_hash(self):
        """Read the last entry from the log file to resume the hash chain, or return genesis hash."""
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            return "0" * 64
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                position = f.tell()
                line = b""
                # Read backwards to find the last non-empty line
                while position > 0:
                    position -= 1
                    f.seek(position)
                    char = f.read(1)
                    if char == b"\n" and line:
                        break
                    if char != b"\n":
                        line = char + line
                
                if line:
                    entry = json.loads(line.decode("utf-8").strip())
                    return entry.get("entry_hash", "0" * 64)
        except Exception:
            pass
        return "0" * 64

    def log_event(self, event_dict):
        event_dict = event_dict.copy()
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = time.time()
        event_dict["prev_hash"] = self.prev_hash
        entry_str = json.dumps(event_dict, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        event_dict["entry_hash"] = entry_hash
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict) + "\n")
        self.prev_hash = entry_hash
        return entry_hash

def verify_chain(log_path="outputs/event_log.jsonl"):
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return True
    
    expected_prev_hash = "0" * 64
    with open(log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                print(f"[Verification Failed] Line {line_num} contains invalid JSON.")
                return False
            
            # Check link in chain
            stored_prev_hash = entry.get("prev_hash")
            if stored_prev_hash != expected_prev_hash:
                print(f"[Verification Failed] Line {line_num}: prev_hash ({stored_prev_hash}) does not match expected ({expected_prev_hash}).")
                return False
            
            # Recompute hash of the content
            stored_entry_hash = entry.get("entry_hash")
            if not stored_entry_hash:
                print(f"[Verification Failed] Line {line_num}: entry_hash is missing.")
                return False
                
            event_to_hash = entry.copy()
            if "entry_hash" in event_to_hash:
                del event_to_hash["entry_hash"]
                
            recomputed_str = json.dumps(event_to_hash, sort_keys=True)
            recomputed_hash = hashlib.sha256(recomputed_str.encode()).hexdigest()
            
            if recomputed_hash != stored_entry_hash:
                print(f"[Verification Failed] Line {line_num}: Recomputed hash ({recomputed_hash}) does not match stored entry_hash ({stored_entry_hash}).")
                return False
                
            expected_prev_hash = stored_entry_hash
            
    return True
