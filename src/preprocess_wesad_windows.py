# src/preprocess_wesad_windows.py
import os, pickle, numpy as np, pandas as pd
from scipy.signal import find_peaks

WESAD_PATH = "WESAD"
OUT = "wesad_window_features.csv"
WINDOW_SECONDS = 30   # choose 10/30/60
STEP_SECONDS = 15     # overlap

def safe_load_pickle(path):
    with open(path,"rb") as f:
        try: return pickle.load(f, encoding="latin1")
        except TypeError:
            f.seek(0); return pickle.load(f)

def windowed(arr, fs, win_s, step_s):
    if arr is None: return []
    step = int(step_s * fs)
    win = int(win_s * fs)
    out = []
    # return list of (start_index, segment)
    for start in range(0, max(1, len(arr)-win+1), max(1,step)):
        out.append((start, arr[start:start+win]))
    return out

def extract_features_from_seg(seg):
    seg = np.asarray(seg).ravel()
    if len(seg)==0:
        return {"mean":np.nan,"std":np.nan,"max":np.nan}
    return {"mean":float(np.nanmean(seg)), "std":float(np.nanstd(seg)), "max":float(np.nanmax(seg))}

rows = []
if not os.path.exists(WESAD_PATH):
    print(f"Directory {WESAD_PATH} not found.")
else:
    for subj in sorted(os.listdir(WESAD_PATH)):
        subj_dir = os.path.join(WESAD_PATH, subj)
        if not os.path.isdir(subj_dir): continue
        pkl = os.path.join(subj_dir, subj + ".pkl")
        if not os.path.exists(pkl): continue
        try:
            data = safe_load_pickle(pkl)
        except Exception as e:
            print(f"Error loading {pkl}: {e}")
            continue

        sig = data.get("signal", {})
        # try wrist, then chest for SIGNALS
        if "wrist" in sig:
            sensor = sig["wrist"]
        elif "chest" in sig:
            sensor = sig["chest"]
        else:
            sensor = sig
            
        fs_bvp = 64
        fs_eda = 4
        bvp = sensor.get("BVP") if isinstance(sensor, dict) else None
        eda = sensor.get("EDA") if isinstance(sensor, dict) else None
        
        # Labels are usually 700Hz
        labels = None
        for k in ("label","labels","y"):
            if k in data:
                labels = np.asarray(data[k]).ravel()
                break
        
        if bvp is None or len(bvp) == 0:
            continue

        # Get windows with indices
        bvp_windows = windowed(bvp, fs_bvp, WINDOW_SECONDS, STEP_SECONDS) # list of (idx, data)
        eda_windows = windowed(eda, fs_eda, WINDOW_SECONDS, STEP_SECONDS) # list of (idx, data)

        # Assuming start times align for BVP/EDA, we use BVP indices to map to labels
        # Calculate ratio: label_len / bvp_len
        ratio = 1.0
        if labels is not None and len(labels) > 0:
             ratio = len(labels) / len(bvp)
        
        n = max(len(bvp_windows), len(eda_windows))
        for i in range(n):
            b_start, bw = bvp_windows[i] if i < len(bvp_windows) else (0, [])
            e_start, ew = eda_windows[i] if i < len(eda_windows) else (0, [])
            
            # Skip if empty
            if len(bw) == 0: continue

            feats = {}
            # bvp features
            b = extract_features_from_seg(bw)
            feats.update({f"bvp_{k}":v for k,v in b.items()})
            # eda features
            e = extract_features_from_seg(ew)
            feats.update({f"eda_{k}":v for k,v in e.items()})
            
            # label mapping
            lbl = np.nan
            if labels is not None:
                # Map BVP index range to Label index range
                # BVP window is [b_start, b_start + len(bw)]
                lbl_start = int(b_start * ratio)
                lbl_end = int((b_start + len(bw)) * ratio)
                # Clamp
                lbl_start = min(lbl_start, len(labels))
                lbl_end = min(lbl_end, len(labels))
                
                lbl_seg = labels[lbl_start:lbl_end]
                if len(lbl_seg) > 0:
                    vals, cnts = np.unique(lbl_seg, return_counts=True)
                    lbl = int(vals[np.argmax(cnts)])
            
            feats.update({"subject":subj, "label":lbl})
            rows.append(feats)
        print(f"Processed {subj}, windows: {n}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("Saved", OUT, "shape:", df.shape)
