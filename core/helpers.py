import numpy as np
import os
import re
import pandas as pd
from core import config


BASE_DIR = os.path.join(config.DATA_DIR, "Preprocessed_VEP_Data")
DEVICES = config.DEVICES
LABELS = config.LABELS
fs = 2000


def load_signal(file):
    df = pd.read_csv(file, skiprows=1)[['Step 1', 'Chan 3']]
    
    # Drop the sub-header row
    df = df.drop(index=0).reset_index(drop=True)
    # Convert to numeric
    df = df.apply(pd.to_numeric)
    
    time = df['Step 1'].values
    signal = df['Chan 3'].values

    return time, signal


def load_preprocessed_signal(file):
    df = pd.read_csv(file)
    # No skiprows, no sub-header
    df = df[['Time', 'Signal']]
    signal = df['Signal'].values
    time = df['Time'].values
    return time, signal

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def load_blind_signal(base_dir=BASE_DIR, ext=".csv"):
    blind_dir = os.path.join(base_dir, "BLIND")
    if not os.path.exists(blind_dir):
        return []
    return sorted(
        [os.path.join(blind_dir, f) for f in os.listdir(blind_dir) if f.endswith(ext)],
        key=natural_key
    )


def process_file(filepath, delay=0, t_min=0, t_max=200, normalize=True):
    # 1) Extract pulsewidth from summary file
    # Get device and category from filepath Assuming structure: BASE_DIR / DEVICE / CATEGORY / filename.csv
    parts = os.path.normpath(filepath).split(os.sep)
    device = parts[-3]
    category = parts[-2]

    # Load summary file for this device & category
    summary_path = os.path.join(BASE_DIR, device, category, f"SNR_summary_{category}.csv")
    summary_df = pd.read_csv(summary_path)

    # Find matching file row in summary
    summary_df["FileName"] = summary_df["FileName"].astype(str).str.strip()
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    match = summary_df[summary_df["FileName"] == file_name]
    if match.empty:
        raise ValueError(f"No matching file found in summary for {file_name} ({device}/{category})")
    # Extract pulse width
    pulse_width = float(match["PulseWidth_ms"].iloc[0])

    # 2) Load and process raw data
    df = pd.read_csv(filepath, header=None, names=["time_ms", "ch3", "ch1"])
    idx_after_pulse = df.index[df["time_ms"] > pulse_width][0]

    # Align ch1 and ch3 so that they are zero after pulse
    df["ch1"] = df["ch1"] - df.loc[idx_after_pulse, "ch1"]
    df["ch3"] = df["ch3"] - df.loc[idx_after_pulse, "ch3"]
    df["ch3"] = df["ch3"].where(df["time_ms"] > pulse_width, 0)
    df["time_ms"] = df["time_ms"] - pulse_width + delay

    # Trim to [t_min, t_max]
    df_sliced = df[(df["time_ms"] >= t_min) & (df["time_ms"] <= t_max)].copy()
    time = df_sliced["time_ms"].values
    signal = df_sliced["ch3"].values

    # Normalize (zero mean, unit std)
    if normalize:
        signal = (signal - np.mean(signal)) / np.std(signal)
    return time, signal


def load_dataset_paths(base_dir=BASE_DIR, devices=DEVICES, labels=LABELS, ext=".csv"):
    """
    Returns a nested dictionary with all file paths per device and label.
    Example: paths["PRIMA_LE_DA"]["BC_Only"] list of CSV file paths
    Skips directories that don't exist.
    """
    paths = {}
    for device in devices:
        device_dir = os.path.join(base_dir, device)
        paths[device] = {}
        for label in labels:
            dir_path = os.path.join(device_dir, label)
            try:
                if not os.path.exists(dir_path):
                    print(f"Warning: Directory not found, skipping: {dir_path}")
                    paths[device][label] = []
                    continue
                    
                file_list = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if f.endswith(ext)
                ]
                paths[device][label] = file_list
                if len(file_list) == 0:
                    print(f"Warning: No {ext} files found in {dir_path}")
            except FileNotFoundError:
                print(f"Warning: Directory not found, skipping: {dir_path}")
                paths[device][label] = []
            except Exception as e:
                print(f"Warning: Error accessing {dir_path}: {e}")
                paths[device][label] = []
    return paths


def compute_average_signal(file_list, t_min=0, t_max=200, normalize=True, delay=0):
    all_times = []
    all_signals = []

    for filepath in file_list:
        time, signal = process_file(filepath, delay=delay, t_min=t_min, t_max=t_max, normalize=normalize)
        if len(signal) > 0:
            all_times.append(time)
            all_signals.append(signal)

    # Assumes all time vectors are identical
    if len(all_times) == 0:
        return np.array([]), np.array([])
    
    avg_time = all_times[0]
    signals_matrix = np.stack(all_signals, axis=1)
    avg_signal = np.mean(signals_matrix, axis=1)
    return avg_time, avg_signal


def get_data(device="PRIMA_LE_DA", classes=3, labels=None):
    if labels is None:
        labels = ["BC_Only", "BC_and_RGC"] if classes == 2 else ["BC_Only", "RGC_Only", "BC_and_RGC"]

    all_paths_preprocessed = load_dataset_paths(
        base_dir=BASE_DIR,
        devices=DEVICES,
        labels=labels,
    )

    X, y, raw_X, file_list = [], [], [], []

    # BC_Only
    for file in all_paths_preprocessed[device].get("BC_Only", []):
        try:
            _, signal = load_preprocessed_signal(file)
            X.append(signal)
            y.append("BC_Only")
            file_list.append(file)
        except Exception as e:
            print(f"Warning: Error processing {file}: {e}")

    # BC_and_RGC
    for file in all_paths_preprocessed[device].get("BC_and_RGC", []):
        try:
            _, signal = load_preprocessed_signal(file)
            X.append(signal)
            y.append("BC_and_RGC")
            file_list.append(file)
        except Exception as e:
            print(f"Warning: Error processing {file}: {e}")

    # RGC_Only (only if 3 classes)
    if classes == 3:
        for file in all_paths_preprocessed[device].get("RGC_Only", []):
            try:
                _, signal = load_preprocessed_signal(file)
                X.append(signal)
                y.append("RGC_Only")
                file_list.append(file)
            except Exception as e:
                print(f"Warning: Error processing {file}: {e}")

    return X, y, raw_X, file_list


def parse_filename_info(filename):
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    device = parts[0] if parts[0] not in ("LE", "ML") else (parts[0] if parts[1].isdigit() else parts[1])
    try:
        animal = int(parts[2] if parts[0] == "RCS" else (parts[1] if parts[1].isdigit() else parts[2]))
    except (ValueError, IndexError):
        animal = None
    return animal, device