from core.helpers import load_dataset_paths
from core import config
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import shutil
import pywt


BASE_DIR = os.path.join(config.DATA_DIR, "Labelled_VEP_Data")
OUTPUT_DIR = os.path.join(config.DATA_DIR, "Preprocessed_VEP_Data")
DEVICES = config.DEVICES
LABELS = config.LABELS
BLIND_NAME = "BLIND"
SNR_THRESHOLD = 1.0


def dwt_downsampling(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    downsampled_signal = np.asarray(coeffs[0])
    return downsampled_signal


def preprocess_signal(file, normalize=True, do_artifact_removal=False, do_dwt_downsampling=True, tmin=0, tmax=400, dwt_level=4):
    time, ch1, ch3 = load_signal_both_channels(file)
    stim_dur, irradiance = extract_PulseWidth_SignalPower(file)
    if do_artifact_removal:
        time_clean, ch3_clean = artifact_removal(ch1, ch3, time, t_min=tmin, stim_dur=stim_dur)
    else:
        time_clean = time
        ch3_clean = ch3
    ch3_avg, time_avg = average_two_phases(ch3_clean, time_clean)
    snr = compute_vep_snr(time_avg, ch3_avg)
    time_trimmed, ch3_trimmed = trim(ch3_avg, time_avg, t_min=tmin, t_max=tmax)
    if do_dwt_downsampling:
        ch3_dwt = dwt_downsampling(ch3_trimmed, wavelet="db4", level=dwt_level)
        time_dwt = np.linspace(time_trimmed[0], time_trimmed[-1], len(ch3_dwt))
    else:
        ch3_dwt = ch3_trimmed
        time_dwt = time_trimmed
    if normalize:
        ch3_norm = normalize_signal(ch3_dwt)
    else:
        ch3_norm = ch3_dwt
    return time_dwt, ch3_norm, snr



def load_signal_both_channels(file):
    """Load both Channel 1 and Channel 3 from CSV file"""
    df = pd.read_csv(file, skiprows=1)[['Step 1', 'Chan 1', 'Chan 3']]
    
    # Drop the sub-header row
    df = df.drop(index=0).reset_index(drop=True)
    # Convert to numeric
    df = df.apply(pd.to_numeric)
    
    time = df['Step 1'].values
    ch1 = df['Chan 1'].values
    ch3 = df['Chan 3'].values

    # ch1 at t=0 should be 0
    ch1 = ch1 - ch1[time == 0][0]
    ch3 = ch3 - ch3[time == 0][0]
    return time, ch1, ch3

def average_two_phases(signal, time):
    N = len(signal) // 2
    # split signal
    sig1 = signal[:N]
    sig2 = signal[N:2*N]
    # average
    signal_avg = (sig1 + sig2) / 2
    time_avg   = time[:N]     
    return signal_avg, time_avg


def extract_PulseWidth_SignalPower(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    
    # FIND PART THAT CONTAINS "ms"
    pulsePart = None
    pulseIndex = None
    try: 
        for i, p in enumerate(parts):
            if "ms" in p.lower():
                pulsePart = p
                pulseIndex = i
                break
        if pulseIndex is not None and pulseIndex > 0 and parts[pulseIndex - 1] == '0':
            # combine "0" and "5ms" → "0_5ms"
            pulsePart = f"{parts[pulseIndex - 1]}_{parts[pulseIndex]}"

        # remove "ms"
        pulsePart = pulsePart.replace("ms", "")

        # replace "_" with "." → matches MATLAB strrep
        pulsePart = pulsePart.replace("_", ".")
        pulseWidth = float(pulsePart)
    except Exception as e:
        pulseWidth = 10.0
        print(f"Could not parse pulse width for file: {filename}, using default 10 ms")

    try:
        # Extract irradiance / power
        # Find the part that contains "mWmm2"
        signalPart = None
        for p in parts:
            if "mWmm2" in p:
                signalPart = p.replace("mWmm2", "")
                break
        
        if signalPart is None:
            raise ValueError("No mWmm2 found")
            
        signalPower = float(signalPart)
    except (ValueError, AttributeError):
        signalPower = 0  # is not used anyways 
        
    return pulseWidth, signalPower


def artifact_removal(ch1, ch3, time, t_min=0, stim_dur=10):
    """Remove artifacts by scaling and subtracting Channel 1 from Channel 3"""
    mask = (time >= t_min) & (time <= stim_dur)
    
    if not np.any(mask):
        return time, ch3
    
    ch3_min = np.min(ch3[mask])
    ch1_min = np.min(-ch1[mask])
    
    if abs(ch1_min) < 1e-9:
        return time, ch3
    
    ch1_scaled = ch1 * (ch3_min / ch1_min)
    ch1_scaled_neg = np.where(ch1_scaled > 0, ch1_scaled, 0)
    ch3_corrected = ch3 + ch1_scaled_neg
    
    if not np.all(np.isfinite(ch3_corrected)):
        return time, ch3
    
    return time, ch3_corrected


def trim(signal, time, t_min=None, t_max=200):
    if t_min is None:
        t_min = time[0]
    mask = (time >= t_min) & (time <= t_max)
    return time[mask], signal[mask]

def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


def compute_vep_snr(time, signal, signal_window=100, noise_center=450, noise_window=100):
    sig_mask = (time >= 0) & (time <= signal_window)

    half_w = noise_window / 2
    noise_start = noise_center - half_w
    noise_end = noise_center + half_w
    noise_mask = (time >= noise_start) & (time <= noise_end)

    sig_segment = signal[sig_mask]
    noise_segment = signal[noise_mask]

    # Peak-to-peak amplitudes
    signal_p2p = np.max(sig_segment) - np.min(sig_segment)
    noise_p2p = np.max(noise_segment) - np.min(noise_segment)

    # SNR
    snr = signal_p2p / noise_p2p if noise_p2p != 0 else np.inf
    return snr


def preprocess_save_all(base_dir=BASE_DIR, 
        output_dir=OUTPUT_DIR, 
        devices=DEVICES, 
        labels=LABELS, 
        normalize=True,
        do_artifact_removal=False, 
        do_dwt_downsampling=True,
        tmin=0,
        tmax=400, 
        SNR_filtering=True,
        SNR_threshold=SNR_THRESHOLD, 
        include_blind=False):

    # Create main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- counters ---
    excluded_files = 0
    excluded_stats = defaultdict(lambda: defaultdict(int))
    excluded_blind = 0

    # ---------- LABELED DATA ----------
    all_paths_raw = load_dataset_paths(
        base_dir=base_dir,
        devices=devices,
        labels=labels
    )

    for device, label_dict in all_paths_raw.items():
        # Only delete the specific device folder being processed
        device_output_dir = os.path.join(output_dir, device)
        if os.path.exists(device_output_dir):
            shutil.rmtree(device_output_dir)
            print(f"Removed existing folder: {device_output_dir}")
        
        for label, file_list in label_dict.items():
            out_dir = os.path.join(output_dir, device, label)
            os.makedirs(out_dir, exist_ok=True)

            for file_path in file_list:
                time, signal, snr = preprocess_signal(
                    file_path,
                    tmin=tmin,
                    tmax=tmax,
                    normalize=normalize,
                    do_artifact_removal=do_artifact_removal,
                    do_dwt_downsampling=do_dwt_downsampling,
                )

                # ---- SNR FILTERING ----
                if SNR_filtering and snr < SNR_threshold:
                    excluded_files += 1
                    excluded_stats[device][label] += 1
                    continue

                out_path = os.path.join(out_dir, os.path.basename(file_path))
                pd.DataFrame({"Time": time, "Signal": signal}).to_csv(out_path, index=False)

    # ---------- BLIND DATA ----------
    blind_dir = os.path.join(base_dir, BLIND_NAME)

    if include_blind and os.path.exists(blind_dir):
        # Only delete blind folder if we're including blind data
        out_blind = os.path.join(output_dir, BLIND_NAME)
        if os.path.exists(out_blind):
            shutil.rmtree(out_blind)
            print(f"Removed existing folder: {out_blind}")
        
        os.makedirs(out_blind, exist_ok=True)

        for file in os.listdir(blind_dir):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(blind_dir, file)
            time, signal, snr = preprocess_signal(
                file_path,
                tmin=tmin,
                tmax=tmax,
                normalize=normalize,
                do_artifact_removal=do_artifact_removal,
                do_dwt_downsampling=do_dwt_downsampling,
            )

            if SNR_filtering and snr < SNR_threshold:
                excluded_files += 1
                excluded_blind += 1
                continue

            out_path = os.path.join(out_blind, file)
            pd.DataFrame({"Time": time, "Signal": signal}).to_csv(out_path, index=False)

    # ---------- REPORT ----------
    print("\nExcluded files due to low SNR")
    print(f"Total excluded (<{SNR_threshold}): {excluded_files}")

    for device in excluded_stats:
        if excluded_stats[device]:
            device_total = sum(excluded_stats[device].values())
            print(f"\n  {device}: {device_total}")
            for label in excluded_stats[device]:
                print(f"    {label}: {excluded_stats[device][label]}")
        else:
            print(f"\n  {device}: 0")

    if include_blind:
        print(f"\n  {BLIND_NAME}: {excluded_blind}")