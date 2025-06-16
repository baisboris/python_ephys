import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from scipy.integrate import simps

# File path
file_path = "E:\\DATA\\24_BRAINS_8_9_1JB\\20250521_CD1_24008_58bruker\\VoltageRecording-01072025-1149-007\\VoltageRecording-01072025-1149-007_Cycle00001_VoltageRecording_001.csv"

# Extract base directory
base_dir = os.path.dirname(file_path)

# Create subfolders for outputs
save_dir = os.path.join(base_dir, "01_Sensory_LFP")
os.makedirs(save_dir, exist_ok=True)

save_dir_25 = os.path.join(base_dir, "01_Sensory_LFP_25")
os.makedirs(save_dir_25, exist_ok=True)

# Load CSV file
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Extract signals
time = df["Time(ms)"] / 1000  # Convert to seconds
primary = df["Primary"] * 100  # LFP signal, in mV
airpuff = df["AIRPUFF"]

# Seaborn styling
sns.set_style("darkgrid")
sns.set_palette("colorblind")

# === FILTERING ===
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def apply_notch_filter(data, notch_freq, fs, quality=30):
    b, a = iirnotch(notch_freq, quality, fs)
    return filtfilt(b, a, data)

### 🔹 Airpuff Onset Detection

# Detect airpuff TTL onsets
airpuff_threshold = 0.5
airpuff_onsets = np.where(np.diff((airpuff > airpuff_threshold).astype(int)) == 1)[0]
airpuff_times = time.iloc[airpuff_onsets]

print(f"\n💨 Detected {len(airpuff_times)} airpuff stimuli.")

# Common parameters
sampling_interval = time.iloc[1] - time.iloc[0]
fs = 1 / sampling_interval
lfp_pre = 0.5
lfp_post = 5.0
cutoff_freq = 100
notch_freq = 60
artifact_window_ms = (0, 15)  # artifact detection window (in ms)
quant_window = (15, 100)  # post-stimulus window for quantification (in ms)


def process_groups(group_size, save_directory):
    n_groups = len(airpuff_times) // group_size
    summary = []

    for group_idx in range(n_groups):
        start_i = group_idx * group_size
        end_i = start_i + group_size
        group_airpuffs = airpuff_times[start_i:end_i]

        group_traces = []
        for i, stim_time in enumerate(group_airpuffs):
            stim_idx = np.argmin(np.abs(time - stim_time))
            start_idx = stim_idx - int(lfp_pre / sampling_interval)
            end_idx = stim_idx + int(lfp_post / sampling_interval)

            if start_idx < 0 or end_idx >= len(primary):
                continue

            trace = primary.iloc[start_idx:end_idx].reset_index(drop=True)
            baseline = trace[:int(lfp_pre / sampling_interval)].median()
            trace_corrected = trace - baseline
            trace_filtered = butter_lowpass_filter(trace_corrected, cutoff=cutoff_freq, fs=fs)
            trace_filtered = apply_notch_filter(trace_filtered, notch_freq=notch_freq, fs=fs)

            time_vector = np.linspace(-lfp_pre, lfp_post, len(trace_filtered)) * 1000

            # Detect peaks within artifact window and blank them
            artifact_mask = (time_vector >= artifact_window_ms[0]) & (time_vector <= artifact_window_ms[1])
            artifact_segment = trace_filtered[artifact_mask]
            peak_indices, _ = find_peaks(np.abs(artifact_segment), height=0.8)
            if len(peak_indices) > 0:
                trace_filtered[artifact_mask] = np.nan

            group_traces.append(trace_filtered)

        if group_traces:
            group_mat = np.vstack(group_traces)
            mean_trace = np.nanmean(group_mat, axis=0)
            sem_trace = np.nanstd(group_mat, axis=0) / np.sqrt(np.sum(~np.isnan(group_mat), axis=0))
            time_window = np.linspace(-lfp_pre, lfp_post, group_mat.shape[1]) * 1000

            # Quantification metrics
            quant_mask = (time_window >= quant_window[0]) & (time_window <= quant_window[1])
            peak_amp = np.nanmax(np.abs(mean_trace[quant_mask]))
            latency_idx = np.nanargmax(np.abs(mean_trace[quant_mask]))
            latency = time_window[quant_mask][latency_idx]
            auc = simps(np.abs(mean_trace[quant_mask]), time_window[quant_mask])

            summary.append({
                "Group": f"Airpuffs {start_i+1}–{end_i}",
                "Peak Amplitude (mV)": peak_amp,
                "Latency to Peak (ms)": latency,
                "AUC (mV*ms)": auc
            })

            plt.figure(figsize=(10, 4))
            for trace in group_mat:
                plt.plot(time_window, trace, color='gray', alpha=0.3)

            plt.plot(time_window, mean_trace, color='blue', label='Mean LFP')
            plt.fill_between(time_window, mean_trace - sem_trace, mean_trace + sem_trace,
                             alpha=0.3, color='blue')
            plt.axvline(0, color='red', linestyle='--', label='Airpuff Onset')
            plt.axvspan(*artifact_window_ms, color='red', alpha=0.1, label='Blanked')
            plt.xlabel("Time (ms)")
            plt.ylabel("LFP (mV)")
            plt.title(f"LFP Overlay: Airpuffs {start_i+1}–{end_i}")
            plt.legend()
            plt.tight_layout()

            overlay_path = os.path.join(save_directory, f"LFP_Overlay_Airpuff_{start_i+1}_{end_i}.png")
            plt.savefig(overlay_path, dpi=300)
            plt.close()
            print(f"🧠 LFP overlay saved: {overlay_path}")
        else:
            print(f"⚠️ Skipping group {group_idx+1} — not enough valid trials.")

    # Save quantification summary
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(save_directory, "LFP_Quantification_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"📊 Quantification summary saved: {summary_path}")

# Process every 10 stimuli into original folder
process_groups(10, save_dir)

# Process every 25 stimuli into new folder
process_groups(25, save_dir_25)
