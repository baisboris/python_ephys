import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_ind

# File path
file_path = 'your_file_path_here.csv'  # Replace with your actual file path

# Extract base directory
base_dir = os.path.dirname(file_path)

# Create subfolder for outputs
save_dir = os.path.join(base_dir, "03_Sensory_Stimuli")
os.makedirs(save_dir, exist_ok=True)

# Load CSV file
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Extract signals
time = df["Time(ms)"] / 1000  # Convert to seconds
primary = df["Primary"] * 100  # Voltage in mV
secondary = df["Secondary"] * 1000  # Current in pA
ecg = df["ECG"]
airpuff = df["AIRPUFF"]

# Seaborn styling
sns.set_style("darkgrid")
sns.set_palette("colorblind")

### 🔹 1. Detect Resting Membrane Potential (RMP)
# Detect RMP using a 100 ms window instead of 500 ms
sampling_interval = time.iloc[1] - time.iloc[0]  # in seconds
window_duration_sec = 0.1  # 100 ms
window_size = int(window_duration_sec / sampling_interval)

# Compute rolling mean
rolling_mean = primary.rolling(window=window_size).mean()

# Drop NaNs and get minimum mean value (as an estimate of RMP)
valid_rolling = rolling_mean.dropna()
if not valid_rolling.empty:
    min_rmp_value = valid_rolling.min()
    print(f"\n🔹 Estimated Resting Membrane Potential (RMP): {min_rmp_value:.2f} mV")
else:
    print("\n⚠️ Rolling mean window too large for signal. Could not estimate RMP.")


### 🔹 2. Detect Spikes Across Entire Recording
spike_threshold = 0  # Spikes go well above this
peaks, _ = find_peaks(primary, height=spike_threshold, distance=10)  # Add `prominence` if needed

# Spike count and firing rate
n_spikes = len(peaks)
duration = time.iloc[-1] - time.iloc[0]
firing_rate = n_spikes / duration if duration > 0 else 0

print(f"\n🔹 Total Number of Spikes: {n_spikes}")
print(f"🔹 Average Firing Rate: {firing_rate:.2f} Hz (over entire recording)")

### 🔹 3. Save Spike Detection Plot
plt.figure(figsize=(12, 6))
sns.lineplot(x=time, y=primary, label="Voltage (mV)")
sns.scatterplot(x=time.iloc[peaks], y=primary.iloc[peaks], color="red", label="Detected Spikes")

plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Full Spike Detection Across Recording")
plt.legend()
spike_fig_path = os.path.join(save_dir, "Full_Spike_Detection.png")
plt.savefig(spike_fig_path, dpi=300)
plt.close()

print(f"✅ Spike detection figure saved: {spike_fig_path}")
print("✅ All updated analyses completed successfully!")

from scipy.ndimage import gaussian_filter1d  # Add this at the top with your imports

def analyze_ap_features(time, voltage, peaks, current, window_ms=50, smooth=True):
    from scipy.ndimage import gaussian_filter1d
    sampling_interval = time.iloc[1] - time.iloc[0]
    window_pts = int(window_ms / 1000 / sampling_interval)
    features = []

    for peak_idx in peaks:
        # Classify as spontaneous or evoked
        stim_value = current.iloc[peak_idx]
        label = "spontaneous" if abs(stim_value) < 5 else "evoked"

        # Define window
        start_idx = max(peak_idx - window_pts, 0)
        end_idx = min(peak_idx + window_pts, len(voltage))
        t_win = time.iloc[start_idx:end_idx].reset_index(drop=True)
        v_win = voltage.iloc[start_idx:end_idx].reset_index(drop=True)

        # Smoothing (optional)
        v_win_proc = gaussian_filter1d(v_win, sigma=1) if smooth else v_win
        dvdt = np.gradient(v_win_proc, sampling_interval)

        try:
            peak_slope = dvdt.max()
            slope_thresh = peak_slope * 0.05
            thresh_idx_candidates = np.where(dvdt[:window_pts] > slope_thresh)[0]
            thresh_idx = thresh_idx_candidates[0] if len(thresh_idx_candidates) > 0 else None
            ap_thresh = v_win[thresh_idx] if thresh_idx is not None else np.nan
        except:
            ap_thresh = np.nan
            thresh_idx = None

        ap_peak = v_win.max()
        ap_peak_idx = v_win.idxmax()
        ap_min = v_win.min()
        ap_amp = ap_peak - ap_min
        max_slope = dvdt.max()

        half_height = ap_min + (ap_amp / 2)
        above_half = np.where(v_win > half_height)[0]
        fwhm = (above_half[-1] - above_half[0]) * sampling_interval * 1000 if len(above_half) > 1 else np.nan

        post_peak_v = v_win[ap_peak_idx + 1:]
        ahp = post_peak_v.min() if not post_peak_v.empty else np.nan

        features.append({
            "APVmin": ap_min,
            "APVpeak": ap_peak,
            "APVthresh": ap_thresh,
            "APVslope": max_slope,
            "APVhalf": fwhm,
            "APVamp": ap_amp,
            "AHP": ahp,
            "type": label,
            "peak_index": peak_idx  # Keep for plotting later
        })

    return pd.DataFrame(features)


### 🧠 Call function and summarize
# 🔍 Analyze AP features or prepare empty structure if no spikes
if len(peaks) == 0:
    print("⚠️ No spikes detected — creating empty AP dataframe with all expected columns.")
    ap_df = pd.DataFrame({
        "APVmin": pd.Series(dtype=float),
        "APVpeak": pd.Series(dtype=float),
        "APVthresh": pd.Series(dtype=float),
        "APVslope": pd.Series(dtype=float),
        "APVhalf": pd.Series(dtype=float),
        "APVamp": pd.Series(dtype=float),
        "AHP": pd.Series(dtype=float),
        "type": pd.Series(dtype=str),
        "peak_index": pd.Series(dtype=int),
        "burst": pd.Series(dtype=bool),
        "sustained_firing": pd.Series(dtype=bool),
        "stimulus_coupling": pd.Series(dtype=str),
        "airpuff_latency (s)": pd.Series(dtype=float),
        "APfreqmax": pd.Series(dtype=float)
    })
else:
    ap_df = analyze_ap_features(time, primary, peaks, secondary)



if "APVhalf" in ap_df.columns:
    ap_df = ap_df[ap_df["APVhalf"] <= 5]
else:
    print("⚠️ No 'APVhalf' column found — skipping AP width filter.")


### 🔥 Detect Burst Firing
burst_isi_thresh = 0.015  # 15 ms
spike_times = time.iloc[peaks].values
isis = np.diff(spike_times)

# Find indices of spikes where ISI < threshold (suggesting burst)
burst_indices = np.where(isis < burst_isi_thresh)[0]

# Each burst involves the spike at burst_indices[i] and burst_indices[i+1]
burst_starts = peaks[burst_indices]
burst_ends = peaks[burst_indices + 1]
burst_peaks = np.unique(np.concatenate([burst_starts, burst_ends]))

# ✅ Make sure 'peak_index' column exists
if "peak_index" not in ap_df.columns:
    ap_df["peak_index"] = pd.Series(dtype=int)

# ✅ Then compute burst info (if needed)
ap_df["burst"] = ap_df["peak_index"].isin(burst_peaks)


print(f"💥 Burst spikes detected: {ap_df['burst'].sum()} out of {len(ap_df)} APs")


burst_csv_path = os.path.join(save_dir, "Burst_AP_Features.csv")
ap_df[ap_df["burst"]].to_csv(burst_csv_path, index=False)
print(f"💾 Burst spike table saved: {burst_csv_path}")


### 🔥 Detect Active Epochs (Sustained Firing Events)
window_size_sec = 1.0     # 1-second sliding window
min_spikes_per_window = 4  # at least 4 spikes to call it an "active epoch"
step_size_sec = 0.1        # slide every 100 ms

spike_times = time.iloc[peaks].values
window_starts = np.arange(time.iloc[0], time.iloc[-1] - window_size_sec, step_size_sec)

active_epochs = []
for start in window_starts:
    end = start + window_size_sec
    n_spikes = np.sum((spike_times >= start) & (spike_times < end))
    if n_spikes >= min_spikes_per_window:
        active_epochs.append((start, end))

# Merge overlapping/adjacent windows
merged_epochs = []
for start, end in active_epochs:
    if not merged_epochs or start > merged_epochs[-1][1]:
        merged_epochs.append([start, end])
    else:
        merged_epochs[-1][1] = max(merged_epochs[-1][1], end)

# Label APs that fall within any epoch
in_epoch = []
for idx, row in ap_df.iterrows():
    spike_time = time.iloc[int(row["peak_index"])]
    in_epoch.append(any(start <= spike_time <= end for start, end in merged_epochs))

ap_df["sustained_firing"] = in_epoch
print(f"📈 Detected {sum(in_epoch)} spikes within sustained firing epochs.")

plt.figure(figsize=(12, 4))
plt.plot(time, primary, label="Vm", alpha=0.6)
for start, end in merged_epochs:
    plt.axvspan(start, end, color='orange', alpha=0.2, label="Sustained Firing")

if "peak_index" not in ap_df.columns:
    ap_df["peak_index"] = pd.Series(dtype=int)


# Safety checks for missing columns
if "peak_index" not in ap_df.columns:
    ap_df["peak_index"] = pd.Series(dtype=int)
if "sustained_firing" not in ap_df.columns:
    ap_df["sustained_firing"] = False

# ✅ Ensure safe plotting even if DataFrame is empty
if "peak_index" not in ap_df.columns:
    ap_df["peak_index"] = pd.Series(dtype=int)
if "sustained_firing" not in ap_df.columns:
    ap_df["sustained_firing"] = False

# 🧠 Only plot sustained firing spikes if they exist
if not ap_df[ap_df["sustained_firing"]].empty:
    plt.scatter(
        time.iloc[ap_df[ap_df["sustained_firing"]]["peak_index"]],
        ap_df[ap_df["sustained_firing"]]["APVpeak"],
        color="purple", s=10, label="Sustained Firing"
    )
    plt.scatter(
        time.iloc[ap_df[ap_df["sustained_firing"]]["peak_index"]],
        primary.iloc[ap_df[ap_df["sustained_firing"]]["peak_index"]],
        color='red', s=10, label="Sustained Spikes"
    )
else:
    print("⚠️ No sustained firing spikes to plot.")




# ✅ Ensure required columns exist even if no spikes are detected
if "peak_index" not in ap_df.columns:
    ap_df["peak_index"] = pd.Series(dtype=int)

for col in ["APVpeak", "APVmin", "APVthresh", "APVslope", "APVhalf", "APVamp", "AHP"]:
    if col not in ap_df.columns:
        ap_df[col] = pd.Series(dtype=float)


### 🔹 4. Airpuff Onset Detection & Coupling Analysis

# Detect airpuff TTL onsets
airpuff_threshold = 0.5  # adjust as needed
airpuff_onsets = np.where(np.diff((airpuff > airpuff_threshold).astype(int)) == 1)[0]
airpuff_times = time.iloc[airpuff_onsets]

print(f"\n💨 Detected {len(airpuff_times)} airpuff stimuli.")


# Set post-stimulus coupling window (e.g., 100 ms)
coupling_window = 1  # in seconds

# Analyze each AP's timing relative to airpuffs
is_coupled = []
latencies = []

for i, row in ap_df.iterrows():
    spike_time = time.iloc[int(row["peak_index"])]
    delta_times = spike_time - airpuff_times
    valid_deltas = delta_times[(delta_times >= 0) & (delta_times <= coupling_window)]

    if not valid_deltas.empty:
        is_coupled.append("airpuff-coupled")
        latencies.append(valid_deltas.min())
    else:
        is_coupled.append("non-coupled")
        latencies.append(np.nan)

ap_df["stimulus_coupling"] = is_coupled
ap_df["airpuff_latency (s)"] = latencies

# Compute coupling stats
n_coupled = sum(label == "airpuff-coupled" for label in is_coupled)
response_prob = n_coupled / len(airpuff_times) if len(airpuff_times) > 0 else 0
print(f"🔗 Coupled APs: {n_coupled} / {len(airpuff_times)} → Response probability: {response_prob:.2f}")

###  Add Optional Raster Plot (PSTH-style)

plt.figure(figsize=(10, 2))
for stim_time in airpuff_times:
    plt.axvline(stim_time, color='gray', linestyle='--', alpha=0.5)

coupled_spikes = ap_df[ap_df["stimulus_coupling"] == "airpuff-coupled"]
plt.scatter(time.iloc[coupled_spikes["peak_index"]], np.ones(len(coupled_spikes)), 
            color='red', s=10, label="Coupled Spikes")

plt.xlabel("Time (s)")
plt.yticks([])
plt.title("Airpuff-Coupled Spike Raster")
plt.legend()
plt.tight_layout()
raster_path = os.path.join(save_dir, "Airpuff_Coupled_Raster.png")
plt.savefig(raster_path, dpi=300)
plt.close()
print(f"🧪 Raster plot saved: {raster_path}")

###  Visualize to Confirm

plt.figure(figsize=(12, 3))
plt.plot(time, primary, label="Vm")
for stim_time in airpuff_times:
    plt.axvline(stim_time, color="gray", linestyle="--", alpha=0.6)
    plt.axvspan(stim_time, stim_time + coupling_window, color="blue", alpha=0.1)
plt.scatter(time.iloc[ap_df[ap_df["stimulus_coupling"] == "airpuff-coupled"]["peak_index"]],
            ap_df[ap_df["stimulus_coupling"] == "airpuff-coupled"]["APVpeak"],
            color="red", s=10, label="Coupled Spikes")
plt.xlabel("Time (s)")
plt.ylabel("Vm (mV)")
plt.title("Airpuff-Coupled Spike Visualization")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Airpuff_Coupled_Window_Check.png"), dpi=300)
plt.close()


# Add APfreqmax (Hz)
if len(peaks) >= 2:
    isi = np.diff(time.iloc[peaks])  # inter-spike intervals (s)
    ap_freq_max = 1 / isi.min()
else:
    ap_freq_max = 0

ap_df["APfreqmax"] = ap_freq_max

# Save table to CSV
ap_csv_path = os.path.join(save_dir, "AP_Feature_Table.csv")
# Rename columns with units
ap_df_with_units = ap_df.rename(columns={
    "APVmin": "APVmin (mV)",
    "APVpeak": "APVpeak (mV)",
    "APVthresh": "APVthresh (mV)",
    "APVslope": "APVslope (mV/s)",
    "APVhalf": "APVhalf (ms)",
    "APVamp": "APVamp (mV)",
    "AHP": "AHP (mV)",
    "APfreqmax": "APfreqmax (Hz)",
    "type": "Type",
    "peak_index": "Peak Index",
    "stimulus_coupling": "Airpuff Coupling",
    "airpuff_latency (s)": "Airpuff Latency (s)"
})

# Save table to CSV with headers that include units
ap_df_with_units.to_csv(ap_csv_path, index=False)
print(f"\n📈 Full AP feature table (with airpuff analysis) saved: {ap_csv_path}")

# Plot the first spike (if exists)
if not ap_df.empty:
    first_peak = peaks[0]
    w = int(0.05 / sampling_interval)  # 50 ms window
    idx_range = slice(max(first_peak - w, 0), min(first_peak + w, len(primary)))

    plt.figure(figsize=(8, 4))
    plt.plot(time.iloc[idx_range], primary.iloc[idx_range], label="AP")
    plt.axhline(ap_df["APVthresh"].iloc[0], ls="--", label="Threshold")
    plt.axhline(ap_df["APVpeak"].iloc[0], ls="--", label="Peak")
    plt.axhline(ap_df["APVmin"].iloc[0], ls="--", label="Min")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Example AP with Features")
    plt.legend()
    plt.tight_layout()
    ap_plot_path = os.path.join(save_dir, "Single_AP_Feature_Annotated.png")
    plt.savefig(ap_plot_path, dpi=300)
    plt.close()
    print(f"🖼️ Annotated AP plot saved: {ap_plot_path}")

def plot_aligned_aps(time, voltage, ap_df, save_dir, window_ms=50):
    sampling_interval = time.iloc[1] - time.iloc[0]
    window_pts = int(window_ms / 1000 / sampling_interval)
    t_centered = np.linspace(-window_ms/2, window_ms/2, 2*window_pts)

    plt.figure(figsize=(10, 5))

    for i, row in ap_df.iterrows():
        peak_idx = int(row["peak_index"])
        start_idx = max(peak_idx - window_pts, 0)
        end_idx = min(peak_idx + window_pts, len(voltage))
        v_segment = voltage.iloc[start_idx:end_idx].reset_index(drop=True)
        label = row["type"]
        color = "blue" if label == "spontaneous" else "orange"
        plt.plot(t_centered[:len(v_segment)], v_segment, color=color, alpha=0.4)

    plt.axhline(0, linestyle="--", color="black", linewidth=0.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Aligned Action Potentials")
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "All_APs_Aligned.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🧬 Aligned AP plot saved: {plot_path}")

plot_aligned_aps(time, primary, ap_df, save_dir)

### 🔹 Subthreshold Response Averaging Aligned to Airpuff

# Parameters
pre_stim = 0.05   # 50 ms before airpuff
post_stim = 1  # 500 ms after
window_length = pre_stim + post_stim
sampling_interval = time.iloc[1] - time.iloc[0]
window_pts = int(window_length / sampling_interval)

# Build matrix of subthreshold traces
subthreshold_trials = []

for stim_time in airpuff_times:
    stim_idx = np.argmin(np.abs(time - stim_time))
    start_idx = stim_idx - int(pre_stim / sampling_interval)
    end_idx = stim_idx + int(post_stim / sampling_interval)

    if start_idx < 0 or end_idx >= len(primary):
        continue  # Skip incomplete trials at edges

    trace = primary.iloc[start_idx:end_idx].reset_index(drop=True)

    # Check if there's a spike in this window
    spikes_in_window = [p for p in peaks if start_idx <= p < end_idx]
    if len(spikes_in_window) == 0:
        subthreshold_trials.append(trace)

# Stack trials and compute mean ± SEM
if subthreshold_trials:
    sub_mat = np.vstack(subthreshold_trials)
    mean_response = np.mean(sub_mat, axis=0)
    sem_response = np.std(sub_mat, axis=0) / np.sqrt(sub_mat.shape[0])
    time_window = np.linspace(-pre_stim, post_stim, len(mean_response))

    # Plot all trials + mean
    plt.figure(figsize=(8, 4))
    for trial in sub_mat:
        plt.plot(time_window * 1000, trial, color='lightgray', alpha=0.3)

    plt.plot(time_window * 1000, mean_response, label="Mean Vm", color='blue')
    plt.fill_between(time_window * 1000, mean_response - sem_response, mean_response + sem_response,
                     alpha=0.3, color='blue')
    plt.axvline(0, color='gray', linestyle='--', label='Airpuff')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title(f"Subthreshold Average Response (n={len(subthreshold_trials)} trials)")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, "Subthreshold_Average_Response.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📉 Subthreshold average plot saved: {plot_path}")

    # Save to CSV
    sub_response_df = pd.DataFrame({
        "Time (ms)": time_window * 1000,
        "Mean Vm (mV)": mean_response,
        "SEM (mV)": sem_response
    })
    csv_path = os.path.join(save_dir, "Subthreshold_Average_Response.csv")
    sub_response_df.to_csv(csv_path, index=False)
    print(f"📄 Subthreshold average data saved: {csv_path}")

else:
    print("⚠️ No subthreshold trials found (all airpuff windows contain spikes).")

### 🔹 Subthreshold Response: Early vs Late Trial Comparison

# Reuse same parameters
half_point = len(airpuff_times) // 2
airpuff_early = airpuff_times[:half_point]
airpuff_late = airpuff_times[half_point:]

# --- Replace compute_subthreshold_average with baseline-corrected version ---
def compute_subthreshold_average(airpuff_subset, label):
    traces = []
    for stim_time in airpuff_subset:
        stim_idx = np.argmin(np.abs(time - stim_time))
        start_idx = stim_idx - int(pre_stim / sampling_interval)
        end_idx = stim_idx + int(post_stim / sampling_interval)

        if start_idx < 0 or end_idx >= len(primary):
            continue

        trace = primary.iloc[start_idx:end_idx].reset_index(drop=True)

        # Check for spikes
        spikes_in_window = [p for p in peaks if start_idx <= p < end_idx]
        if len(spikes_in_window) == 0:
            # ✅ Subtract baseline (−50 to 0 ms)
            baseline_pts = int(pre_stim / sampling_interval)
            baseline_value = trace[:baseline_pts].mean()
            trace_corrected = trace - baseline_value
            traces.append(trace_corrected)

    if traces:
        mat = np.vstack(traces)
        mean_resp = np.mean(mat, axis=0)
        sem_resp = np.std(mat, axis=0) / np.sqrt(mat.shape[0])
        time_win = np.linspace(-pre_stim, post_stim, len(mean_resp))

        # Save all individual traces to CSV
        trace_df = pd.DataFrame(mat.T, columns=[f"Trial_{i+1}" for i in range(mat.shape[0])])
        trace_df.insert(0, "Time (ms)", time_win * 1000)
        csv_file_all = os.path.join(save_dir, f"Subthreshold_{label}_Traces.csv")
        trace_df.to_csv(csv_file_all, index=False)
        print(f"💾 All subthreshold traces saved: {csv_file_all}")

        # Plot all trials + mean
        plt.figure(figsize=(8, 4))
        for trial in mat:
            plt.plot(time_win * 1000, trial, color='lightgray', alpha=0.3)

        plt.plot(time_win * 1000, mean_resp, label=f"Mean Vm ({label})", color='blue' if label == 'early' else 'green')
        plt.fill_between(time_win * 1000, mean_resp - sem_resp, mean_resp + sem_resp, alpha=0.3,
                         color='blue' if label == 'early' else 'green')
        plt.axvline(0, color='gray', linestyle='--', label='Airpuff')
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.title(f"Subthreshold Avg ({label.title()} Trials), n={len(traces)}")
        plt.legend()
        plt.tight_layout()

        # Save figure
        plot_file = os.path.join(save_dir, f"Subthreshold_{label}_Response.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"🧪 Subthreshold plot saved: {plot_file}")

        # Save data
        df = pd.DataFrame({
            "Time (ms)": time_win * 1000,
            f"Mean Vm ({label})": mean_resp,
            f"SEM ({label})": sem_resp
        })
        csv_file = os.path.join(save_dir, f"Subthreshold_{label}_Response.csv")
        df.to_csv(csv_file, index=False)
        print(f"📄 CSV saved: {csv_file}")

        return mat  # return matrix for stats

    else:
        print(f"⚠️ No subthreshold {label} trials (all had spikes or incomplete).")
        return None


# --- Baseline-corrected overall subthreshold average ---
subthreshold_trials = []

for stim_time in airpuff_times:
    stim_idx = np.argmin(np.abs(time - stim_time))
    start_idx = stim_idx - int(pre_stim / sampling_interval)
    end_idx = stim_idx + int(post_stim / sampling_interval)

    if start_idx < 0 or end_idx >= len(primary):
        continue  # Skip incomplete trials

    trace = primary.iloc[start_idx:end_idx].reset_index(drop=True)

    spikes_in_window = [p for p in peaks if start_idx <= p < end_idx]
    if len(spikes_in_window) == 0:
        baseline_pts = int(pre_stim / sampling_interval)
        baseline_value = trace[:baseline_pts].mean()
        trace_corrected = trace - baseline_value
        subthreshold_trials.append(trace_corrected)

if subthreshold_trials:
    sub_mat = np.vstack(subthreshold_trials)
    mean_response = np.mean(sub_mat, axis=0)
    sem_response = np.std(sub_mat, axis=0) / np.sqrt(sub_mat.shape[0])
    time_window = np.linspace(-pre_stim, post_stim, len(mean_response))

    plt.figure(figsize=(8, 4))
    for trial in sub_mat:
        plt.plot(time_window * 1000, trial, color='lightgray', alpha=0.3)

    plt.plot(time_window * 1000, mean_response, label="Mean Vm", color='blue')
    plt.fill_between(time_window * 1000, mean_response - sem_response, mean_response + sem_response,
                     alpha=0.3, color='blue')
    plt.axvline(0, color='gray', linestyle='--', label='Airpuff')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title(f"Subthreshold Average Response (n={len(subthreshold_trials)} trials)")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "Subthreshold_Average_Response.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📉 Subthreshold average plot saved: {plot_path}")

    sub_response_df = pd.DataFrame({
        "Time (ms)": time_window * 1000,
        "Mean Vm (mV)": mean_response,
        "SEM (mV)": sem_response
    })
    csv_path = os.path.join(save_dir, "Subthreshold_Average_Response.csv")
    sub_response_df.to_csv(csv_path, index=False)
    print(f"📄 Subthreshold average data saved: {csv_path}")
else:
    print("⚠️ No subthreshold trials found (all airpuff windows contain spikes).")


# --- Run and compare early vs late with stats ---
mat_early = compute_subthreshold_average(airpuff_early, "early")
mat_late = compute_subthreshold_average(airpuff_late, "late")

if mat_early is not None and mat_late is not None:
    # Compare peak response in post-stimulus window (e.g., 0–100 ms)
    start_pt = int(pre_stim / sampling_interval)
    end_pt = start_pt + int(0.1 / sampling_interval)

    peak_early = mat_early[:, start_pt:end_pt].min(axis=1)
    peak_late = mat_late[:, start_pt:end_pt].min(axis=1)

    t_stat, p_value = ttest_ind(peak_early, peak_late)

    print(f"📊 Statistical test (early vs late response peak): p = {p_value:.4f}")

# 🔹 Plot ALL individual Vm responses to airpuffs (with or without spikes)
pre_stim = 0.05  # 50 ms before
post_stim = 1  # 1000 ms after
sampling_interval = time.iloc[1] - time.iloc[0]
pre_pts = int(pre_stim / sampling_interval)
post_pts = int(post_stim / sampling_interval)
t_segment = np.linspace(-pre_stim, post_stim, pre_pts + post_pts)

for i, stim_time in enumerate(airpuff_times):
    stim_idx = np.argmin(np.abs(time - stim_time))
    start_idx = stim_idx - pre_pts
    end_idx = stim_idx + post_pts

    if start_idx < 0 or end_idx >= len(primary):
        continue

    v_segment = primary.iloc[start_idx:end_idx].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_segment[:len(v_segment)], v_segment, color="black", linewidth=1)
    ax.axvline(0, color="red", linestyle="--", label="Airpuff")
    ax.set_title(f"Airpuff Trial {i+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Membrane Voltage (mV)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    trial_path = os.path.join(save_dir, f"Airpuff_Voltage_Trial_{i+1:02d}.png")
    fig.savefig(trial_path, dpi=300)
    plt.close(fig)
