import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objs as go
import plotly.offline as pyo

# File path
file_path = 'your_file_path_here.csv'  # Replace with your actual file path

# Extract directory path to save figures
base_dir = os.path.dirname(file_path)
save_dir = os.path.join(base_dir, "02_Spontaneous_Firing")
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

# Restrict analysis to the first X seconds
"""X = 20  # Change this to however many seconds you want
time_mask = (time <= X)

# Apply the mask to all relevant signals
time = time[time_mask]
primary = primary[time_mask]
secondary = secondary[time_mask]
ecg = ecg[time_mask]
airpuff = airpuff[time_mask]"""


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
    # Save RMP to CSV
    rmp_df = pd.DataFrame({"Estimated RMP (mV)": [min_rmp_value]})
    rmp_csv_path = os.path.join(save_dir, "Estimated_RMP.csv")
    rmp_df.to_csv(rmp_csv_path, index=False)
    print(f"📄 RMP value saved to: {rmp_csv_path}")
else:
    print("\n⚠️ Rolling mean window too large for signal. Could not estimate RMP.")

### 🔹 2. Detect Spikes Only During Spontaneous (No Current Injection) and Above 0 mV
no_current_mask = secondary.abs() < 5  # ±5 pA threshold
primary_no_current = primary[no_current_mask]
time_no_current = time[no_current_mask]

spike_threshold = -9.9  # mV
min_peak_distance = int(0.002 / sampling_interval)  # 2 ms
peaks_filtered, _ = find_peaks(primary_no_current, height=spike_threshold, distance=min_peak_distance)

# Additional filtering to avoid detecting same spike multiple times
filtered_peaks = []
last_peak_time = -np.inf
min_interval_sec = 0.003  # 3 ms
for idx in peaks_filtered:
    current_time = time_no_current.iloc[idx]
    if current_time - last_peak_time > min_interval_sec:
        filtered_peaks.append(idx)
        last_peak_time = current_time

original_indices = primary_no_current.index[filtered_peaks]

# Spike count and firing rate
n_spikes = len(original_indices)
duration = time.iloc[-1] - time.iloc[0]
firing_rate = n_spikes / duration if duration > 0 else 0

print(f"\n🔹 Total Number of Spikes: {n_spikes}")
print(f"🔹 Average Firing Rate: {firing_rate:.2f} Hz (over entire recording)")

### 🔹 3. Save Spike Detection Plot
spike_trace = go.Scatter(x=time, y=primary, mode='lines', name='Voltage (mV)')
spike_markers = go.Scatter(x=time.iloc[original_indices], y=primary.iloc[original_indices],
                           mode='markers', name='Detected Spikes', marker=dict(color='red', size=6))

layout = go.Layout(title="Full Spike Detection (Interactive)",
                   xaxis=dict(title='Time (s)'),
                   yaxis=dict(title='Voltage (mV)'))

fig = go.Figure(data=[spike_trace, spike_markers], layout=layout)
html_path = os.path.join(save_dir, "Full_Spike_Detection_Interactive.html")
pyo.plot(fig, filename=html_path, auto_open=False)

print(f"✅ Interactive spike detection plot saved: {html_path}")
print("✅ All updated analyses completed successfully!")

def analyze_ap_features(time, voltage, peaks, current, window_ms=50, smooth=True):
    sampling_interval = time.iloc[1] - time.iloc[0]
    window_pts = int(window_ms / 1000 / sampling_interval)
    features = []

    for peak_idx in peaks:
        stim_value = current.iloc[peak_idx]
        label = "spontaneous" if abs(stim_value) < 5 else "evoked"

        start_idx = max(peak_idx - window_pts, 0)
        end_idx = min(peak_idx + window_pts, len(voltage))
        t_win = time.iloc[start_idx:end_idx].reset_index(drop=True)
        v_win = voltage.iloc[start_idx:end_idx].reset_index(drop=True)

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
            "peak_index": peak_idx
        })

    return pd.DataFrame(features)

ap_df = analyze_ap_features(time, primary, original_indices, secondary)

if len(original_indices) >= 2:
    isi = np.diff(time.iloc[original_indices])
    ap_freq_max = 1 / isi.min()
else:
    ap_freq_max = 0

ap_df["APfreqmax"] = ap_freq_max

ap_csv_path = os.path.join(save_dir, "AP_Feature_Table.csv")
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
    "peak_index": "Peak Index"
})
ap_df_with_units.to_csv(ap_csv_path, index=False)
print(f"\n📈 Action Potential feature table (with units) saved: {ap_csv_path}")

if not ap_df.empty:
    first_peak = original_indices[0]
    w = int(0.05 / sampling_interval)
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


### 🔹 4. Inter-Spike Interval (ISI) Distribution
if len(original_indices) > 1:
    spike_times = time.iloc[original_indices].values
    isi = np.diff(spike_times)
    plt.figure(figsize=(8, 4))
    sns.histplot(isi * 1000, bins=30, kde=True)  # Convert to ms
    plt.xlabel("Inter-Spike Interval (ms)")
    plt.ylabel("Count")
    plt.title("ISI Distribution")
    isi_fig_path = os.path.join(save_dir, "ISI_Distribution.png")
    plt.tight_layout()
    plt.savefig(isi_fig_path, dpi=300)
    plt.close()
    print(f"📊 ISI histogram saved: {isi_fig_path}")
else:
    print("⚠️ Not enough spikes to calculate ISI distribution.")