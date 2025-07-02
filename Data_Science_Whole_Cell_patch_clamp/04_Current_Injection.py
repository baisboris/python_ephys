import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.cm as cm
import seaborn as sns
import random

# Load data
file_path = "your_file_path_here.csv"  # Replace with your actual file path
save_dir = os.path.join(os.path.dirname(file_path), "04_current_injection")
os.makedirs(save_dir, exist_ok=True)

# Read CSV and clean headers
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Extract signals
time = df["Time(ms)"] / 1000  # seconds
primary = df["Primary"] * 100  # mV
secondary = df["Secondary"] * 2000  # pA

# Parameters
threshold = 9  # pA change to detect a step
min_duration = 0.01  # 10 ms duration
pre_window = 0.05  # 50 ms before step
post_window = 0.05  # 50 ms after step
train_gap_threshold = 30  # seconds between steps to start a new train

# Initializations
baseline_current = np.median(secondary[:100])
start_idx = None
step_count = 0
train_count = 1
last_step_time = None

# Spike count collector
all_spike_counts = []

# Prepare first train directory
train_dir = os.path.join(save_dir, f"Train_{train_count:02d}")
os.makedirs(train_dir, exist_ok=True)

# Loop through the current trace
for i in range(1, len(secondary)):
    current_diff = np.abs(secondary.iloc[i] - baseline_current)

    if current_diff > threshold:
        if start_idx is None:
            start_idx = i
    else:
        if start_idx is not None:
            duration = time.iloc[i] - time.iloc[start_idx]
            if duration >= min_duration:
                end_idx = i

                # Check if gap since last step > 30s
                current_step_time = time.iloc[start_idx]
                if last_step_time is not None:
                    if current_step_time - last_step_time > train_gap_threshold:
                        train_count += 1
                        train_dir = os.path.join(save_dir, f"Train_{train_count:02d}")
                        os.makedirs(train_dir, exist_ok=True)
                        print(f"\n🔁 New train started: Train {train_count:02d}")

                last_step_time = current_step_time

                # Extract window
                t_start = time.iloc[start_idx] - pre_window
                t_end = time.iloc[end_idx] + post_window
                mask = (time >= t_start) & (time <= t_end)

                selected_time = time[mask].reset_index(drop=True)
                selected_voltage = primary[mask].reset_index(drop=True)
                selected_current = secondary[mask].reset_index(drop=True)

                # Spike detection
                spike_threshold = -0  # mV
                min_peak_distance = int(0.002 / (time[1] - time[0]))  # 2 ms
                min_spike_height = 30  # mV from local min to peak

                peaks, properties = find_peaks(selected_voltage, height=spike_threshold, distance=min_peak_distance)

                spike_indices = []
                for idx in peaks:
                    local_min = np.min(selected_voltage[max(0, idx - 10):idx + 1])
                    if (selected_voltage[idx] - local_min) >= min_spike_height:
                        spike_indices.append(idx)

                spike_times = selected_time[spike_indices]
                spike_voltages = selected_voltage[spike_indices]

                # Store spike count
                all_spike_counts.append({
                    "Train": train_count,
                    "Step": step_count,
                    "Num Spikes": len(spike_indices)
                })

                # Save spike CSV
                spike_df = pd.DataFrame({
                    "Spike Time (s)": spike_times,
                    "Spike Voltage (mV)": spike_voltages
                })
                spike_csv_path = os.path.join(train_dir, f"Step_{step_count:02d}_Spikes.csv")
                spike_df.to_csv(spike_csv_path, index=False)
                print(f"📁 Saved spike CSV for step {step_count}: {spike_csv_path}")

                # Save voltage/current CSV
                out_df = pd.DataFrame({
                    "Time (s)": selected_time,
                    "Voltage (mV)": selected_voltage,
                    "Current (pA)": selected_current
                })
                csv_path = os.path.join(train_dir, f"Step_{step_count:02d}_VoltageTrace.csv")
                out_df.to_csv(csv_path, index=False)
                print(f"✅ Saved CSV: {csv_path}")

                # --- Calculate pre-step Vm ---
                baseline_window_sec = 0.5  # 500 ms
                sampling_interval = time.iloc[1] - time.iloc[0]
                baseline_samples = int(baseline_window_sec / sampling_interval)

                baseline_start_idx = max(start_idx - baseline_samples, 0)
                baseline_voltage = primary.iloc[baseline_start_idx:start_idx]

                mean_vm = baseline_voltage.mean()

                # Save to a CSV (append to a global list first)
                if 'baseline_vm_list' not in locals():
                    baseline_vm_list = []

                baseline_vm_list.append({
                    "Train": train_count,
                    "Step": step_count,
                    "Pre-step Vm (mV)": mean_vm
                })

                # Plot with spikes
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

                ax1.plot(selected_time, selected_voltage, color="tab:red", label="Voltage")
                ax1.plot(spike_times, spike_voltages, "ko", label="Spikes")
                ax1.set_ylabel("Voltage (mV)")
                ax1.set_title(f"Step {step_count}: Voltage Response")
                ax1.grid(True)
                ax1.legend(loc="upper right")

                ax2.plot(selected_time, selected_current, color="tab:blue", label="Current")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Current (pA)")
                ax2.set_title("Current Injection")
                ax2.grid(True)

                fig.tight_layout()
                fig_path = os.path.join(train_dir, f"Step_{step_count:02d}_VoltageCurrent.png")
                plt.savefig(fig_path, dpi=300)
                plt.close()
                print(f"🖼️ Saved plot for step {step_count} with spikes: {fig_path}")

                step_count += 1

            start_idx = None  # Reset

print(f"\n🎯 Total steps saved: {step_count}")
print(f"📦 Total trains detected: {train_count}")

# Save spike counts summary CSV
spike_counts_df = pd.DataFrame(all_spike_counts)
spike_counts_path = os.path.join(save_dir, "Spike_Counts_PerStep_PerTrain.csv")
spike_counts_df.to_csv(spike_counts_path, index=False)
print(f"📈 Saved spike count summary: {spike_counts_path}")


# Save baseline Vm values
vm_df = pd.DataFrame(baseline_vm_list)
vm_path = os.path.join(save_dir, "BaselineVm_BeforeEachStep.csv")
vm_df.to_csv(vm_path, index=False)
print(f"📄 Saved baseline Vm before each step: {vm_path}")

train_dirs = [f"Train_{i:02d}" for i in range(1, train_count + 1)]


for train_id in range(1, train_count + 1):
    train_dir = os.path.join(save_dir, f"Train_{train_id:02d}")
    voltage_traces = []
    current_traces = []
    time_vectors = []

    for file in sorted(os.listdir(train_dir)):
        if file.endswith("_VoltageTrace.csv"):
            df = pd.read_csv(os.path.join(train_dir, file))
            time = df["Time (s)"]
            voltage = df["Voltage (mV)"]
            current = df["Current (pA)"]

            # Align to -50ms
            time_aligned = time - time.iloc[0] - 0.05

            time_vectors.append(time_aligned)
            voltage_traces.append(voltage)
            current_traces.append(current)

    # Compute mean baseline Vm before t=0 across all traces
    baseline_vms = [v[t < 0].mean() for v, t in zip(voltage_traces, time_vectors)]
    global_baseline_vm = np.mean(baseline_vms)

    # Subtract individual baseline and add global mean
    voltage_traces = [
        v - v[t < 0].mean() + global_baseline_vm
        for v, t in zip(voltage_traces, time_vectors)
    ]

    # Create DataFrame for seaborn
    plot_df = pd.DataFrame()
    # Injected currents per step (assuming 12 steps exactly in this order)
    injected_currents = [-100, -50, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    plot_df = pd.DataFrame()
    trace_labels = []

    for i, (t, v) in enumerate(zip(time_vectors, voltage_traces)):
        if i < len(injected_currents):
            current_label = f"{injected_currents[i]} pA"
        else:
            current_label = f"Extra_{i}"
        trace_labels.append(current_label)

        df = pd.DataFrame({
            "Time (s)": t,
            "Voltage (mV)": v,
            "Trace": current_label
        })
        plot_df = pd.concat([plot_df, df], ignore_index=True)

    # Pick 4 currents to highlight
    highlight_traces = random.sample(trace_labels, min(4, len(trace_labels)))

    # Plot using seaborn
    plt.figure(figsize=(10, 5))
    for trace_id in trace_labels:
        trace_data = plot_df[plot_df["Trace"] == trace_id]
        if trace_id in highlight_traces:
            sns.lineplot(data=trace_data, x="Time (s)", y="Voltage (mV)", label=trace_id)
        else:
            sns.lineplot(data=trace_data, x="Time (s)", y="Voltage (mV)", color="lightgray", linewidth=1, alpha=0.6)

    plt.title(f"Train {train_id:02d} - Aligned Voltage Traces")
    plt.ylabel("Voltage (mV)")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.tight_layout()

    fig_path = os.path.join(train_dir, "Overlapping_Traces.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"🎨 Saved seaborn overlapping trace plot: {fig_path}")


# ➕ Calculate Rheobase for each train and collect summary
rheobase_summary = []

for train_id in range(1, train_count + 1):
    train_dir = os.path.join(save_dir, f"Train_{train_id:02d}")
    rheobase_found = False
    rheobase_value = None
    step_number = None

    for file in sorted(os.listdir(train_dir)):
        if file.endswith("_Spikes.csv"):
            spike_path = os.path.join(train_dir, file)
            spike_df = pd.read_csv(spike_path)

            if len(spike_df) > 0:
                # Get corresponding voltage/current trace
                step_num = file.split("_")[1]
                trace_file = os.path.join(train_dir, f"Step_{step_num}_VoltageTrace.csv")
                if os.path.exists(trace_file):
                    trace_df = pd.read_csv(trace_file)
                    current_vals = trace_df["Current (pA)"]
                    median_current = np.median(current_vals)
                    if median_current <= 0:
                        continue  # Skip non-depolarizing steps
                    rheobase_value = median_current
                    step_number = int(step_num)
                    rheobase_found = True
                    break  # Stop after finding first depolarizing spike step

    rheobase_path = os.path.join(train_dir, "Rheobase.txt")
    with open(rheobase_path, "w") as f:
        if rheobase_found:
            f.write(f"Rheobase for Train {train_id:02d}: {rheobase_value:.1f} pA\n")
            print(f"⚡ Train {train_id:02d} Rheobase: {rheobase_value:.1f} pA")

            rheobase_summary.append({
                "Train": train_id,
                "Step File": file,
                "Step Number": step_number,
                "Rheobase (pA)": rheobase_value
            })
        else:
            f.write(f"No spikes detected in Train {train_id:02d}. Rheobase not found.\n")
            print(f"❌ Train {train_id:02d}: No spikes found. Rheobase not determined.")

# Save summary CSV
summary_df = pd.DataFrame(rheobase_summary)
summary_csv_path = os.path.join(save_dir, "Rheobase_Summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"📊 Saved rheobase summary: {summary_csv_path}")

# --- Identify 1.5x Rheobase steps and compute firing frequency ---
for row in rheobase_summary:
    train_id = row["Train"]
    rheobase = row["Rheobase (pA)"]
    target_current = round(rheobase * 1.5)
    train_dir = os.path.join(save_dir, f"Train_{train_id:02d}")

    match_file = None
    best_current = None
    for file in sorted(os.listdir(train_dir)):
        if file.endswith("_VoltageTrace.csv"):
            df = pd.read_csv(os.path.join(train_dir, file))
            median_current = np.median(df["Current (pA)"])
            if median_current >= target_current:
                if match_file is None or abs(median_current - target_current) < abs(best_current - target_current):
                    match_file = file
                    best_current = median_current

    if match_file:
        voltage_df = pd.read_csv(os.path.join(train_dir, match_file))
        time = voltage_df["Time (s)"]
        voltage = voltage_df["Voltage (mV)"]

        spike_threshold = 0  # mV
        min_peak_distance = int(0.002 / (time[1] - time[0]))  # 2 ms
        min_spike_height = 30

        peaks, properties = find_peaks(voltage, height=spike_threshold, distance=min_peak_distance)
        spike_indices = [idx for idx in peaks if (voltage[idx] - np.min(voltage[max(0, idx - 10):idx + 1])) >= min_spike_height]
        spike_times = time[spike_indices]
        firing_rate = len(spike_times) / (time.iloc[-1] - time.iloc[0])

        with open(os.path.join(train_dir, "FiringRate_1.5xRheobase.txt"), "w") as f:
            f.write(f"Firing frequency at 1.5x Rheobase ({target_current} pA): {firing_rate:.2f} Hz\n")
        print(f"🔥 Train {train_id:02d} - Firing rate at 1.5x Rheobase: {firing_rate:.2f} Hz")

        rheobase_file = f"Step_{row['Step Number']:02d}_VoltageTrace.csv"
        rheo_df = pd.read_csv(os.path.join(train_dir, rheobase_file))

        def detect_onset(current_trace):
            dI = np.diff(current_trace)
            onset_idx = np.where(dI < -5)[0][0] if np.any(dI < -5) else 0
            return onset_idx

        rheo_onset = detect_onset(rheo_df["Current (pA)"].values)
        target_onset = detect_onset(voltage_df["Current (pA)"].values)

        step_duration = int(0.5 / (time[1] - time[0]))  # assume 500 ms step duration
        pre_window = int(0.1 / (time[1] - time[0]))  # 100 ms before
        post_window = int(0.1 / (time[1] - time[0]))  # 100 ms after

        def clip(start, end, length):
            return max(0, start), min(length, end)

        r_start, r_end = clip(rheo_onset - pre_window, rheo_onset + step_duration + post_window, len(rheo_df))
        t_start, t_end = clip(target_onset - pre_window, target_onset + step_duration + post_window, len(voltage_df))

        aligned_rheo_t = rheo_df["Time (s)"].values[r_start:r_end] - rheo_df["Time (s)"].values[rheo_onset]
        aligned_rheo_v = rheo_df["Voltage (mV)"].values[r_start:r_end]

        aligned_target_t = voltage_df["Time (s)"].values[t_start:t_end] - voltage_df["Time (s)"].values[target_onset]
        aligned_target_v = voltage_df["Voltage (mV)"].values[t_start:t_end]

        plt.figure(figsize=(8, 4))
        plt.plot(aligned_rheo_t, aligned_rheo_v, label=f"Rheobase ({rheobase:.0f} pA)", color="tab:blue")
        plt.plot(aligned_target_t, aligned_target_v, label=f"1.5xRheobase (closest {best_current:.0f} pA)", color="tab:red")
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Time from step onset (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Train {train_id:02d} - Aligned Traces: Rheobase vs 1.5xRheobase")
        plt.legend()
        plt.grid(True)
        plt.ylim(min(aligned_rheo_v.min(), aligned_target_v.min()) - 5, max(aligned_rheo_v.max(), aligned_target_v.max()) + 5)
        plt.tight_layout()

        fig_path = os.path.join(train_dir, "Overlay_Rheo_1.5xRheo_Aligned.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"🖼️ Saved aligned overlay plot for Train {train_id:02d}: {fig_path}")
    else:
        print(f"⚠️ Train {train_id:02d}: No suitable step found for 1.5x Rheobase ({target_current} pA)")

# === Add block to calculate latency, threshold, FWHM, amplitude, and AHP at rheobase ===
latency_results = []

for row in rheobase_summary:
    train_id = row["Train"]
    step_file = row["Step File"]
    train_dir = os.path.join(save_dir, f"Train_{train_id:02d}")
    trace_path = os.path.join(train_dir, step_file.replace("_Spikes", "_VoltageTrace"))

    if os.path.exists(trace_path):
        df = pd.read_csv(trace_path)
        time = df["Time (s)"].values
        voltage = df["Voltage (mV)"].values
        current = df["Current (pA)"].values

        dI = np.diff(current)
        onset_idx = np.where(dI < -5)[0][0] if np.any(dI < -5) else 0
        onset_time = time[onset_idx]

        spike_threshold = 0
        min_peak_distance = int(0.002 / (time[1] - time[0]))
        min_spike_height = 30

        peaks, _ = find_peaks(voltage, height=spike_threshold, distance=min_peak_distance)
        valid_peaks = [idx for idx in peaks if (voltage[idx] - np.min(voltage[max(0, idx - 10):idx + 1])) >= min_spike_height]

        if valid_peaks:
            first_spike_idx = valid_peaks[0]
            first_spike_time = time[first_spike_idx]
            latency = first_spike_time - onset_time

            search_window = voltage[max(0, first_spike_idx - 10):first_spike_idx + 1]
            threshold_value = np.min(search_window)

            peak_voltage = voltage[first_spike_idx]
            half_max = (peak_voltage + threshold_value) / 2
            left_idx = first_spike_idx
            while left_idx > 0 and voltage[left_idx] > half_max:
                left_idx -= 1
            right_idx = first_spike_idx
            while right_idx < len(voltage) and voltage[right_idx] > half_max:
                right_idx += 1
            fwhm = time[right_idx] - time[left_idx] if right_idx > left_idx else np.nan

            # Amplitude
            amplitude = peak_voltage - threshold_value

            # AHP: find minimum voltage after the peak within next 100 ms
            after_spike_window = int(0.1 / (time[1] - time[0]))
            ahp_region = voltage[first_spike_idx:first_spike_idx + after_spike_window]
            ahp_value = np.min(ahp_region) if len(ahp_region) > 0 else np.nan
            ahp_amplitude = threshold_value - ahp_value

            latency_results.append({
                "Train": train_id,
                "Rheobase (pA)": row["Rheobase (pA)"],
                "Latency to 1st Spike (s)": latency,
                "Threshold (mV)": threshold_value,
                "Amplitude (mV)": amplitude,
                "AHP (mV)": ahp_amplitude,
                "FWHM (s)": fwhm
            })

            plt.figure(figsize=(8, 4))
            plt.plot(time, voltage, label='Voltage Trace', color='black')
            plt.axvline(onset_time, color='cyan', linestyle='--', label='Step Onset')
            plt.plot(first_spike_time, voltage[first_spike_idx], 'ro', label='1st Spike Peak')
            plt.hlines(threshold_value, time[first_spike_idx - 20], time[first_spike_idx + 20], color='purple', linestyle='--', label='Threshold')
            plt.hlines(half_max, time[left_idx], time[right_idx], color='orange', linestyle='-', label='FWHM')
            plt.axvline(time[left_idx], color='orange', linestyle=':')
            plt.axvline(time[right_idx], color='orange', linestyle=':')
            if not np.isnan(ahp_value):
                ahp_time_idx = np.argmin(ahp_region) + first_spike_idx
                plt.plot(time[ahp_time_idx], ahp_value, 'bs', label='AHP')
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (mV)")
            plt.title(f"Train {train_id} - Rheobase Spike Analysis")
            plt.legend()
            plt.tight_layout()
            fig_path = os.path.join(train_dir, f"{step_file.replace('_Spikes.csv', '_FirstSpikeAnalysis.png')}")
            plt.savefig(fig_path)
            plt.close()

latency_df = pd.DataFrame(latency_results)
latency_csv_path = os.path.join(save_dir, "Rheobase_FirstSpike_Properties.csv")
latency_df.to_csv(latency_csv_path, index=False)
print(f"📄 Saved latency, threshold, amplitude, AHP, and FWHM at rheobase: {latency_csv_path}")



"""
print("\n⚡ Calculating firing rate at ~1.5× Rheobase per train...")

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    rheobase_file = os.path.join(train_path, "Rheobase.txt")

    # Skip if no rheobase info
    if not os.path.exists(rheobase_file):
        print(f"❌ {train}: Rheobase.txt not found. Skipping.")
        continue

    # Load rheobase value
    with open(rheobase_file, "r") as f:
        line = f.readline()
        if "Rheobase" in line and ":" in line:
            try:
                rheobase_pA = float(line.split(":")[1].split()[0])
                if rheobase_pA < 10:  # Ignore nonsensical or empty values
                    raise ValueError
            except:
                print(f"⚠️ {train}: Could not parse rheobase.")
                continue
        else:
            print(f"⚠️ {train}: No valid rheobase value in file.")
            continue

    target_current = 1.5 * rheobase_pA
    closest_step = None
    closest_diff = float("inf")
    closest_step_current = None

    # Find step with closest current to 1.5× Rheobase
    for file in os.listdir(train_path):
        if file.endswith("_VoltageTrace.csv"):
            df = pd.read_csv(os.path.join(train_path, file))
            step_current = df["Current (pA)"].mean()

            # ✅ Skip low-current steps clearly below 1.5× rheobase
            if step_current < 0.8 * target_current:
                continue  # ✅ Add this line to skip low-current steps
            
            diff = abs(step_current - target_current)

            if diff < closest_diff:
                closest_diff = diff
                closest_step = file
                closest_step_current = step_current  # ✅ store correct current


    # If a valid closest step is found
    if closest_step:
        voltage_trace_path = os.path.join(train_path, closest_step)
        spike_csv_path = voltage_trace_path.replace("_VoltageTrace.csv", "_Spikes.csv")

        if os.path.exists(spike_csv_path):
            spikes_df = pd.read_csv(spike_csv_path)
            step_df = pd.read_csv(voltage_trace_path)

            duration_s = step_df["Time (s)"].iloc[-1] - step_df["Time (s)"].iloc[0]
            n_spikes = len(spikes_df)
            firing_rate = n_spikes / duration_s if duration_s > 0 else 0

            print(f"✅ {train} → 1.5× Rheobase step: {closest_step}, Firing rate: {firing_rate:.2f} Hz")

            # Save info
            with open(os.path.join(train_path, "FiringRate_1.5xRheobase.txt"), "w") as f:
                f.write(f"Train: {train}\n")
                f.write(f"Step: {closest_step}\n")
                f.write(f"Mean Current (pA): {closest_step_current:.2f}\n")
                f.write(f"Firing rate (Hz): {firing_rate:.2f}\n")
        else:
            print(f"⚠️ {train}: Spike file not found for step {closest_step}")
    else:
        print(f"⚠️ {train}: No step found close to 1.5× Rheobase")


print("\n⏱ Calculating first spike latency for each step and at rheobase...")

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    step_files = [f for f in os.listdir(train_path) if f.endswith("_Spikes.csv")]

    latency_records = []
    rheobase_latency = None
    rheobase_current = None

    # Load rheobase from text file
    rheobase_file = os.path.join(train_path, "Rheobase.txt")
    if os.path.exists(rheobase_file):
        with open(rheobase_file, "r") as f:
            for line in f:
                if "Rheobase" in line and "pA" in line:
                    try:
                        rheobase_current = float(line.strip().split(":")[1].replace("pA", "").strip())
                    except:
                        pass

    for spike_file in step_files:
        step_num = spike_file.split("_")[1]
        spike_path = os.path.join(train_path, spike_file)
        trace_path = os.path.join(train_path, f"Step_{step_num}_VoltageTrace.csv")

        if not os.path.exists(trace_path):
            continue

        df_spikes = pd.read_csv(spike_path)
        df_trace = pd.read_csv(trace_path)

        if not df_spikes.empty:
            first_spike_time = df_spikes["Spike Time (s)"].iloc[0]
            step_start_time = df_trace["Time (s)"].iloc[0]
            latency = first_spike_time - step_start_time
            mean_current = df_trace["Current (pA)"].mean()

            is_rheobase_step = False
            if rheobase_current is not None and abs(mean_current - rheobase_current) < 10:
                is_rheobase_step = True
                rheobase_latency = latency * 1000

            latency_records.append({
                "Step": f"Step_{step_num}_VoltageTrace.csv",
                "Mean Current (pA)": mean_current,
                "First Spike Latency (ms)": latency * 1000,
                "Rheobase Step": is_rheobase_step
            })

    if latency_records:
        latency_df = pd.DataFrame(latency_records)
        latency_csv = os.path.join(train_path, "First_Spike_Latency.csv")
        latency_df.to_csv(latency_csv, index=False)
        print(f"✅ {train} → First spike latency saved: {latency_csv}")

        # Plotting
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(latency_df["Mean Current (pA)"], latency_df["First Spike Latency (ms)"], marker="o")
        ax.set_title(f"{train}: Latency vs Injected Current")
        ax.set_xlabel("Mean Current (pA)")
        ax.set_ylabel("First Spike Latency (ms)")
        ax.grid(True)

        fig.tight_layout()
        plot_path = os.path.join(train_path, "Latency_vs_Current.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
    else:
        print(f"⚠️ {train}: No spikes detected in any step, latency not calculated.")


print("\n📊 Generating spike summary table with current values...")

summary_records = []

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    for file in os.listdir(train_path):
        if file.endswith("_Spikes.csv"):
            step_num = file.split("_")[1]
            spikes_path = os.path.join(train_path, file)
            trace_path = os.path.join(train_path, f"Step_{step_num}_VoltageTrace.csv")

            # Read spike file
            try:
                df_spikes = pd.read_csv(spikes_path)
                spike_count = len(df_spikes)
            except:
                spike_count = 0

            # Read current from voltage trace
            try:
                df_trace = pd.read_csv(trace_path)
                mean_current = df_trace["Current (pA)"].mean()
            except:
                mean_current = np.nan

            summary_records.append({
                "Train": train,
                "Step": step_num,
                "Spikes Detected": spike_count,
                "Mean Current (pA)": mean_current
            })

summary_df = pd.DataFrame(summary_records)
summary_csv = os.path.join(save_dir, "Spike_Summary_With_Current.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"📄 Spike summary table saved: {summary_csv}")

print("\n📊 Generating spike summary table with current values and maximum firing rate...")

summary_records = []

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    for file in os.listdir(train_path):
        if file.endswith("_Spikes.csv"):
            step_num = file.split("_")[1]
            spikes_path = os.path.join(train_path, file)
            trace_path = os.path.join(train_path, f"Step_{step_num}_VoltageTrace.csv")

            # Read spike file
            try:
                df_spikes = pd.read_csv(spikes_path)
                spike_count = len(df_spikes)
            except:
                spike_count = 0

            # Read current from voltage trace
            try:
                df_trace = pd.read_csv(trace_path)
                mean_current = df_trace["Current (pA)"].mean()
            except:
                mean_current = np.nan

            summary_records.append({
                "Train": train,
                "Step": step_num,
                "Spikes Detected": spike_count,
                "Mean Current (pA)": mean_current
            })

summary_df = pd.DataFrame(summary_records)

# ➕ Add max spike count and max firing rate per train
max_freq_records = []
for train in train_dirs:
    df_train = summary_df[summary_df["Train"] == train].copy()
    df_train = df_train.dropna(subset=["Mean Current (pA)"])
    if not df_train.empty:
        df_train = df_train[df_train["Spikes Detected"] > 0]
        if not df_train.empty:
            for idx, row in df_train.iterrows():
                step_file = os.path.join(save_dir, train, f"Step_{int(row['Step']):02d}_VoltageTrace.csv")
                if os.path.exists(step_file):
                    df_trace = pd.read_csv(step_file)
                    duration = df_trace["Time (s)"].iloc[-1] - df_trace["Time (s)"].iloc[0]
                    firing_rate = row["Spikes Detected"] / duration if duration > 0 else 0
                    summary_df.loc[idx, "Firing Rate (Hz)"] = firing_rate

            max_idx = df_train["Spikes Detected"].idxmax()
            max_row = summary_df.loc[max_idx]
            max_freq_records.append({
                "Train": train,
                "Max Firing Step": f"Step_{int(max_row['Step']):02d}",
                "Max Firing Rate (Hz)": max_row["Firing Rate (Hz)"],
                "Current (pA)": max_row["Mean Current (pA)"]
            })

# Save maximum firing rate per train
max_freq_df = pd.DataFrame(max_freq_records)
max_freq_csv = os.path.join(save_dir, "Max_FiringRate_Per_Train.csv")
max_freq_df.to_csv(max_freq_csv, index=False)
print(f"📈 Saved maximum firing rate per train: {max_freq_csv}")

# Save complete summary table
summary_csv = os.path.join(save_dir, "Spike_Summary_With_Current.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"📄 Spike summary table saved: {summary_csv}")


# === Extract AP features for each spike ===
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import sem

print("\n🧐 Extracting action potential features...")

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    step_files = [f for f in os.listdir(train_path) if f.endswith("_Spikes.csv")]

    for spike_file in step_files:
        step_num = spike_file.split("_")[1]
        voltage_file = os.path.join(train_path, f"Step_{step_num}_VoltageTrace.csv")
        spike_path = os.path.join(train_path, spike_file)

        if not os.path.exists(voltage_file):
            continue

        df_trace = pd.read_csv(voltage_file)
        df_spikes = pd.read_csv(spike_path)

        if df_spikes.empty:
            continue

        time = df_trace["Time (s)"].values
        voltage = df_trace["Voltage (mV)"].values
        dt = time[1] - time[0]

        voltage_smooth = gaussian_filter1d(voltage, sigma=1)
        window_ms = 10
        window_pts = int(window_ms / 1000 / dt)

        features = []
        aligned_traces = []
        aligned_times = []

        for spike_time in df_spikes["Spike Time (s)"].values:
            spike_idx = np.argmin(np.abs(time - spike_time))
            start = max(spike_idx - window_pts, 0)
            end = min(spike_idx + window_pts, len(voltage))

            v_win = voltage_smooth[start:end]
            t_win = time[start:end] - spike_time

            if len(v_win) < 5:
                continue

            aligned_traces.append(v_win)
            aligned_times.append(t_win)

            v_peak = np.max(v_win)
            v_trough = np.min(v_win)
            peak_idx = np.argmax(v_win)

            dVdt = np.gradient(v_win, dt)
            thresh_idx = np.argmax(dVdt > 10)
            v_thresh = v_win[thresh_idx] if np.any(dVdt > 10) else np.nan

            half_amp = (v_peak + v_trough) / 2
            above_half = np.where(v_win > half_amp)[0]
            if len(above_half) >= 2:
                v_halfwidth = (t_win[above_half[-1]] - t_win[above_half[0]]) * 1000
            else:
                v_halfwidth = np.nan

            ahp = np.min(v_win[peak_idx:]) if peak_idx < len(v_win) else np.nan

            features.append({
                "Spike Time (s)": spike_time,
                "APVpeak (mV)": v_peak,
                "APVthresh (mV)": v_thresh,
                "APVamp (mV)": v_peak - v_trough,
                "APVhalf (ms)": v_halfwidth,
                "AHP (mV)": ahp,
                "APVslope (mV/s)": np.max(dVdt),
                "APVmin (mV)": v_trough
            })

        if features:
            df_feat = pd.DataFrame(features)
            feat_path = os.path.join(train_path, f"Step_{step_num}_APFeatures.csv")
            df_feat.to_csv(feat_path, index=False)
            print(f"🧪 {train} Step {step_num}: Saved AP features to {feat_path}")

        if aligned_traces:
            min_len = min(len(trace) for trace in aligned_traces)
            aligned_traces = [trace[:min_len] for trace in aligned_traces]
            aligned_times = [time[:min_len] for time in aligned_times]

            mean_trace = np.mean(aligned_traces, axis=0)
            mean_time = np.mean(aligned_times, axis=0)

            avg_trace_df = pd.DataFrame({"Time (ms)": mean_time * 1000, "Voltage (mV)": mean_trace})
            avg_path = os.path.join(train_path, f"Step_{step_num}_APTraceAvg.csv")
            avg_trace_df.to_csv(avg_path, index=False)

            plt.figure(figsize=(6, 4))
            for trace, t in zip(aligned_traces, aligned_times):
                plt.plot(t[:min_len] * 1000, trace[:min_len], color='lightgray', linewidth=0.5, alpha=0.6)
            plt.plot(mean_time * 1000, mean_trace, label='Average AP', color='black')
            plt.axhline(0, color='gray', linestyle='--', lw=0.5)
            plt.title(f"{train} Step {step_num} — Avg AP Trace")
            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage (mV)")
            plt.grid(True)
            plt.tight_layout()
            fig_path = avg_path.replace(".csv", ".png")
            plt.savefig(fig_path)
            plt.close()
            print(f"📉 {train} Step {step_num}: Saved average AP trace to {avg_path} and {fig_path}")

# === Compare and summarize AP features for 1.5× Rheobase and Max firing steps ===
print("\n📊 Generating summary stats for 1.5× Rheobase and Max firing rate APs...")

summary_all = []

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    fr_path = os.path.join(train_path, "FiringRate_1.5xRheobase.txt")
    max_path = os.path.join(save_dir, "Max_FiringRate_Per_Train.csv")
    if not os.path.exists(fr_path) or not os.path.exists(max_path):
        continue

    with open(fr_path, 'r') as f:
        lines = f.readlines()
        step1 = lines[1].split(":")[1].strip().replace("_VoltageTrace.csv", "")

    df_max = pd.read_csv(max_path)
    row = df_max[df_max["Train"] == train]
    if row.empty:
        continue
    step2 = row["Max Firing Step"].values[0]

    feat1_path = os.path.join(train_path, f"{step1}_APFeatures.csv")
    feat2_path = os.path.join(train_path, f"{step2}_APFeatures.csv")

    if os.path.exists(feat1_path):
        df_feat1 = pd.read_csv(feat1_path)
        row1 = {"Train": train, "Condition": "1.5x Rheobase"}
        for col in ["APVmin (mV)", "APVpeak (mV)", "APVthresh (mV)", "APVslope (mV/s)", "APVhalf (ms)", "APVamp (mV)", "AHP (mV)"]:
            row1[col + " Mean"] = df_feat1[col].mean()
            row1[col + " SEM"] = sem(df_feat1[col])
        summary_all.append(row1)

    if os.path.exists(feat2_path):
        df_feat2 = pd.read_csv(feat2_path)
        row2 = {"Train": train, "Condition": "Max Firing Rate"}
        for col in ["APVmin (mV)", "APVpeak (mV)", "APVthresh (mV)", "APVslope (mV/s)", "APVhalf (ms)", "APVamp (mV)", "AHP (mV)"]:
            row2[col + " Mean"] = df_feat2[col].mean()
            row2[col + " SEM"] = sem(df_feat2[col])
        summary_all.append(row2)

if summary_all:
    df_summary = pd.DataFrame(summary_all)
    summary_path = os.path.join(save_dir, "APFeatures_Comparison_Summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"📄 Saved overall AP feature summary to {summary_path}")
"""