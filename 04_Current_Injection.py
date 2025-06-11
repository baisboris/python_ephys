import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.cm as cm
import seaborn as sns
import random

# Load data
file_path = "E:\\DATA\\29\\20250529_CD1_29003_58bruker_JB_BRAIN2\\TSeries-01072025-1149-002\\TSeries-01072025-1149-002_Cycle00001_VoltageRecording_001.csv"
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

# Save baseline Vm values
vm_df = pd.DataFrame(baseline_vm_list)
vm_path = os.path.join(save_dir, "BaselineVm_BeforeEachStep.csv")
vm_df.to_csv(vm_path, index=False)
print(f"📄 Saved baseline Vm before each step: {vm_path}")

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