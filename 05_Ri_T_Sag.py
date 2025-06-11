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
save_dir = os.path.join(os.path.dirname(file_path), "05_Ri_Tm_Sag")
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

# ============================================================
# ✅ DETECT HYPERPOLARIZING STEPS — Add this to the END
# ============================================================
print("\n🔍 Scanning for hyperpolarizing steps per train...")

train_dirs = [d for d in os.listdir(save_dir) if d.startswith("Train_")]
hyperpolarizing_steps = {}

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    step_files = [f for f in os.listdir(train_path) if f.endswith("_VoltageTrace.csv")]

    for step_file in step_files:
        csv_path = os.path.join(train_path, step_file)
        df_step = pd.read_csv(csv_path)

        current_trace = df_step["Current (pA)"]
        mean_current = current_trace.mean()

        if mean_current < 0:
            if train not in hyperpolarizing_steps:
                hyperpolarizing_steps[train] = []
            hyperpolarizing_steps[train].append((step_file, mean_current))

# ✅ Display summary
for train, steps in hyperpolarizing_steps.items():
    print(f"\n🚨 {train} - Hyperpolarizing Steps Detected:")
    for step_file, current in steps:
        print(f"   - {step_file}: {current:.1f} pA")


# Store sag results
sag_results = {}

train_dirs = [d for d in os.listdir(save_dir) if d.startswith("Train_")]

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    step_files = [f for f in os.listdir(train_path) if f.endswith("_VoltageTrace.csv")]
    train_sags = []

    for step_file in step_files:
        csv_path = os.path.join(train_path, step_file)
        df = pd.read_csv(csv_path)

        current_trace = df["Current (pA)"]
        mean_current = current_trace.mean()

        if mean_current < 0:  # Only consider hyperpolarizing steps
            voltage = df["Voltage (mV)"]
            time = df["Time (s)"]
            sampling_interval = time[1] - time[0]

            baseline_voltage = voltage.iloc[:int(0.05 / sampling_interval)].mean()
            min_voltage = voltage.min()
            steady_state_voltage = voltage.iloc[-int(0.01 / sampling_interval):].mean()

            sag_ratio = (steady_state_voltage - min_voltage) / (baseline_voltage - min_voltage) if (baseline_voltage - min_voltage) != 0 else np.nan

            train_sags.append({
                "Step": step_file,
                "Mean Current (pA)": mean_current,
                "Baseline V (mV)": baseline_voltage,
                "Min V (mV)": min_voltage,
                "Steady-State V (mV)": steady_state_voltage,
                "Sag Ratio": sag_ratio
            })

    if train_sags:
        sag_results[train] = pd.DataFrame(train_sags)
        print(f"\n📊 Sag Results for {train}")
        print(sag_results[train])

# Optionally save results per train
for train, df in sag_results.items():
    df.to_csv(os.path.join(save_dir, train, "Sag_Results.csv"), index=False)


print("\n📏 Calculating input resistance (Rin) for each train...")

for train in train_dirs:
    train_path = os.path.join(save_dir, train)
    step_files = [f for f in os.listdir(train_path) if f.endswith("_VoltageTrace.csv")]

    records = []

    for step_file in step_files:
        csv_path = os.path.join(train_path, step_file)
        df_step = pd.read_csv(csv_path)

        time = df_step["Time (s)"].values
        voltage = df_step["Voltage (mV)"].values
        current = df_step["Current (pA)"].values

        mean_current = np.mean(current)

        if mean_current < 0:  # Only use hyperpolarizing steps
            sampling_rate = 1 / (time[1] - time[0])
            desired_pts = int(0.5 * sampling_rate)  # 500 ms worth of points

            available_pts = len(voltage)
            baseline_pts = min(desired_pts, available_pts // 2)
            steady_pts = min(desired_pts, available_pts // 2)

            if baseline_pts < 1 or steady_pts < 1:
                print(f"⚠️ {step_file}: Not enough data points even for reduced baseline/steady-state.")
                continue

            baseline_v = np.mean(voltage[:baseline_pts])
            steady_v = np.mean(voltage[-steady_pts:])
            delta_v = steady_v - baseline_v  # mV

            if delta_v > 0:
                print(f"⚠️ {step_file}: Unexpected positive ΔV for hyperpolarizing step. Skipping.")
                continue

            records.append({
                "Step": step_file,
                "Mean Current (pA)": mean_current,
                "Baseline V (mV)": baseline_v,
                "Steady-State V (mV)": steady_v,
                "ΔV (mV)": delta_v
            })

    if records:
        df_rin = pd.DataFrame(records)
        csv_out = os.path.join(train_path, "Input_Resistance_Steps.csv")
        df_rin.to_csv(csv_out, index=False)
        print(f"✅ Saved input resistance table for {train}: {csv_out}")

        # Linear fit: ΔV vs ΔI
        delta_i_vals = df_rin["Mean Current (pA)"].values / 1000  # pA → nA
        delta_v_vals = df_rin["ΔV (mV)"].values

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(delta_i_vals, delta_v_vals, color="tab:blue", label="Data points")

        coeffs = np.polyfit(delta_i_vals, delta_v_vals, 1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(min(delta_i_vals), max(delta_i_vals), 100)
        ax.plot(x_fit, fit_line(x_fit), "r--", label=f"Fit: Rin = {coeffs[0]:.1f} MΩ")

        ax.set_xlabel("Injected Current (nA)")
        ax.set_ylabel("ΔV (mV)")
        ax.set_title(f"{train}: Input Resistance")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        fig_path = os.path.join(train_path, "Input_Resistance_Plot.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"📐 {train} → ΔV vs ΔI plot saved: {fig_path}")
    else:
        print(f"⚠️ No valid hyperpolarizing steps found for {train}, skipping input resistance plot.")

from scipy.optimize import curve_fit

def exp_decay(t, V_inf, V_0, tau):
    return V_inf + (V_0 - V_inf) * np.exp(-t / tau)

print("\n⏱ Calculating Membrane Time Constants (Tau_m) — Only for ~-50 pA Steps...")

tau_data = []

for train, steps in hyperpolarizing_steps.items():
    print(f"\n📘 Train {train} — Time Constants:")
    train_path = os.path.join(save_dir, train)
    
    for step_file, mean_current in steps:
        # Only analyze steps near -50 pA
        if not (-60 <= mean_current <= -40):
            continue

        csv_path = os.path.join(train_path, step_file)
        df_step = pd.read_csv(csv_path)

        t = df_step["Time (s)"].to_numpy()
        v = df_step["Voltage (mV)"].to_numpy()

        # Estimate when the voltage starts dropping (onset)
        voltage_diff = np.diff(v)
        try:
            onset_idx = np.where(voltage_diff < -0.1)[0][0]
        except IndexError:
            print(f"⚠️ No sharp drop in {step_file}, skipping")
            continue

        # Fitting window: 200 ms after onset
        sample_rate = 1 / (t[1] - t[0])
        fit_samples = int(0.2 * sample_rate)
        end_idx = min(onset_idx + fit_samples, len(t))

        fit_t = t[onset_idx:end_idx] - t[onset_idx]
        fit_v = v[onset_idx:end_idx]

        try:
            p0 = [fit_v[-1], fit_v[0], 0.02]  # initial guess
            popt, _ = curve_fit(exp_decay, fit_t, fit_v, p0=p0)
            V_inf, V_0, tau_m = popt
            tau_ms = tau_m * 1000  # Convert to ms

            # Initial guess: V_inf, V_0, tau
            p0 = [fit_v[-1], fit_v[0], 0.02]
            popt, _ = curve_fit(exp_decay, fit_t, fit_v, p0=p0)
            V_inf, V_0, tau_m = popt
            tau_ms = tau_m * 1000  # Convert to ms

            # Skip unphysiological τm
            if tau_m > 1 or tau_m < 0.0005:
                print(f"   ⚠️ Ignored unphysiological τm = {tau_ms:.2f} ms")
                continue

            # Store result
            tau_data.append({
                "Train": train,
                "Step": step_file,
                "Mean Current (pA)": mean_current,
                "Tau_m (ms)": tau_ms
            })
            print(f"   ✅ {step_file}: τm = {tau_ms:.2f} ms")

            # Plot and save fit
            plt.figure(figsize=(6, 3))
            plt.plot(fit_t * 1000, fit_v, label="Data", color="tab:blue")
            plt.plot(fit_t * 1000, exp_decay(fit_t, *popt),
                     label=f"Fit: τm = {tau_ms:.2f} ms", linestyle="--", color="tab:red")
            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage (mV)")
            plt.title(f"{train} — {step_file}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            fig_path = os.path.join(train_path, f"{step_file.replace('.csv', '')}_TauFit.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()

        except Exception as e:
            print(f"   ⚠️ Fit failed for {step_file} — {e}")

# Save all τm results to CSV
tau_df = pd.DataFrame(tau_data)
tau_csv_path = os.path.join(save_dir, "MembraneTimeConstants_Tau.csv")
tau_df.to_csv(tau_csv_path, index=False)
print(f"\n📄 Saved Tau (τm) values to CSV: {tau_csv_path}")