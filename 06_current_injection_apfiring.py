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
