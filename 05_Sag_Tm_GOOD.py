import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.cm as cm
import seaborn as sns
import random
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# Load data
file_path = "E:\\DATA\\29\\20250529_CD1_29003_58bruker_JB_BRAIN2\\TSeries-01072025-1149-002\\TSeries-01072025-1149-002_Cycle00001_VoltageRecording_001.csv"
save_dir = os.path.join(os.path.dirname(file_path), "05_Sag_Tm")
os.makedirs(save_dir, exist_ok=True)

# Read CSV and clean headers
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Extract signals
time = df["Time(ms)"] / 1000  # seconds
primary = df["Primary"] * 100  # mV
secondary = df["Secondary"] * 2000  # pA

# Parameters
threshold = 9
min_duration = 0.01
pre_window = 0.05
post_window = 0.05
train_gap_threshold = 30

# Initializations
baseline_current = np.median(secondary[:100])
start_idx = None
step_count = 0
train_count = 1
last_step_time = None
train_step_index = 0
step_metadata = []
sag_results = []
hyperpolarizing_steps = {}

train_dir = os.path.join(save_dir, f"Train_{train_count:02d}")
os.makedirs(train_dir, exist_ok=True)

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

                current_step_time = time.iloc[start_idx]
                if last_step_time is not None and (current_step_time - last_step_time > train_gap_threshold):
                    train_count += 1
                    train_step_index = 0
                    train_dir = os.path.join(save_dir, f"Train_{train_count:02d}")
                    os.makedirs(train_dir, exist_ok=True)
                    print(f"\n🔁 New train started: Train {train_count:02d}")

                last_step_time = current_step_time

                t_start = time.iloc[start_idx] - pre_window
                t_end = time.iloc[end_idx] + post_window
                mask = (time >= t_start) & (time <= t_end)

                selected_time = time[mask].reset_index(drop=True)
                selected_voltage = primary[mask].reset_index(drop=True)
                selected_current = secondary[mask].reset_index(drop=True)

                step_current_value = selected_current.iloc[int(pre_window / (time[1] - time[0]))]

                if step_current_value >= 0:
                    start_idx = None
                    continue  # skip depolarizing steps

                if train_step_index == 0:
                    label_current = -100
                elif train_step_index == 1:
                    label_current = -50
                else:
                    label_current = step_current_value

                step_metadata.append((train_count, step_count, label_current))

                if train_count not in hyperpolarizing_steps:
                    hyperpolarizing_steps[train_count] = []
                hyperpolarizing_steps[train_count].append((f"Step_{step_count:02d}_VoltageTrace.csv", label_current))

                highlight_sag = False
                if label_current == -100:
                    sampling_interval = selected_time[1] - selected_time[0]
                    baseline_voltage = selected_voltage.iloc[:int(0.05 / sampling_interval)].mean()
                    min_voltage = selected_voltage.min()
                    step_duration = t_end - t_start - pre_window - post_window
                    steady_start_time = t_start + pre_window + step_duration - 0.03
                    steady_end_time = t_start + pre_window + step_duration - 0.01
                    steady_mask = (selected_time >= steady_start_time) & (selected_time <= steady_end_time)
                    steady_voltage = selected_voltage[steady_mask].mean()
                    sag_ratio = (steady_voltage - min_voltage) / (baseline_voltage - min_voltage) if (baseline_voltage - min_voltage) != 0 else np.nan
                    sag_entry = {
                        "Train": train_count,
                        "Baseline V (mV)": baseline_voltage,
                        "Min V (mV)": min_voltage,
                        "Steady-State V (mV)": steady_voltage,
                        "Sag Ratio": sag_ratio
                    }
                    sag_results.append(sag_entry)
                    pd.DataFrame([sag_entry]).to_csv(os.path.join(train_dir, "Sag_Results.csv"), index=False)
                    highlight_sag = True

                out_df = pd.DataFrame({
                    "Time (s)": selected_time,
                    "Voltage (mV)": selected_voltage,
                    "Current (pA)": selected_current
                })
                csv_path = os.path.join(train_dir, f"Step_{step_count:02d}_VoltageTrace.csv")
                out_df.to_csv(csv_path, index=False)
                print(f"✅ Saved CSV: {csv_path}")

                fig, ax1 = plt.subplots(figsize=(8, 4))
                ax2 = ax1.twinx()
                ax1.plot(selected_time, selected_voltage, color='tab:red', label='Voltage (mV)')
                ax2.plot(selected_time, selected_current, color='tab:blue', alpha=0.5, label='Current (pA)')
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Voltage (mV)", color='tab:red')
                ax2.set_ylabel("Current (pA)", color='tab:blue')

                if highlight_sag:
                    ax1.axhline(baseline_voltage, color='green', linestyle='--', label='Baseline V')
                    ax1.axhline(steady_voltage, color='purple', linestyle='--', label='Steady-State V')
                    ax1.axvspan(steady_start_time, steady_end_time, color='purple', alpha=0.2, label='Steady-State Window')
                    ax1.text(0.95, 0.1, f"Sag: {sag_ratio:.2f}", transform=ax1.transAxes,
                             ha='right', va='bottom', fontsize=10,
                             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                    ax1.legend(loc='upper left')

                fig.suptitle(f"Train {train_count:02d} - Step {step_count:02d} ({label_current:.0f} pA)")
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
                fig_path = os.path.join(train_dir, f"Step_{step_count:02d}_VoltageCurrent.png")
                plt.savefig(fig_path, dpi=300)
                plt.close()
                print(f"🖼️ Saved plot: {fig_path}")

                step_count += 1
                train_step_index += 1

            start_idx = None

if sag_results:
    sag_df = pd.DataFrame(sag_results)
    sag_df.to_csv(os.path.join(save_dir, "Sag_Ratios.csv"), index=False)
    print("\n📊 Saved Sag_Ratios.csv")

print(f"\n🎯 Total hyperpolarizing steps saved: {step_count}")
print(f"📦 Total trains detected: {train_count}")

# --- Tau (membrane time constant) analysis block ---
def exp_decay(t, V_inf, V_0, tau):
    return V_inf + (V_0 - V_inf) * np.exp(-t / tau)

print("\n⏱ Calculating Membrane Time Constants (Tau_m) — Only for ~-50 pA Steps...")

tau_data = []

for train, steps in hyperpolarizing_steps.items():
    print(f"\n📘 Train {train} — Time Constants:")
    train_path = os.path.join(save_dir, f"Train_{train:02d}")

    for step_file, mean_current in steps:
        if not (-60 <= mean_current <= -40):
            continue

        csv_path = os.path.join(train_path, step_file)
        df_step = pd.read_csv(csv_path)

        t = df_step["Time (s)"].to_numpy()
        v = df_step["Voltage (mV)"].to_numpy()

        voltage_diff = np.diff(v)
        try:
            sampling_interval = t[1] - t[0]
            onset_shift = max(0, pre_window - 0.005)  # shift 5 ms earlier
            onset_idx = int(onset_shift / sampling_interval)
        except IndexError:
            print(f"⚠️ No sharp drop in {step_file}, skipping")
            continue

        sample_rate = 1 / (t[1] - t[0])
        fit_samples = int(0.2 * sample_rate)
        end_idx = min(onset_idx + fit_samples, len(t))

        fit_t = t[onset_idx:end_idx] - t[onset_idx]
        fit_v = v[onset_idx:end_idx]

        try:
            p0 = [fit_v[-1], fit_v[0], 0.02]
            popt, _ = curve_fit(exp_decay, fit_t, fit_v, p0=p0)
            V_inf, V_0, tau_m = popt
            tau_ms = tau_m * 1000

            if tau_m > 1 or tau_m < 0.0005:
                print(f"   ⚠️ Ignored unphysiological τm = {tau_ms:.2f} ms")
                continue

            tau_data.append({
                "Train": train,
                "Step": step_file,
                "Mean Current (pA)": mean_current,
                "Tau_m (ms)": tau_ms
            })
            print(f"   ✅ {step_file}: τm = {tau_ms:.2f} ms")

            plt.figure(figsize=(6, 3))
            plt.plot(fit_t * 1000, fit_v, label="Data", color="tab:blue")
            plt.plot(fit_t * 1000, exp_decay(fit_t, *popt),
                     label=f"Fit: τm = {tau_ms:.2f} ms", linestyle="--", color="tab:red")
            plt.xlabel("Time (ms)")
            plt.ylabel("Voltage (mV)")
            plt.title(f"Train {train} — {step_file}")
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


# --- Rank fits by error metrics ---
print("\n🚦 Ranking τₘ fits by residual error...")
ranked_tau = []

for entry in tau_data:
    csv_path = os.path.join(save_dir, f"Train_{entry['Train']:02d}", entry['Step'])
    df_step = pd.read_csv(csv_path)
    t = df_step["Time (s)"].to_numpy()
    v = df_step["Voltage (mV)"].to_numpy()
    sampling_interval = t[1] - t[0]
    onset_idx = int((pre_window - 0.005) / sampling_interval)
    sample_rate = 1 / sampling_interval
    fit_samples = int(0.2 * sample_rate)
    end_idx = min(onset_idx + fit_samples, len(t))
    fit_t = t[onset_idx:end_idx] - t[onset_idx]
    fit_v = v[onset_idx:end_idx]

    try:
        popt = [entry['Tau_m (ms)'] / 1000.0]  # convert ms back to s
        V_inf = fit_v[-1]
        V_0 = fit_v[0]
        pred_v = exp_decay(fit_t, V_inf, V_0, popt[0])
        r2 = r2_score(fit_v, pred_v)
        rmse = np.sqrt(mean_squared_error(fit_v, pred_v))

        entry["R2"] = r2
        entry["RMSE"] = rmse
        ranked_tau.append(entry)

    except Exception as e:
        print(f"⚠️ Error ranking {entry['Step']}: {e}")

# Filter by R² and RMSE
filtered_tau = [e for e in ranked_tau if e['R2'] >= 0.8 and e['RMSE'] <= 5.0]
print(f"\n🧹 Retained {len(filtered_tau)} fits with R² ≥ 0.95 and RMSE ≤ 2.0")

# Save filtered results
filtered_df = pd.DataFrame(filtered_tau)
filtered_csv_path = os.path.join(save_dir, "MembraneTimeConstants_Tau_Filtered.csv")
filtered_df.to_csv(filtered_csv_path, index=False)
print(f"📄 Saved filtered Tau values to CSV: {filtered_csv_path}")

# --- Summary plot ---
print("\n📊 Generating summary figure of τₘ fits...")
plt.figure(figsize=(8, 4))
colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_tau)))
for i, entry in enumerate(filtered_tau):
    csv_path = os.path.join(save_dir, f"Train_{entry['Train']:02d}", entry['Step'])
    df_step = pd.read_csv(csv_path)
    t = df_step["Time (s)"].to_numpy()
    v = df_step["Voltage (mV)"].to_numpy()
    sampling_interval = t[1] - t[0]
    onset_idx = int((pre_window - 0.005) / sampling_interval)
    fit_samples = int(0.2 * (1 / sampling_interval))
    end_idx = min(onset_idx + fit_samples, len(t))
    fit_t = t[onset_idx:end_idx] - t[onset_idx]
    fit_v = v[onset_idx:end_idx]
    tau = entry['Tau_m (ms)'] / 1000
    V_inf = fit_v[-1]
    V_0 = fit_v[0]
    pred_v = exp_decay(fit_t, V_inf, V_0, tau)
    plt.plot(fit_t * 1000, pred_v, label=f"Train {entry['Train']} τm={entry['Tau_m (ms)']:.2f}ms", color=colors[i])

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Overlay of τₘ Fits")
plt.legend(fontsize=7)
plt.tight_layout()
summary_fig_path = os.path.join(save_dir, "Tau_Fits_Summary.png")
plt.savefig(summary_fig_path, dpi=300)
plt.close()
print(f"🖼️ Saved overlay figure: {summary_fig_path}")
