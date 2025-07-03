import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# === Your File Path ===
file_path = "your_file_path_here.csv"  # Replace with your actual file path
# Save MP4
save_path = "your_output_path_here.mp4"  # Replace with your desired output path

# === Load and clean data ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Convert time to seconds
df['Time(s)'] = df['Time(ms)'] / 1000.0

# Extract and scale signals
time = df['Time(s)'].values
voltage_mV = df['Primary'].values * 100   # Vm scaled by 100
ecg_mV = -df['ECG'].values * 1000  # Flipped ECG
airpuff = df["AIRPUFF"]

# Detect airpuff TTL onsets
airpuff_threshold = 0.5  # adjust as needed
airpuff_onsets = np.where(np.diff((airpuff > airpuff_threshold).astype(int)) == 1)[0]
airpuff_times = pd.Series(time).iloc[airpuff_onsets]

print(f"💨 Detected {len(airpuff_times)} airpuff stimuli.")
print("First 5 airpuff onsets (s):", airpuff_times.head().values)
print("AIRPUFF column value counts:", df['AIRPUFF'].value_counts().to_dict())

# === Parameters ===
sampling_rate = 10000  # Hz
window_duration = 10   # seconds
window_size = int(sampling_rate * window_duration)

# === Plot setup ===
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={'height_ratios': [1.5, 1.2]})

# Vm plot
line_vm, = ax1.plot([], [], lw=1)
airpuff_lines = [ax1.axvline(x=0, color='blue', linestyle='--', lw=1.2, alpha=0.7) for _ in range(10)]
ax1.set_ylabel('Vm (mV)')
ax1.set_ylim(voltage_mV.min(), voltage_mV.max())
ax1.set_xlim(0, window_duration)
ax1.set_title('AIRPUFF')

# ECG plot
line_ecg, = ax2.plot([], [], lw=1, color='tab:red')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('ECG (mV)')
ax2.set_ylim(ecg_mV.min(), ecg_mV.max())
ax2.set_xlim(0, window_duration)

# Time counter
time_text = ax1.text(0.95, 0.9, '', transform=ax1.transAxes,
                     ha='right', va='top', fontsize=12,
                     bbox=dict(facecolor='white', edgecolor='gray'))

# Init
def init():
    line_vm.set_data([], [])
    line_ecg.set_data([], [])
    for line in airpuff_lines:
        line.set_xdata(-1)  # hide initially
    time_text.set_text('')
    return [line_vm, line_ecg, *airpuff_lines, time_text]

# Update
def update(frame):
    start = frame
    end = frame + window_size
    if end >= len(time):
        ani.event_source.stop()
        return [line_vm, line_ecg, *airpuff_lines, time_text]

    x = time[start:end] - time[start]
    y_vm = voltage_mV[start:end]
    y_ecg = ecg_mV[start:end]

    line_vm.set_data(x, y_vm)
    line_ecg.set_data(x, y_ecg)

    # Update vertical airpuff lines
    onset_times_in_window = airpuff_times[(airpuff_times >= time[start]) & (airpuff_times < time[end])] - time[start]
    for i, line in enumerate(airpuff_lines):
        if i < len(onset_times_in_window):
            line.set_xdata(onset_times_in_window.iloc[i])
        else:
            line.set_xdata(-1)

    time_text.set_text(f"Time: {time[end]:.1f}s")

    return [line_vm, line_ecg, *airpuff_lines, time_text]

# Animate
step_size = int(sampling_rate * 0.25)
ani = FuncAnimation(fig, update, frames=range(0, len(time) - window_size, step_size),
                    init_func=init, blit=True, interval=5)

plt.tight_layout()
plt.show()

ani.save(save_path, writer='ffmpeg', fps=20)  # adjust fps if needed
