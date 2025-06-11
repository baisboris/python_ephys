import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Your File Path ===
file_path = "E:\\DATA\\29\\20250602_CD1_29005_58bruker_BRAIN10\\TSeries-01072025-1149-011\\TSeries-01072025-1149-011_Cycle00001_VoltageRecording_001.csv"
# Save MP4
save_path = "E:\\DATA\\29\\20250602_CD1_29005_58bruker_BRAIN10\\TSeries-01072025-1149-011\\real_time_playback.mp4"

# === Load and clean data ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Convert time to seconds
df['Time(s)'] = df['Time(ms)'] / 1000.0

# Extract and scale signals
time = df['Time(s)'].values
voltage_mV = df['Primary'].values * 100   # Vm scaled by 100
ecg_mV = -df['ECG'].values * 1000  # Flipped ECG


# === Parameters ===
sampling_rate = 10000  # Hz
window_duration = 1   # seconds
window_size = int(sampling_rate * window_duration)

# === Plot setup ===
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]})

# Vm plot
line_vm, = ax1.plot([], [], lw=1)
ax1.set_ylabel('Vm (mV)')  # back to mV label
ax1.set_ylim(voltage_mV.min(), voltage_mV.max())
ax1.set_xlim(0, window_duration)
ax1.set_title('Real-Time Ephys + ECG Playback')

# ECG plot
line_ecg, = ax2.plot([], [], lw=1, color='tab:red')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('ECG (mV)')
ax2.set_ylim(ecg_mV.min(), ecg_mV.max())  # ECG y-limits based only on ECG
ax2.set_xlim(0, window_duration)

# Time counter
time_text = ax1.text(0.95, 0.9, '', transform=ax1.transAxes,
                     ha='right', va='top', fontsize=12,
                     bbox=dict(facecolor='white', edgecolor='gray'))

# Init
def init():
    line_vm.set_data([], [])
    line_ecg.set_data([], [])
    time_text.set_text('')
    return line_vm, line_ecg, time_text

# Update
def update(frame):
    start = frame
    end = frame + window_size
    if end >= len(time):
        ani.event_source.stop()
        return line_vm, line_ecg, time_text

    x = time[start:end] - time[start]
    y_vm = voltage_mV[start:end]
    y_ecg = ecg_mV[start:end]

    line_vm.set_data(x, y_vm)
    line_ecg.set_data(x, y_ecg)
    time_text.set_text(f"Time: {time[end]:.1f}s")

    return line_vm, line_ecg, time_text

# Animate
step_size = int(sampling_rate * 0.05)
ani = FuncAnimation(fig, update, frames=range(0, len(time) - window_size, step_size),
                    init_func=init, blit=True, interval=5)

plt.tight_layout()
plt.show()

ani.save(save_path, writer='ffmpeg', fps=20)  # adjust fps if needed