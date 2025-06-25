import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import find_peaks

# File pathcd
file_path = "your_file_path_here.csv"  # Replace with your actual file path

# Extract the directory path to save figures in the same location
save_dir = os.path.dirname(file_path)

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load CSV file in chunks if it's too large
chunksize = 100000
data_chunks = pd.read_csv(file_path, chunksize=chunksize)
df = pd.concat(data_chunks)

# Clean column names
df.columns = df.columns.str.strip()

# Extract signals
time = df["Time(ms)"] / 1000  # Convert to seconds
primary = df["Primary"].values * 100 if "Primary" in df.columns else None
secondary = df["Secondary"].values * 1000 if "Secondary" in df.columns else None
ecg = df["ECG"].values if "ECG" in df.columns else None
airpuff = df["AIRPUFF"].values if "AIRPUFF" in df.columns else None

# Set Seaborn Style
sns.set_style("darkgrid")
sns.set_palette("colorblind")

# === 1️⃣ Save Multi-Channel Signal Plot (Seaborn) ===
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

if primary is not None:
    sns.lineplot(x=time, y=primary, ax=axes[0], color="blue")
    axes[0].set_ylabel("Vm (mV)")
    axes[0].set_title("Multi-Channel Signals")
else:
    axes[0].set_visible(False)

if secondary is not None:
    sns.lineplot(x=time, y=secondary, ax=axes[1], color="green")
    axes[1].set_ylabel("pA (Current Injection)")
else:
    axes[1].set_visible(False)

if ecg is not None:
    sns.lineplot(x=time, y=ecg, ax=axes[2], color="red")
    axes[2].set_ylabel("ECG Signal")
else:
    axes[2].set_visible(False)

if airpuff is not None:
    sns.lineplot(x=time, y=airpuff, ax=axes[3], color="black")
    axes[3].set_ylabel("AIRPUFF Stimulus")
    axes[3].set_xlabel("Time (s)")
else:
    axes[3].set_visible(False)

plt.tight_layout()
multi_channel_fig_path = os.path.join(save_dir, "Multi_Channel_Signals_Seaborn.png")
plt.savefig(multi_channel_fig_path, dpi=300)
plt.close()
print(f"✅ Multi-channel figure saved: {multi_channel_fig_path}")

# ==== 2️⃣ Save Each Channel Independently ====
channel_data = {
    "Primary (Vm mV)": (primary, "blue", "Vm (mV)"),
    "Secondary (pA)": (secondary, "green", "Current Injection (pA)"),
    "ECG": (ecg, "red", "ECG Signal"),
    "AIRPUFF": (airpuff, "black", "Airpuff Stimulus"),
}

for label, (data, color, ylabel) in channel_data.items():
    if data is not None:
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.lineplot(x=time, y=data, ax=ax, color=color)
        ax.set_title(label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (s)")
        plt.tight_layout()
        filename = f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"✅ Saved: {filepath}")
