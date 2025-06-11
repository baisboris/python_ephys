import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# === Load Data ===
file_path = "E:\\DATA\\29\\20250602_CD1_29005_58bruker_BRAIN10\\TSeries-01072025-1149-001\\TSeries-01072025-1149-001_Cycle00001_VoltageRecording_001.csv"

save_dir = os.path.dirname(file_path)

# Load CSV
chunksize = 100000
df = pd.concat(pd.read_csv(file_path, chunksize=chunksize))
df.columns = df.columns.str.strip()

# Extract signals
time = df["Time(ms)"] / 1000
primary = df["Primary"] * 100       # mV
secondary = df["Secondary"] * 1000  # pA
ecg = df["ECG"]
airpuff = df["AIRPUFF"]

# === Create Interactive Plot ===
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                    subplot_titles=("Vm (mV)", "Current Injection (pA)", "ECG", "Airpuff"))

# Add each channel
fig.add_trace(go.Scatter(x=time, y=primary, mode='lines', name='Vm (mV)', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=secondary, mode='lines', name='Current Injection', line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=time, y=ecg, mode='lines', name='ECG', line=dict(color='red')), row=3, col=1)
fig.add_trace(go.Scatter(x=time, y=airpuff, mode='lines', name='Airpuff', line=dict(color='black')), row=4, col=1)

# Layout settings
fig.update_layout(
    height=900,
    title_text="Interactive Multi-Channel Signals",
    showlegend=False,
    hovermode='x unified',
    xaxis4=dict(title="Time (s)"),
    template="plotly_white"
)

# === Save to HTML file ===
interactive_path = os.path.join(save_dir, "Interactive_Multi_Channel_Signals.html")
fig.write_html(interactive_path, auto_open=True)  # auto_open launches in browser

print(f"✅ Interactive plot saved and opened: {interactive_path}")
