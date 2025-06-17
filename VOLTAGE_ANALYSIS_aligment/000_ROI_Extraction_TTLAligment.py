#!/usr/bin/env python3

"""
Script: ROI Signal Extraction from Denoised TIFF T-series
Author: Boris Bouazza-Arostegui (Updated)
Date: May 2025

Includes: Baseline normalization using 20th percentile to compute ΔF/F₀
Note: No explicit photobleaching correction is applied
"""

import os
import numpy as np
import pandas as pd
from tifffile import imread
from read_roi import read_roi_zip
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==== METADATA PARSER ====
def parse_pv_metadata(xml_file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    root = tree.getroot()

    frame_period, pix_x, pix_y = None, None, None
    for elem in root.iter('PVStateValue'):
        key = elem.attrib.get('key')
        if key == 'framePeriod': frame_period = float(elem.attrib.get('value'))
        elif key == 'pixelsPerLine': pix_x = int(elem.attrib.get('value'))
        elif key == 'linesPerFrame': pix_y = int(elem.attrib.get('value'))

    frame_rate = round(1.0 / frame_period, 4)
    frame_count = len(root.findall('.//Frame'))
    total_time_sec = round(frame_count * frame_period, 2)
    return frame_rate, pix_x, pix_y, frame_count, total_time_sec

# ==== CONFIGURATION ====
tif_path = r'E:/DATA/29/20250526_CD1_29001_58bruker/TSeries-01072025-1149-002/motioncorr_output/SUPPORT/20250612_190608/denoised.tif'
roi_zip = r'E:/DATA/29/20250526_CD1_29001_58bruker/TSeries-01072025-1149-002/RoiSet_soma_Background.zip'
xml_path = r'E:/DATA/29/20250526_CD1_29001_58bruker/TSeries-01072025-1149-002/TSeries-01072025-1149-002.xml'
ephys_path = r"E:/DATA/29/20250526_CD1_29001_58bruker/TSeries-01072025-1149-002/TSeries-01072025-1149-002_Cycle00001_VoltageRecording_001.csv"
output_dir = os.path.join(os.path.dirname(tif_path), '00_ROI_Extraction_TTL_aligment')
os.makedirs(output_dir, exist_ok=True)

# ==== LOAD METADATA ====
frame_rate, pix_x, pix_y, frame_count, total_time_sec = parse_pv_metadata(xml_path)
print(f"🧪 Frame rate: {frame_rate} Hz | Image size: {pix_x}x{pix_y} | Duration: {total_time_sec}s")

# ==== LOAD AIRPUFF METADATA ====
df_ephys = pd.read_csv(ephys_path)
df_ephys.columns = df_ephys.columns.str.strip()

ephys_time = df_ephys["Time(ms)"] / 1000
airpuff = df_ephys["AIRPUFF"]

# Detect TTL airpuff onsets
airpuff_threshold = 0.5
airpuff_onsets = np.where(np.diff((airpuff > airpuff_threshold).astype(int)) == 1)[0]
airpuff_times = ephys_time.iloc[airpuff_onsets].values

print(f"💨 Detected {len(airpuff_times)} airpuff stimuli.")

# Plot TTL detection and save
plt.figure(figsize=(12, 4))
plt.plot(ephys_time, airpuff, label='AIRPUFF TTL', color='black')
plt.plot(ephys_time[airpuff_onsets], airpuff.iloc[airpuff_onsets], 'ro', label='Detected Onsets')
plt.xlabel("Time (s)")
plt.ylabel("TTL Voltage")
plt.title("Detected Airpuff TTL Events")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(output_dir, "airpuff_ttl_detection.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"📸 Saved TTL plot to: {plot_path}")

# ==== LOAD TIFF DATA ====
print("📂 Loading denoised TIFF movie...")
movie = imread(tif_path)
T, H, W = movie.shape
std_img = np.std(movie, axis=0).astype(np.float32)
print(f"✅ Movie shape: {movie.shape} (frames, height, width)")


# Confirm video duration matches metadata
movie_duration_sec = round(T / frame_rate, 2)

# ==== CLIP AIRPUFF TIMES TO TIFF DURATION ====
airpuff_times = airpuff_times[airpuff_times <= movie_duration_sec]

# ==== GENERATE FRAME TIMESTAMPS ====
timestamps = np.arange(T) / frame_rate

# ==== COUNT AND PLOT CLIPPED AIRPUFF TIMES ====
print(f"📏 {len(airpuff_times)} airpuff stimuli remain after clipping to TIFF duration.")

plt.figure(figsize=(12, 4))
plt.eventplot(airpuff_times, orientation='horizontal', colors='red', lineoffsets=1, linelengths=0.8)
plt.xlabel("Time (s)")
plt.yticks([])
plt.title("Clipped Airpuff TTL Onsets Within TIFF Window")
plt.xlim(0, movie_duration_sec)
plt.tight_layout()
clipped_plot_path = os.path.join(output_dir, "airpuff_ttl_clipped.png")
plt.savefig(clipped_plot_path, dpi=300)
plt.close()
print(f"🖼️ Saved clipped TTL eventplot to: {clipped_plot_path}")

print(f"🕒 Video duration from TIFF: {movie_duration_sec} s")
if abs(movie_duration_sec - total_time_sec) > 0.1:
    print(f"⚠️ Duration mismatch: XML metadata = {total_time_sec}s, TIFF = {movie_duration_sec}s")
else:
    print("✅ TIFF video duration matches XML metadata.")

# ==== LOAD ROIs ====
rois = read_roi_zip(roi_zip)
print(f"✅ Loaded {len(rois)} ROIs")

# ==== EXTRACT ROI SIGNALS ====
roi_signals = pd.DataFrame({"Time (s)": timestamps})

for i, (name, roi_dict) in enumerate(rois.items()):
    mask = None

    # Handle polygon/freehand ROIs
    if 'x' in roi_dict and 'y' in roi_dict:
        x, y = np.array(roi_dict['x']), np.array(roi_dict['y'])
        coords = np.vstack([x, y]).T
        roi_contour = pcv.roi.custom(img=std_img, vertices=coords)
        mask = pcv.roi.roi2mask(std_img, roi_contour) // 255

    # Handle rectangular ROIs
    elif all(k in roi_dict for k in ('top', 'left', 'width', 'height')):
        mask = np.zeros((H, W), dtype=np.uint8)
        top = int(roi_dict['top'])
        left = int(roi_dict['left'])
        height = int(roi_dict['height'])
        width = int(roi_dict['width'])
        mask[top:top+height, left:left+width] = 1

    else:
        print(f"❌ ROI {name} missing coordinate or rectangular keys. Skipping.")
        continue

    if mask.sum() < 5:
        print(f"⚠️ ROI {name} too small. Skipping.")
        continue

    mask_bool = mask.astype(bool)
    trace = movie[:, mask_bool].mean(axis=1)
    roi_signals[f"ROI_{i+1}"] = trace
    print(f"📈 Extracted signal for ROI_{i+1} ({name})")

# Save raw intensity traces (like ImageJ Z-axis profile) (like ImageJ Z-axis profile)
trace_path = os.path.join(output_dir, "roi_zaxis_profile_traces.csv")
roi_signals.to_csv(trace_path, index=False)
print(f"💾 Saved ROI Z-axis intensity profiles to: {trace_path}")

# ==== PLOT ROI TRACES ====
plt.figure(figsize=(12, 6))
for col in roi_signals.columns[1:]:  # skip "Time (s)"
    plt.plot(roi_signals["Time (s)"], roi_signals[col], label=col)

plt.xlabel("Time (s)")
plt.ylabel("Mean Intensity (AU)")
plt.title("ROI Intensity Over Time (Z-axis Profile)")
plt.legend(fontsize="small", loc="upper right")
plt.tight_layout()
plot_out = os.path.join(output_dir, "roi_zaxis_profile_traces.png")
plt.savefig(plot_out, dpi=300)
plt.close()
print(f"🖼️ Saved Z-axis profile plot to: {plot_out}")

# ==== TRIAL-AVERAGED ROI RESPONSES ====
pre_sec = 2  # seconds before airpuff
post_sec = 5  # seconds after airpuff
n_frames_pre = int(pre_sec * frame_rate)
n_frames_post = int(post_sec * frame_rate)
window_length = n_frames_pre + n_frames_post

trial_averages = {}
trial_traces = {}
for col in roi_signals.columns[1:]:  # skip "Time (s)"
    traces = []
    for t in airpuff_times:
        idx = np.searchsorted(timestamps, t)
        start = idx - n_frames_pre
        end = idx + n_frames_post
        if start >= 0 and end < len(timestamps):
            trace = roi_signals[col].values[start:end]
            traces.append(trace)
    if traces:
        traces = np.array(traces)
        trial_traces[col] = traces
        trial_averages[col] = traces.mean(axis=0)

trial_df = pd.DataFrame(trial_averages)
trial_df.insert(0, "Time (s)", np.linspace(-pre_sec, post_sec, window_length))

# Save and plot trial-averaged responses
trial_csv = os.path.join(output_dir, "trial_averaged_roi_responses.csv")
trial_df.to_csv(trial_csv, index=False)
print(f"📊 Saved trial-averaged ROI responses to: {trial_csv}")

# Plot with individual trials in gray
plt.figure(figsize=(12, 6))
for col in trial_df.columns[1:]:
    if col in trial_traces:
        traces = trial_traces[col]
        for tr in traces:
            plt.plot(trial_df["Time (s)"], tr, color='gray', alpha=0.3, linewidth=0.7)
        mean = traces.mean(axis=0)
        sem = traces.std(axis=0) / np.sqrt(traces.shape[0])
        plt.plot(trial_df["Time (s)"], mean, label=col, linewidth=2)
        plt.fill_between(trial_df["Time (s)"], mean - sem, mean + sem, alpha=0.2)

avg_plot = os.path.join(output_dir, "trial_averaged_roi_responses.png")
plt.savefig(avg_plot, dpi=300)
plt.close()
print(f"🖼️ Saved trial-averaged ROI plot to: {avg_plot}")
print(f"🖼️ Saved trial-averaged ROI plot to: {avg_plot}")
print(f"🖼️ Saved Z-axis profile plot to: {plot_out}")
