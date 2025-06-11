# Neurophysiology Analysis Toolkit

This repository contains Python scripts for analyzing whole-cell patch-clamp electrophysiology and two-photon voltage imaging data. The pipeline supports:

- Spike detection and waveform characterization
- Subthreshold membrane potential averaging
- Current injection analysis (e.g., rheobase, sag ratio, input resistance)
- ΔF/F computation from denoised GEVI movies
- Stimulus-aligned trial averaging
- Batch processing and figure export

## Structure

- `00_Load_Files.py` — Load and parse raw electrophysiology and imaging data
- `01_Load_Files_Interactive.py` — Interactive version of the loader
- `02_Spontaneous_Firing.py` — Spike detection and spontaneous activity analysis
- `03_Sensory_Stimuli.py` — Airpuff stimulus detection and response alignment
- `04_Current_Injection.py` — Analysis of current step injection protocols
- `05_Ri_T_Sag.py` — Input resistance, time constant, and sag ratio analysis
- `06_current_injection_apfiring.py` — Firing rate, rheobase, and AP properties
- `07_real_time_plot.py` — Real-time plotting of ongoing recordings
- `run_all_sensory.py` — Batch process for sensory stimuli scripts
- `run_all_steps.py` — Batch process for step injection scripts

## Requirements

- Python ≥ 3.8  
- Dependencies listed in `requirements.txt`

## License

MIT License

## Author

Boris Bouazza Arostegui — Postdoctoral Research Scientist, Columbia University NeuroTechnology Center