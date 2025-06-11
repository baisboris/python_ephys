# Neurophysiology Analysis Toolkit

This repository contains Python scripts for analyzing whole-cell patch-clamp electrophysiology and two-photon voltage imaging data. The pipeline supports:

- Spike detection and waveform characterization
- Subthreshold membrane potential averaging
- Current injection analysis (e.g., rheobase, sag ratio, input resistance)
- ΔF/F computation from denoised GEVI movies
- Stimulus-aligned trial averaging
- Batch processing and figure export

## Structure

- `01_Load_Files.py` — Data loading and preprocessing
- `02_Spike_Analysis.py` — Spike detection and AP features
- `03_Current_Injection.py` — Step protocol analysis
- `04_Stimulus_Alignment.py` — Sensory stimulus alignment
- `05_Voltage_Imaging.py` — ΔF/F and ROI-based imaging analysis

## Requirements

- Python ≥ 3.8  
- Dependencies listed in `requirements.txt`

## License

MIT License

## Author

Boris Bouazza Arostegui — Postdoctoral Research Scientist, Columbia University NeuroTechnology Center
