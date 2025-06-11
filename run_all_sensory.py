import subprocess

scripts = [
    "00_Load_Files.py",
    "01_Load_Files_Interactive.py",
    "02_Spontaneous_Firing.py",
    "03_Sensory_Stimuli.py",
    "07_real_time_plot.py"
]

processes = []
for script in scripts:
    print(f"🚀 Launching {script}")
    p = subprocess.Popen(["python", script])
    processes.append(p)

# Optional: Wait for all to complete
for p in processes:
    p.wait()

print("✅ All scripts finished.")
