#!/usr/bin/env bash
# One-time setup on the Jetson Orin Nano (Ubuntu 24.04). Distro packages only; no pip, no CUDA required.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3 python3-numpy python3-pandas python3-scipy python3-sklearn \
     python3-matplotlib python3-psutil python3-tk git tmux fonts-liberation fonts-urw-base35
python3 -c "import numpy,pandas,scipy,sklearn,matplotlib,psutil,platform;print('python',platform.python_version(),'numpy',numpy.__version__,'pandas',pandas.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__)"
tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
head -1 /etc/nv_tegra_release 2>/dev/null || echo "no /etc/nv_tegra_release"
command -v tegrastats >/dev/null && echo "tegrastats: $(command -v tegrastats)" || echo "tegrastats NOT found -> power/GPU columns will be empty"
command -v nvpmodel >/dev/null && sudo nvpmodel -q || echo "nvpmodel NOT found"
python3 -c "import torch;print('optional PyTorch', torch.__version__)" 2>/dev/null || echo "PyTorch not installed (optional; only used by bench_latency.py for comparison)"
rm -rf ~/.cache/matplotlib
chmod +x get_data.sh run_experiment.sh
