#!/bin/bash
#SBATCH --job-name=bench-legacy
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --constraint=a100@80g

# ===== This script benchmarks the LEGACY record scripts =====
# Target: A100-SXM4-80GB (d1028/d1029) with torch==2.1.2+cu118

set -e

REPO_DIR="$HOME/Lion_K_Optimizers"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_DIR/benchmark_logs/legacy_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
exec > >(tee "${LOG_DIR}/full_output.log") 2>&1

module load anaconda3/2024.06

# ---------- Setup conda env ----------
ENV_DIR="$HOME/.conda/envs/torch212_bench"
PYTHON="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

if [ ! -f "$PYTHON" ]; then
    echo "Creating conda env with Python 3.11..."
    conda create -y -p "$ENV_DIR" python=3.11 -q 2>&1 | tail -5
fi

if ! $PYTHON -c "import torch; assert torch.__version__.startswith('2.1')" 2>/dev/null; then
    echo "Installing torch==2.1.2+cu118..."
    $PIP install --upgrade pip -q 2>&1 | tail -1
    $PIP install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -5
fi

echo "Verifying torch install..."
$PYTHON -c "import torch; print('torch', torch.__version__)"

set +e

# ---------- Environment info ----------
echo ""
echo "=========================================="
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Log dir:   ${LOG_DIR}"
echo "=========================================="
nvidia-smi
echo ""

$PYTHON -c "
import torch, sys
print('Python:', sys.version)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
"
echo ""

cd "$REPO_DIR"

# ---------- CIFAR-10 data ----------
echo "Downloading CIFAR-10 data..."
$PYTHON -c "
import torchvision
torchvision.datasets.CIFAR10('cifar10', download=True, train=True)
torchvision.datasets.CIFAR10('cifar10', download=True, train=False)
print('CIFAR-10 data ready')
"
echo ""

# ---------- Benchmark runner ----------
run_benchmark() {
    local script=$1
    local n_runs=$2
    local expected=$3
    local label=$(basename "$script" .py)
    local tmp_script="/tmp/bench_${label}.py"
    local log_file="${LOG_DIR}/${label}.log"

    echo ""
    echo "============================================================"
    echo "BENCHMARKING: $script ($n_runs trials)"
    echo "Expected per-trial time on A100-80GB: $expected"
    echo "Log file:  $log_file"
    echo "============================================================"
    echo ""

    # Replace the main trial loop range
    sed "s/for run in range([0-9]*)/for run in range($n_runs)/g" "$script" > "$tmp_script"
    $PYTHON "$tmp_script" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    echo ""
    echo "[$script] Exit code: $exit_code"
    rm -f "$tmp_script"
}

# ---------- Legacy records (authored on A100-80GB, torch 2.1.2) ----------
run_benchmark "legacy/airbench94.py"   5  "~3.83s"
run_benchmark "legacy/airbench95.py"   3  "~10.4s"
run_benchmark "legacy/airbench96.py"   3  "~34.7s"

# ---------- Summary ----------
echo ""
echo "============================================================"
echo "ALL BENCHMARKS COMPLETE (legacy records)"
echo "Date: $(date)"
echo "Logs saved to: ${LOG_DIR}"
echo "============================================================"
echo ""
ls -lh "$LOG_DIR"
