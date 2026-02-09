#!/bin/bash
#SBATCH --job-name=airbench-bench
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:30:00

# ---------- logging setup ----------
REPO_DIR="$HOME/Lion_K_Optimizers"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU_TAG=""  # filled in after we know the GPU
LOG_DIR="$REPO_DIR/benchmark_logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Redirect SLURM stdout/stderr into the log directory
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
exec > >(tee "${LOG_DIR}/full_output.log") 2>&1

module load cuda/12.8.0 anaconda3/2024.06

# ---------- environment info ----------
echo "=========================================="
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Log dir:   ${LOG_DIR}"
echo "=========================================="
nvidia-smi
echo ""

python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    print('GPU:', gpu)
    print('CUDA version:', torch.version.cuda)
"
echo ""

cd "$REPO_DIR"

# ---------- CIFAR-10 data ----------
echo "Downloading CIFAR-10 data..."
python3 -c "
import torchvision
torchvision.datasets.CIFAR10('cifar10', download=True, train=True)
torchvision.datasets.CIFAR10('cifar10', download=True, train=False)
print('CIFAR-10 data ready')
"
echo ""

# ---------- benchmark runner ----------
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
    echo "Expected per-trial time on A100: $expected"
    echo "Log file:  $log_file"
    echo "============================================================"
    echo ""

    sed "s/range(200)/range($n_runs)/g; s/range(1000)/range($n_runs)/g" "$script" > "$tmp_script"

    # Run and tee to both console and per-script log
    python3 "$tmp_script" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    echo ""
    echo "[$script] Exit code: $exit_code"
    rm -f "$tmp_script"
}

# ---------- current records ----------
run_benchmark "airbench94_muon.py"              5  "~2.59s"
run_benchmark "airbench96_faster.py"            3  "~27.3s"

# ---------- legacy records ----------
run_benchmark "legacy/airbench94.py"            5  "~3.83s"
run_benchmark "legacy/airbench94_compiled.py"   5  "~3.09s"
run_benchmark "legacy/airbench95.py"            3  "~10.4s"
run_benchmark "legacy/airbench96.py"            3  "~34.7s"

# ---------- summary ----------
echo ""
echo "============================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "Date: $(date)"
echo "Logs saved to: ${LOG_DIR}"
echo "============================================================"
echo ""
ls -lh "$LOG_DIR"
