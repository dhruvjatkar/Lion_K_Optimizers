#!/bin/bash
#SBATCH --job-name=lionk-search
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --nodelist=d1026

# =============================================================================
# Lion-K Spectral Search on A100-SXM4-40GB (d1026)
#
# Near-exclusive allocation: 56/64 CPUs, 480/500G mem.
# --exclusive is blocked by QOSMaxGRESPerJob (node has 3 GPUs, QOS caps at 1).
#
# The search explores convex spectral K-functions parameterised by:
#   (delta, delta_final, schedule, ns_steps, alpha)
# across soft_huber_k and power_k kernel families.
# The Muon baseline (h=1, polar factor) is ALWAYS the first candidate.
# =============================================================================

set -e  # exit on error during setup

REPO_DIR="$HOME/Lion_K_Optimizers"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_DIR/search_logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Tee all output to a master log
exec > >(tee "${LOG_DIR}/full_output.log") 2>&1

echo "============================================================"
echo "Lion-K Spectral Search"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Log dir:   ${LOG_DIR}"
echo "============================================================"

# ---------- Load conda ----------
module load anaconda3/2024.06

ENV_DIR="$HOME/.conda/envs/torch241_bench"
PYTHON="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

# Create env if missing
if [ ! -f "$PYTHON" ]; then
    echo "Creating conda env with Python 3.11..."
    conda create -y -p "$ENV_DIR" python=3.11 -q 2>&1 | tail -5
fi

# Install torch if missing or wrong version
if ! $PYTHON -c "import torch; assert torch.__version__.startswith('2.4')" 2>/dev/null; then
    echo "Installing torch==2.4.1+cu121..."
    $PIP install --upgrade pip -q 2>&1 | tail -1
    $PIP install torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5
fi

set +e  # don't exit on benchmark/search errors

# ---------- Environment verification ----------
echo ""
nvidia-smi
echo ""
$PYTHON -c "
import torch, sys
print('Python:        ', sys.version)
print('PyTorch:       ', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:           ', torch.cuda.get_device_name(0))
    print('CUDA version:  ', torch.version.cuda)
"
echo ""

cd "$REPO_DIR"

# ---------- Ensure CIFAR-10 is cached ----------
echo "Ensuring CIFAR-10 data is cached..."
$PYTHON -c "
import torchvision
torchvision.datasets.CIFAR10('cifar10', download=True, train=True)
torchvision.datasets.CIFAR10('cifar10', download=True, train=False)
print('CIFAR-10 data ready')
"
echo ""

# ---------- Verify imports work ----------
echo "Verifying research package imports..."
$PYTHON -c "
from research.lionk import KConfig, SearchConfig, MuonK, run_search
print('All imports OK')
cfg = KConfig(mode='power_k', alpha=0.3, delta=0.1).validate()
print(f'Sample config: {cfg}')
"
echo ""

# ---------- Run the search ----------
# The search:
#   1. Generates 80 candidates via 5D Sobol over (delta, delta_final, schedule, ns_steps, alpha)
#   2. Always starts with Muon baseline as candidate #0
#   3. F0 prunes candidates with >10% kernel overhead vs Muon
#   4. F1 screens with 2 seeds on 94% target
#   5. F2 validates top-20 with 3 seeds
#   6. F3 evaluates top-6 with 5 seeds + transfer to 95%/96%

STUDY_ID="lionk_d1026_${TIMESTAMP}"

echo "============================================================"
echo "Starting Lion-K search: ${STUDY_ID}"
echo "  F1 budget:   80 candidates"
echo "  F2 budget:   20 promoted"
echo "  F3 budget:   6 finalists"
echo "  Seeds:       F1=2, F2=3, F3=5"
echo "  Delta range: [0.001, 0.8]"
echo "  Alpha range: [0.1, 1.0]"
echo "  NS steps:    [2, 4]"
echo "============================================================"
echo ""

$PYTHON research/run_muonk_search.py \
    --study-id "$STUDY_ID" \
    --trials-root "research/lionk/trials" \
    --script94 "research/airbench94_muon_simple.py" \
    --script95 "research/airbench95_muonk_transfer.py" \
    --script96 "research/airbench96_muonk_transfer.py" \
    --target-acc 0.94 \
    --max-overhead-pct 10.0 \
    --delta-min 0.001 \
    --delta-max 0.8 \
    --alpha-min 0.1 \
    --alpha-max 1.0 \
    --ns-step-min 2 \
    --ns-step-max 4 \
    --f1-budget 80 \
    --f2-budget 20 \
    --f3-budget 6 \
    --seeds-f1 2 \
    --seeds-f2 3 \
    --seeds-f3 5 \
    --search-seed 1337 \
    --python-exe "$PYTHON" \
    2>&1 | tee "${LOG_DIR}/search_output.log"

SEARCH_EXIT=$?

echo ""
echo "============================================================"
echo "Search complete.  Exit code: ${SEARCH_EXIT}"
echo "Date: $(date)"
echo ""

# ---------- Show results summary ----------
STUDY_DIR="research/lionk/trials/studies/${STUDY_ID}"
if [ -f "${STUDY_DIR}/best.json" ]; then
    echo "BEST CANDIDATE:"
    cat "${STUDY_DIR}/best.json"
    echo ""
fi

if [ -f "${STUDY_DIR}/summary.md" ]; then
    echo "--- SUMMARY ---"
    cat "${STUDY_DIR}/summary.md"
fi

# Copy study artifacts to log dir for easy access
if [ -d "${STUDY_DIR}" ]; then
    cp -r "${STUDY_DIR}" "${LOG_DIR}/study_artifacts"
    echo ""
    echo "Study artifacts copied to: ${LOG_DIR}/study_artifacts"
fi

echo ""
echo "Logs saved to: ${LOG_DIR}"
echo "============================================================"
