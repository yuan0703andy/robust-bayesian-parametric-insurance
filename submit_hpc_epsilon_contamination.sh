#!/bin/bash
#SBATCH --job-name=epsilon_contamination_unleashed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=epsilon_contamination_unleashed_%j.out
#SBATCH --error=epsilon_contamination_unleashed_%j.err
#SBATCH --partition=compute  # Adjust for your HPC system

# =============================================================================
# HPC ε-Contamination UNLEASHED Analysis
# 🔥 Maximum Parallelization for 32-core Systems
# =============================================================================

echo "🔥 Starting UNLEASHED ε-Contamination Analysis on HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Time limit: 12 hours"
echo "================================"

# Load required modules (adjust for your HPC system)
module load python/3.11
module load gcc/11.2.0
# Add any other required modules

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# PyMC/PyTensor configuration for HPC
export JAX_PLATFORMS=cpu
export PYTENSOR_FLAGS="device=cpu,floatX=float64,mode=FAST_COMPILE,linker=py,allow_gc=True"
export MKL_THREADING_LAYER=GNU
export PYTHONHASHSEED=0

# Progress bar and output configuration
export PYMC_PROGRESS=True
export ARVIZ_RCPARAMS=display.progress=True
export PYTHONUNBUFFERED=1

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment (adjust path as needed)
# source ~/venv/bin/activate
# conda activate climada_env

echo "🔧 Environment configured:"
echo "   Python version: $(python --version)"
echo "   Working directory: $(pwd)"
echo "   Threads per process: $OMP_NUM_THREADS"

echo "================================"
echo "🚀 Starting UNLEASHED ε-Contamination Analysis..."
echo "Expected features:"
echo "   • 32 MCMC chains (maximum parallelization)"
echo "   • Progressive sampling with HPC optimization"
echo "   • Real-time progress monitoring"
echo "   • Comprehensive ε-contamination framework"
echo "   • Theoretical + MCMC dual analysis"
echo "================================"

# Run the analysis with UNLEASHED mode
python 05_robust_bayesian_cpu_optimized.py \
    --unleashed \
    --verbose \
    --max-chains 32 \
    --max-cores 32

# Alternative configurations (uncomment as needed):
# 
# # High-performance mode (recommended for first run)
# python 05_robust_bayesian_cpu_optimized.py \
#     --high-performance \
#     --balanced-mode \
#     --verbose
#
# # Quick test mode (for debugging)
# python 05_robust_bayesian_cpu_optimized.py \
#     --quick-test \
#     --high-performance \
#     --verbose
#
# # Robust sampling mode (if convergence issues)
# python 05_robust_bayesian_cpu_optimized.py \
#     --unleashed \
#     --robust-sampling \
#     --verbose

echo "================================"
echo "🎉 ε-Contamination analysis completed!"
echo "Results available in: results/robust_bayesian_cpu_comprehensive/"
echo "Job completed at: $(date)"
echo "Check output files for detailed results"
echo "================================"

# Optional: compress results for transfer
# tar -czf epsilon_contamination_results_${SLURM_JOB_ID}.tar.gz results/