#!/bin/bash
#SBATCH --job-name=epsilon_contamination_mcmc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=epsilon_contamination_%j.out
#SBATCH --error=epsilon_contamination_%j.err

# HPC Environment Setup for ε-Contamination MCMC Analysis
# 🖥️ Optimized for 32-core Linux systems with 32GB RAM

echo "🚀 Starting ε-Contamination MCMC Analysis on HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "================================"

# Load required modules (adjust for your HPC system)
module load python/3.11
module load gcc/11.2.0
# module load openmpi/4.1.1  # If needed for your environment

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
export PYTHONUNBUFFERED=1  # Ensure real-time output

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment (adjust path as needed)
# source venv/bin/activate
# conda activate climada_env

echo "🔬 Starting Jupyter notebook execution..."
echo "Expected runtime: 2-6 hours for full ε-contamination analysis"
echo "Progress will be displayed in real-time below:"
echo "================================"

# Execute the notebook
jupyter nbconvert --to notebook --execute 05_robust_bayesian_epsilon_contamination_notebook.ipynb \
    --output=epsilon_contamination_results_${SLURM_JOB_ID}.ipynb \
    --ExecutePreprocessor.timeout=21600 \
    --ExecutePreprocessor.kernel_name=python3

# Alternative: Run as Python script if preferred
# python -u 05_robust_bayesian_epsilon_contamination_script.py

echo "================================"
echo "🎉 ε-Contamination analysis completed!"
echo "Results saved to: epsilon_contamination_results_${SLURM_JOB_ID}.ipynb"
echo "Check results/epsilon_contamination_hpc/ for detailed outputs"
echo "Job completed at: $(date)"