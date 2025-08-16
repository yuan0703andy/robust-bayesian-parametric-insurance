# HPC ε-Contamination Analysis Usage Guide
# HPC ε-污染分析使用指南

Complete guide for running robust Bayesian ε-contamination analysis on HPC systems.

## 🚀 Quick Start

### 1. Basic HPC Submission
```bash
# Submit to SLURM HPC system
sbatch submit_hpc_epsilon_contamination.sh

# Check job status
squeue -u $USER

# Monitor output in real-time
tail -f epsilon_contamination_unleashed_<JOB_ID>.out
```

### 2. Alternative Execution Modes

#### 🔥 UNLEASHED Mode (Maximum Performance)
```bash
python 05_robust_bayesian_cpu_optimized.py --unleashed --verbose
```
- **32 chains × 32 cores**: Maximum parallelization
- **25,600+ samples**: Massive statistical power
- **Progressive sampling**: Optimized for HPC
- **Best for**: 32-core systems with 32GB+ RAM

#### 🚀 High-Performance Mode (Recommended)
```bash
python 05_robust_bayesian_cpu_optimized.py --high-performance --balanced-mode --verbose
```
- **16 chains × 32 cores**: Balanced performance
- **Good convergence**: Reliable results
- **Best for**: Most HPC use cases

#### 🛡️ Robust Mode (Maximum Stability)
```bash
python 05_robust_bayesian_cpu_optimized.py --unleashed --robust-sampling --verbose
```
- **Ultra-conservative settings**: Eliminates divergences
- **Slower but guaranteed**: Maximum reliability
- **Best for**: Difficult convergence scenarios

#### ⚡ Quick Test Mode (Debugging)
```bash
python 05_robust_bayesian_cpu_optimized.py --quick-test --high-performance --verbose
```
- **Fast execution**: 5-10 minutes
- **Minimal samples**: For testing only
- **Best for**: Debugging and validation

## 📊 Expected Performance

### 32-Core HPC System Performance

| Mode | Chains | Samples | Total Samples | Est. Time | Memory |
|------|--------|---------|---------------|-----------|--------|
| UNLEASHED | 32 | 800 | 25,600 | 3-6 hours | ~3.2GB |
| High-Perf | 16 | 1000 | 16,000 | 2-4 hours | ~1.6GB |
| Robust | 32 | 1500 | 48,000 | 6-12 hours | ~4.8GB |
| Quick Test | 6 | 300 | 1,800 | 5-10 min | ~600MB |

### Progressive Sampling Phases

#### Phase 1: Quick Exploration
- **Purpose**: Parameter space exploration
- **Duration**: 20-40% of total time
- **Chains**: Reduced for efficiency
- **Output**: `Multiprocess sampling (8-16 chains in X jobs)`

#### Phase 2: Precise Convergence
- **Purpose**: High-quality posterior sampling
- **Duration**: 60-80% of total time
- **Chains**: Full configuration
- **Output**: `Multiprocess sampling (16-32 chains in Y jobs)`

## 🎯 Key Features

### ε-Contamination Framework
- **Mathematical Model**: π(θ) = (1-ε)π₀(θ) + εq(θ)
- **Dual Analysis**: Theoretical estimation + MCMC inference
- **4-Level Hierarchy**: Complete robust Bayesian model (不簡化)
- **Physical Interpretation**: Normal weather + typhoon events

### HPC Optimizations
- **Auto-detection**: SLURM/PBS environment recognition
- **Thread Management**: Optimal OMP/MKL configuration
- **Memory Management**: Distributed across chains
- **Progress Monitoring**: Real-time HPC-friendly output

### Convergence Features
- **Extreme Reparameterization**: Avoids Neal's funnel
- **Progressive Sampling**: 2-phase strategy
- **Multiple ε Values**: Comprehensive model comparison
- **Strict Diagnostics**: R-hat < 1.01, ESS > 400

## 📁 Output Structure

```
results/robust_bayesian_cpu_comprehensive/
├── comprehensive_results.pkl          # Complete analysis results
├── analysis_report.txt               # Human-readable summary
└── [timestamp]_detailed_log.txt      # Detailed execution log
```

## 🔧 Customization Options

### Command Line Arguments

```bash
python 05_robust_bayesian_cpu_optimized.py \
    --unleashed \                    # Maximum parallelization
    --max-chains 32 \               # Override chain limits
    --max-cores 32 \                # Use all cores
    --verbose \                     # Detailed output
    --quick-test                    # Fast testing mode
```

### Environment Variables

```bash
# HPC system will automatically set these:
export SLURM_CPUS_PER_TASK=32      # Auto-detected
export OMP_NUM_THREADS=32          # Set by script

# Manual override if needed:
export PYMC_PROGRESS=True          # Enable progress bars
export PYTHONUNBUFFERED=1         # Real-time output
```

## 🔍 Monitoring and Debugging

### Real-time Monitoring
```bash
# Watch job output
tail -f epsilon_contamination_unleashed_<JOB_ID>.out

# Check resource usage
sstat $SLURM_JOB_ID

# Monitor specific metrics
watch -n 5 'squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"'
```

### Progress Indicators
```
🔬 Starting comprehensive ε-contamination analysis...
🚀 Phase 1: Quick exploration...
   Multiprocess sampling (16 chains in 16 jobs)
   100%|██████████| 700/700 [01:23<00:00, 8.41it/s]
🎯 Phase 2: Precise convergence...
   Multiprocess sampling (32 chains in 32 jobs)
   100%|██████████| 1400/1400 [02:45<00:00, 8.41it/s]
✅ ε-Contamination analysis complete!
```

### Troubleshooting

#### Common Issues
1. **Memory Error**: Reduce chains or enable quick-test
2. **Convergence Failure**: Use robust-sampling mode
3. **Slow Progress**: Check if all cores are utilized
4. **Module Errors**: Verify environment activation

#### Solutions
```bash
# Memory issues
python 05_robust_bayesian_cpu_optimized.py --high-performance --balanced-mode

# Convergence issues
python 05_robust_bayesian_cpu_optimized.py --unleashed --robust-sampling

# Environment issues
module list  # Check loaded modules
which python  # Verify Python path
```

## 🎯 Best Practices

### For 32-Core HPC Systems
1. **Start with UNLEASHED mode** for maximum performance
2. **Monitor first run closely** to validate configuration
3. **Use robust mode** if convergence issues occur
4. **Set appropriate time limits** (12+ hours for full analysis)

### For Development/Testing
1. **Always test with --quick-test first**
2. **Verify progress bars work** in your environment
3. **Check output file permissions** and disk space
4. **Validate module loading** before submission

### For Production Analysis
1. **Use UNLEASHED or high-performance mode**
2. **Enable verbose output** for detailed logs
3. **Set conservative time limits** (12-24 hours)
4. **Plan for result file transfer** and storage

## 📈 Expected Results

### Theoretical Analysis
- **ε-contamination estimates**: Multiple methods comparison
- **Consensus value**: Weighted average with uncertainty
- **Physical interpretation**: Typhoon vs normal weather proportion

### MCMC Analysis
- **Best model selection**: DIC/WAIC comparison
- **Convergence diagnostics**: R-hat, ESS, divergences
- **Posterior distributions**: High-quality samples
- **Model comparison**: Multiple ε values evaluated

### Comprehensive Report
- **Execution summary**: Performance metrics
- **Convergence assessment**: Diagnostic results
- **Scientific interpretation**: Weather pattern analysis
- **Recommendations**: Optimal ε-contamination level

## 🚀 Advanced Usage

### Custom Configuration
```bash
# Maximum performance for 64-core systems
python 05_robust_bayesian_cpu_optimized.py \
    --unleashed \
    --max-chains 64 \
    --max-cores 64 \
    --verbose

# Memory-constrained systems
python 05_robust_bayesian_cpu_optimized.py \
    --high-performance \
    --balanced-mode \
    --max-chains 8 \
    --verbose
```

### Batch Processing
```bash
# Submit multiple analyses
for mode in unleashed high-performance robust; do
    sbatch --job-name="epsilon_${mode}" submit_hpc_epsilon_contamination.sh
done
```

---

## 📞 Support

For technical issues:
1. Check HPC system documentation for module requirements
2. Verify Python environment and package installations
3. Review job output logs for specific error messages
4. Consider starting with quick-test mode for validation