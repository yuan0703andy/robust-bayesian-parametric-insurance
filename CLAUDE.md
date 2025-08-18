# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robust Bayesian parametric insurance analysis system for North Carolina tropical cyclone risk assessment. The repository implements:

- **CLIMADA-based catastrophic risk modeling** for tropical cyclones (1980-2024)
- **Steinmann et al. (2023) compliant parametric insurance design** (350 products: 70 threshold functions × 5 radii)
- **4-level hierarchical Bayesian uncertainty quantification** with robust analysis
- **GPU-accelerated MCMC sampling** with dual-GPU support
- **Advanced CRPS-based optimization** with basis risk minimization

## Key Commands

### Environment Setup (Required)
```bash
# Activate CLIMADA conda environment - REQUIRED for all operations
conda activate climada_env

# Verify installation
python -c "import climada; print('CLIMADA available')"
python -c "from insurance_analysis_refactored.core import ParametricInsuranceEngine; print('Framework ready')"
```

### Sequential Analysis Workflow (Complete Pipeline: ~5 hours)
```bash
# Run complete analysis pipeline in order
python 01_run_climada.py                    # ~10 min: Generate CLIMADA hazard/exposure data
python 02_spatial_analysis.py               # ~30 min: Cat-in-circle spatial analysis (5 radii)
python 03_insurance_product.py              # ~5 min: Generate 350 Steinmann products
python 04_traditional_parm_insurance.py     # ~45 min: Traditional RMSE evaluation
python 05_complete_integrated_framework.py  # ~2 hours: Robust Bayesian analysis
python 06_sensitivity_analysis.py           # ~30 min: Weight sensitivity analysis  
python 07_technical_premium_analysis.py     # ~1 hour: Premium Pareto optimization
```

### Modern Framework Usage
```bash
# Test unified framework
python -c "from insurance_analysis_refactored.core import UnifiedAnalysisFramework; framework = UnifiedAnalysisFramework(); print('Framework loaded')"

# Verify Steinmann product generation
python -c "from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products; products = generate_steinmann_2023_products(); print(f'Generated {len(products)} products')"

# Check configuration
python -c "from config.settings import NC_BOUNDS, YEAR_RANGE; print(f'Analysis: NC {YEAR_RANGE[0]}-{YEAR_RANGE[1]}')"
```

### GPU-Accelerated Execution (Optional)
```bash
# GPU setup verification
python -c "from robust_hierarchical_bayesian_simulation.mcmc_environment_config import setup_gpu_environment; setup_gpu_environment(enable_gpu=True)"

# Test GPU configuration
python visualize_bayesian_model.py
```

## High-Level Architecture

### Core Analysis Pipeline (Sequential Scripts 01-07)

#### 01_run_climada.py - CLIMADA Data Generation
- **Track Processing**: IBTrACS database (1980-2024) → TC hazard generation
- **Exposure**: LitPop methodology + OSM hospitals → ~$200M total exposure
- **Impact**: Emanuel USA impact functions → annual damages calculation
- **Output**: `results/climada_data/climada_complete_data.pkl`

#### 02_spatial_analysis.py - Spatial Analysis
- **Hospital Coordinates**: Extract from exposure data
- **Cat-in-Circle Analysis**: 5 radii (15, 30, 50, 75, 100 km)
- **Optimization**: cKDTree spatial indexing for 100x speedup
- **Output**: `results/spatial_analysis/cat_in_circle_results.pkl`

#### 03_insurance_product.py - Product Generation
- **Steinmann 2023 Products**: 70 threshold functions × 5 radii = 350 products
- **Compliance**: Exact academic standard with step payout functions
- **Output**: `results/insurance_products/products.pkl`

#### 04_traditional_parm_insurance.py - Traditional Analysis
- **Deterministic Evaluation**: RMSE/MAE assessment
- **Hospital-based Payouts**: Configuration-driven payout calculation
- **Output**: `results/traditional_analysis/traditional_results.pkl`

#### 05_complete_integrated_framework.py - Robust Bayesian Framework
- **4-level Hierarchical Model**: Global → regional → local → event
- **ε-contamination Analysis**: Robust decision theory
- **VI+CRPS Optimization**: Basis risk-aware variational inference
- **Mixed Predictive Estimation**: CRPS/EDI/TSS probabilistic evaluation

#### 06_sensitivity_analysis.py - Sensitivity Analysis
- **Weight Sensitivity**: Under/over penalty robustness analysis
- **Parameter Space**: Comprehensive robustness assessment

#### 07_technical_premium_analysis.py - Premium Optimization
- **VaR Calculations**: Solvency II capital requirements
- **Pareto Optimization**: Multi-objective optimization
- **Market Analysis**: Acceptability assessment

### Modular Architecture

#### `insurance_analysis_refactored/` - Unified Insurance Framework
**Core Modules:**
- **`ParametricInsuranceEngine`**: Product creation, Cat-in-Circle indices, payout calculation
- **`SkillScoreEvaluator`**: RMSE, MAE, CRPS, EDI, TSS with bootstrap confidence intervals
- **`InsuranceProductManager`**: Product lifecycle, portfolio optimization
- **`TechnicalPremiumCalculator`**: VaR, TVaR, Solvency II capital requirements
- **`EnhancedCatInCircleAnalyzer`**: cKDTree-optimized spatial analysis
- **`saffir_simpson_products`**: Steinmann 2023 compliant product generation

#### `robust_hierarchical_bayesian_simulation/` - Bayesian Methods
**Key Components:**
- **`parametric_bayesian_hierarchy`**: 4-level hierarchical models
- **`epsilon_contamination`**: π(θ) = (1-ε)π₀(θ) + εq(θ) contamination models
- **`basis_risk_vi`**: Basis risk-aware variational inference
- **`minimax_credible_intervals`**: Γ-minimax robust credible regions
- **`mcmc_environment_config`**: GPU-accelerated MCMC configuration

#### `config/` - Configuration Management
- **`settings.py`**: NC bounds, years (1980-2024), resolution (0.1°), parameters
- **`hospital_based_payout_config.py`**: Hospital exposure-based payout configuration

#### Data Processing Pipeline
- **`data_processing/`**: IBTrACS track filtering and processing
- **`hazard_modeling/`**: CLIMADA TC hazard generation with centroids
- **`exposure_modeling/`**: LitPop + OSM hospital extraction
- **`impact_analysis/`**: Emanuel USA impact function application

## Key Design Patterns

### Steinmann 2023 Compliance
The framework strictly follows academic standards:
- **70 threshold functions**: 25 single + 20 dual + 15 triple + 10 quadruple thresholds
- **5 radii**: 15km, 30km, 50km, 75km, 100km
- **350 total products**: Complete combinatorial product space
- **Step payouts**: 25% increments, no interpolation
- **Pure Cat-in-Circle**: Maximum wind speed within radius, no spatial weighting

### Dual Evaluation Paradigm
```python
# Traditional: Deterministic CLIMADA → RMSE/MAE
from insurance_analysis_refactored.core.input_adapters import CLIMADAInputAdapter
climada_adapter = CLIMADAInputAdapter(tc_hazard, exposure, impact_func_set)
traditional_results = framework.analyze_with_adapter(climada_adapter)

# Bayesian: Probabilistic uncertainty → CRPS/EDI/TSS
from insurance_analysis_refactored.core.input_adapters import BayesianInputAdapter  
bayesian_adapter = BayesianInputAdapter(bayesian_simulation_results)
probabilistic_results = framework.analyze_with_adapter(bayesian_adapter)
```

### Performance Optimizations
- **cKDTree spatial indexing**: 100x speedup for Cat-in-Circle calculations
- **Vectorized NumPy operations**: Throughout analysis pipeline
- **GPU acceleration**: Dual-GPU MCMC with JAX/PyMC backend
- **Result caching**: Automatic caching to avoid expensive recomputation

## Common Usage Patterns

### Unified Framework Analysis
```python
from insurance_analysis_refactored.core import UnifiedAnalysisFramework
import numpy as np

# Create framework and run comprehensive analysis
framework = UnifiedAnalysisFramework()
parametric_indices = np.random.uniform(20, 45, 100)
observed_losses = np.random.gamma(2, 5e8, 100)

# Execute complete analysis
results = framework.run_comprehensive_analysis(parametric_indices, observed_losses)

# Run Steinmann-compliant analysis
steinmann_results = framework.run_steinmann_analysis(parametric_indices, observed_losses)
```

### Robust Bayesian Analysis
```python
from robust_hierarchical_bayesian_simulation.parametric_bayesian_hierarchy import ParametricHierarchicalModel
from robust_hierarchical_bayesian_simulation.epsilon_contamination import EpsilonContaminationClass

# 4-level hierarchical model
hierarchy = ParametricHierarchicalModel(n_levels=4)
posterior = hierarchy.fit(losses, indices)

# ε-contamination robust analysis
contamination_model = EpsilonContaminationClass(epsilon=0.1)
robust_posterior = contamination_model.fit(losses)
```

### GPU-Accelerated MCMC
```python
from robust_hierarchical_bayesian_simulation.mcmc_environment_config import setup_gpu_environment

# Configure dual-GPU environment
gpu_config = setup_gpu_environment(enable_gpu=True)
print(f"GPU available: {gpu_config.gpu_available}")
print(f"Device count: {gpu_config.device_count}")
```

## Data Flow and Key Files

### Input Data
- **IBTrACS tracks**: Downloaded automatically via CLIMADA API
- **OSM data**: `osm/osm_bpf/nc.osm.pbf` (North Carolina building extract)
- **Configuration**: `config/settings.py` (NC bounds, years, parameters)

### Generated Outputs
```
results/
├── climada_data/climada_complete_data.pkl          # 01: CLIMADA hazard/exposure/impact
├── spatial_analysis/cat_in_circle_results.pkl      # 02: Spatial analysis results
├── insurance_products/products.pkl                 # 03: 350 Steinmann products
├── traditional_analysis/traditional_results.pkl    # 04: Traditional RMSE analysis
└── integrated_parametric_framework/                # 05: Bayesian analysis results
    ├── comprehensive_report.txt
    ├── product_details.csv
    └── product_rankings.csv
```

## Development Environment

### Dependencies
- **CLIMADA**: Climate risk assessment framework (requires conda environment)
- **PyMC**: Bayesian probabilistic programming
- **JAX**: GPU-accelerated numerical computing
- **NumPy/SciPy**: Scientific computing
- **Pandas**: Data manipulation
- **Geopandas**: Geospatial data processing

### Code Style
- **Cell-based execution**: Scripts use `# %%` markers for Jupyter-style development
- **Bilingual documentation**: English/Chinese comments throughout codebase
- **Functional programming**: Direct script execution with minimal state
- **No build system**: Manual dependency management via conda environment

### Performance Notes
- **Runtime**: Complete pipeline ~5 hours on standard HPC system
- **Memory**: 16+ GB RAM recommended for Bayesian analysis
- **GPU**: Optional but provides 3-4x speedup for MCMC sampling
- **Parallelization**: Built-in multi-chain MCMC support

## Research Innovations

### CRPS-Based Optimization
The framework implements world-first **Basis-Risk-Aware Variational Inference**:
- **Traditional**: Posterior sampling → product design → basis risk evaluation
- **Innovation**: VI ELBO directly optimizes basis risk: `L_BR(φ) = -E_q[CRPS_basis_risk] - KL`
- **Result**: End-to-end joint optimization with gradient-guided convergence

### Academic Compliance
- **Exact Steinmann 2023 implementation**: 350 products with academic standards
- **Reproducible workflow**: Numbered scripts ensure consistent execution
- **Publication-ready**: Structured for academic paper supplementary materials
- **Extensible design**: Modular architecture allows easy method comparison

## Troubleshooting

### Common Issues
1. **CLIMADA not available**: Ensure `conda activate climada_env` is executed
2. **GPU setup fails**: GPU acceleration is optional; framework falls back to CPU
3. **Memory issues**: Reduce Monte Carlo samples or use CPU-only mode
4. **Data loading errors**: Verify CLIMADA data directory permissions

### Environment Verification
```bash
# Test complete framework stack
python -c "
import climada
from insurance_analysis_refactored.core import UnifiedAnalysisFramework
from robust_hierarchical_bayesian_simulation.parametric_bayesian_hierarchy import ParametricHierarchicalModel
print('✅ All frameworks loaded successfully')
"
```