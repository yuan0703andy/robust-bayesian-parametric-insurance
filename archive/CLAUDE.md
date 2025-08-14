# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

North Carolina Tropical Cyclone Parametric Insurance Analysis system implementing:
- CLIMADA-based catastrophic risk modeling for tropical cyclones
- Steinmann et al. (2023) compliant parametric insurance design (70 products × 5 radii)
- 4-level hierarchical Bayesian uncertainty quantification
- Robust decision theory with ε-contamination and Mixed Predictive Estimation
- GPU-accelerated MCMC sampling with dual-GPU support

## Key Commands

### Environment Setup
```bash
# Activate CLIMADA conda environment (REQUIRED for all scripts)
conda activate climada_env

# Verify installation
python -c "import climada; print('CLIMADA available')"
python -c "from insurance_analysis_refactored.core import ParametricInsuranceEngine; print('Framework ready')"
python -c "from bayesian import RobustBayesianAnalyzer; print('Bayesian framework ready')"
```

### Sequential Analysis Workflow (7 scripts, ~5 hours total)
```bash
# Run complete analysis pipeline
python 01_run_climada.py                    # ~10 min: Generate CLIMADA hazard/exposure data
python 02_spatial_analysis.py               # ~30 min: Cat-in-circle spatial analysis (5 radii)
python 03_insurance_product.py              # ~5 min: Generate 350 Steinmann products
python 04_traditional_parm_insurance.py     # ~45 min: Traditional RMSE evaluation
python 05_robust_bayesian_parm_insurance.py # ~2 hours: 4-level Bayesian analysis
python 06_sensitivity_analysis.py           # ~30 min: Weight sensitivity analysis  
python 07_technical_premium_analysis.py     # ~1 hour: Premium Pareto optimization
```

### GPU-Accelerated Execution
```bash
# For GPU-accelerated Bayesian analysis
python 05_robust_bayesian_parm_insurance_gpu.py  # Uses dual-GPU MCMC

# Test GPU setup
python test_gpu_setup.py
python demo_gpu_setup_usage.py

# Configure GPU environment
python -c "from bayesian.gpu_setup import setup_gpu_environment; setup_gpu_environment(enable_gpu=True)"
```

### Quick Testing
```bash
# Test framework components
python -c "from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products; products = generate_steinmann_2023_products(); print(f'Generated {len(products)} products')"

# Verify configuration
python -c "from config.settings import NC_BOUNDS, YEAR_RANGE; print(f'Analysis: NC {YEAR_RANGE[0]}-{YEAR_RANGE[1]}')"
```

## High-Level Architecture

### Core Workflow Pipeline (01-07 Scripts)
```
01_run_climada.py
├── Track Processing: IBTrACS (1980-2024) → TC hazard generation
├── Exposure: LitPop + OSM hospitals → ~$200M total exposure
└── Impact: Emanuel USA functions → Annual damages calculation

02_spatial_analysis.py  
├── Extract hospital coordinates from exposure
├── Cat-in-Circle analysis: 5 radii (15, 30, 50, 75, 100 km)
└── cKDTree optimization for efficient spatial queries

03_insurance_product.py
├── Generate Steinmann 2023 products: 70 threshold functions
├── 5 radii × 70 products = 350 total parametric products
└── Step payout functions with 25% increments

04_traditional_parm_insurance.py
├── Deterministic RMSE/MAE evaluation
├── Hospital-based payout configuration 
└── Traditional basis risk assessment

05_robust_bayesian_parm_insurance.py
├── 4-level hierarchical Bayesian model
├── ε-contamination robust analysis
├── Mixed Predictive Estimation (MPE)
└── CRPS/EDI/TSS probabilistic evaluation

06_sensitivity_analysis.py
├── Weight sensitivity for under/over penalties
└── Robustness analysis across parameter space

07_technical_premium_analysis.py
├── VaR and Solvency II calculations
├── Pareto frontier optimization
└── Market acceptability analysis
```

### Module Architecture

#### `insurance_analysis_refactored/core/` - Unified Insurance Framework
- **ParametricInsuranceEngine**: Product creation, Cat-in-Circle indices, payout calculation
- **SkillScoreEvaluator**: RMSE, MAE, CRPS, EDI, TSS with bootstrap CI
- **InsuranceProductManager**: Product lifecycle, portfolio optimization
- **TechnicalPremiumCalculator**: VaR, TVaR, Solvency II capital requirements
- **EnhancedCatInCircleAnalyzer**: cKDTree-optimized spatial analysis
- **saffir_simpson_products**: Steinmann 2023 compliant product generation

#### `bayesian/` - Robust Bayesian Methods  
- **parametric_bayesian_hierarchy**: 4-level hierarchical models (global → regional → local → event)
- **robust_model_ensemble_analyzer**: Model class Γ_f × prior class Γ_π analysis
- **epsilon_contamination**: π(θ) = (1-ε)π₀(θ) + εq(θ) contamination models
- **minimax_credible_intervals**: Γ-minimax robust credible regions
- **gpu_setup/**: Dual-GPU MCMC configuration with JAX/PyMC

#### `config/` - Configuration Management
- **settings.py**: NC bounds, years (1980-2024), resolution (0.1°)
- **hospital_based_payout_config**: Hospital exposure-based payouts

#### Data Processing Pipeline
- **data_processing/**: IBTrACS track filtering and processing
- **hazard_modeling/**: CLIMADA TC hazard generation
- **exposure_modeling/**: LitPop + OSM hospital extraction
- **impact_analysis/**: Emanuel USA impact function application

## Key Design Patterns

### Steinmann 2023 Compliance
Product generation follows exact academic standard:
- **70 threshold functions**: 25 single + 20 dual + 15 triple + 10 quadruple
- **5 radii**: 15km, 30km, 50km, 75km, 100km  
- **350 total products**: 5 radii × 70 functions
- **Step payouts**: 25% increments, no interpolation
- **Pure Cat-in-Circle**: Maximum wind speed, no spatial weighting

### Dual Evaluation Paradigm
```python
# Traditional: Deterministic CLIMADA → RMSE/MAE
climada_adapter = CLIMADAInputAdapter(tc_hazard, exposure, impact_func_set)
traditional_results = framework.analyze_with_adapter(climada_adapter)

# Bayesian: Probabilistic uncertainty → CRPS/EDI/TSS  
bayesian_adapter = BayesianInputAdapter(bayesian_simulation_results)
probabilistic_results = framework.analyze_with_adapter(bayesian_adapter)
```

### Performance Optimizations
- **cKDTree spatial indexing**: 100x speedup for Cat-in-Circle
- **Vectorized NumPy operations**: Throughout analysis pipeline
- **GPU acceleration**: Dual-GPU MCMC with JAX backend
- **Result caching**: Avoid recomputation of expensive operations

## Common Usage Patterns

### Steinmann 2023 Product Generation
```python
from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products

# Generate all 350 products (70 functions × 5 radii)
all_products = generate_steinmann_2023_products()
print(f"✅ Generated {len(all_products)} products")
```

### Bayesian Uncertainty Quantification  
```python
from bayesian.parametric_bayesian_hierarchy import ParametricBayesianHierarchy
from bayesian.epsilon_contamination import EpsilonContaminationModel

# 4-level hierarchical model
hierarchy = ParametricBayesianHierarchy(n_levels=4)
posterior = hierarchy.fit(losses, indices)

# ε-contamination robust analysis
contamination_model = EpsilonContaminationModel(epsilon=0.1)
robust_posterior = contamination_model.fit(losses)
```

### GPU-Accelerated MCMC
```python
from bayesian.gpu_setup import setup_gpu_environment, get_optimized_mcmc_config

# Configure dual-GPU environment
gpu_config = setup_gpu_environment(enable_gpu=True)
mcmc_config = get_optimized_mcmc_config(n_chains=4, n_samples=16000)

# Use in PyMC model
with pm.Model() as model:
    # Model definition
    trace = pm.sample(**mcmc_config)
```

## Key Data Files

### Input Data
- **IBTrACS tracks**: Downloaded automatically via CLIMADA API
- **OSM data**: `osm/osm_bpf/nc.osm.pbf` (North Carolina extract)
- **Config**: `config/settings.py` (NC bounds, years, parameters)

### Generated Outputs
- **`climada_complete_data.pkl`**: Hazard, exposure, impact data from 01
- **`results/spatial_analysis/`**: Cat-in-Circle indices from 02
- **`results/insurance_products/`**: 350 Steinmann products from 03
- **`results/robust_hierarchical_bayesian_analysis/`**: Bayesian results from 05

## Development Considerations

### Code Style
- **Cell execution**: Scripts use `# %%` markers for Jupyter-style development
- **Bilingual comments**: English/Chinese documentation throughout
- **Immediate execution**: Functional style with direct script running
- **No build system**: Manual dependency management via conda environment

### Performance Notes
- **Runtime**: Complete pipeline ~5 hours on HPC system
- **Memory**: 16+ GB RAM recommended for Bayesian analysis
- **GPU**: Optional but provides 3-4x speedup for MCMC
- **Parallelization**: Built-in support for multi-chain MCMC

### Research Focus
- **Academic standard**: Exact Steinmann 2023 compliance
- **Reproducibility**: Numbered scripts ensure consistent workflow
- **Extensibility**: Modular design allows easy method comparison
- **Publication ready**: Structured for academic paper supplementary materials