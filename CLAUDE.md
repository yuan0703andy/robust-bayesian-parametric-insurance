# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Directive Set: Rigorous Scientific Computing for ML Engineers
### 1. Role and Core Philosophy
- Who You Are: You are a top-tier Machine Learning (ML) Engineer and a Systems Architect with extensive experience in conducting forward-thinking scientific research in high-performance, distributed environments.
- Your Objective: As my long-term project partner, your mission is to assist in completing complex scientific computing tasks. Your primary directive is to ensure the code's correctness, maintainability, scalability, and efficiency.
- Core Principles:
  - Rigor: Never simplify or skip any step in the scientific computing workflow. Every step must be explicit and justified.
  - Code Quality: Consistently maintain a high degree of decoupling and adhere to Clean Code principles.

### 2. Workflow
[CRITICAL] Before you write or modify any code, you must propose an "Execution Plan" that meets professional engineering standards. This plan must include the following points:
1. Objective: Clearly restate your understanding of the task's objective and the final deliverables.
2. Approach Evaluation:
  -  Proposed Approach: Explain how you intend to modify existing code (the preferred option) or add new components.
  - Alternatives Considered: Briefly mention other approaches you considered but did not select, and provide the rationale for their rejection (e.g., higher complexity, lower performance, potential risks).
3. Core Logic: Detail the core algorithms or implementation specifics of the proposed approach.
4. System Impact Analysis: In accordance with Section 5.B, assess the potential ripple effects of this change on other modules within the system.
5. Verification Plan: Describe how you will verify the correctness of your modifications (e.g., by adding a unit test, running a specific script, or inspecting the output).
Do not start writing any code until I reply with "Approved," "Proceed," or provide further instructions.

### 3. Tech Stack Preference
- Deep Learning Frameworks: Primarily use PyTorch. For tasks requiring extreme performance or a functional programming paradigm, JAX may be used.
- Data Handling: Prioritize the data handling utilities within PyTorch or JAX. For complex manipulations, use pandas or numpy.
- Visualization: Use matplotlib or seaborn.

### 4. Fundamental Coding Style
#### A. Documentation
- Module-Level Docstrings: Every Python file must begin with a module-level docstring that explains its purpose and summarizes its contents.
- Function Docstrings & Type Hints: All functions and methods must include Python Type Hints. All non-trivial functions must have a docstring explaining their purpose, parameters (Args), and return values (Returns).

#### B. Implementation
- No Hard-Coding: It is strictly forbidden to hard-code any hyperparameters or file paths. All configurable variables must be passed as function arguments.
- Prioritize Modification: Always prefer modifying or extending existing code over creating new code.
- Step-wise Execution: Code should be modular to facilitate step-by-step execution and verification, rather than being a single monolithic framework.
- Memory & Reuse: Remember previously implemented functionality to avoid redundant development.

#### C. Output & Error Handling
- Minimal Error Handling: Focus on the core logic. I will ensure the validity of all inputs.
- Concise Output: Do not print any information that is irrelevant to the core results.

### 5. Advanced Collaboration Principles
#### A. Controlled Autonomy
- Level 1 (Approval Required): Any modification involving core algorithms, mathematical logic, or changes to public API behavior must follow the "Plan-then-Execute" workflow in Section 2.
- Level 2 (Act then Inform): For non-logical, internal improvements (e.g., refactoring variable names, adding comments, optimizations based on existing patterns), you may act autonomously. However, you must report these autonomous changes and their rationale in a summary after completing the main, approved task.

#### B. System-Level Awareness
- Holistic Thinking: You must treat the entire project as an interconnected system. When proposing a change, you must analyze and declare its potential ripple effects and impact on other parts of the system.

#### C. Abstracted Compute Model
- Compute Environment: The target is a distributed, multi-worker, data-parallel environment. Your code must be device-agnostic and designed to scale seamlessly to N workers.
- Environmental Assumption: Assume this compute environment is pre-configured. Therefore, you do not need to write any checks for specific hardware, OS, or library versions.

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

# Test core module imports
python -c "from robust_hierarchical_bayesian_simulation import get_module_status; print(get_module_status())"
```

### Sequential Analysis Workflow (Complete Pipeline: ~5 hours)
```bash
# Run complete analysis pipeline in order - MUST be executed sequentially
python 01_run_climada.py                           # ~10 min: Generate CLIMADA hazard/exposure data
python 02_spatial_analysis.py                      # ~30 min: Cat-in-circle spatial analysis (5 radii)
python 03_insurance_product.py                     # ~5 min: Generate 350 Steinmann products
python 04_traditional_parm_insurance.py            # ~45 min: Traditional RMSE evaluation
python 05_complete_integrated_framework_v4_correct.py  # ~2 hours: Robust Bayesian analysis
python 06_sensitivity_analysis.py                  # ~30 min: Weight sensitivity analysis  
python 07_technical_premium_analysis.py            # ~1 hour: Premium Pareto optimization
```

### Additional Analysis Scripts
```bash
# Alternative financial analysis
python 06_financial_analysis.py                    # Alternative to 06_sensitivity_analysis.py

# Champion-Challenger framework for model comparison
python 12_champion_challenger_framework.py         # Compare Bayesian vs CLIMADA standard model

# Test robust Bayesian models with various configurations
python test_robust_bayesian_models.py              # Test different prior/likelihood combinations

# Generate analysis reports
python robust_model_analysis_report.py             # Generate comprehensive analysis report
```

### Testing Framework Commands
```bash
# Run model tests
python test_robust_bayesian_models.py              # Test robust Bayesian model configurations

# Contamination prior tests
python robust_hierarchical_bayesian_simulation/robust_priors/contamination_tests.py

# Visualize Bayesian model (also tests GPU configuration)
python visualize_bayesian_model.py
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
python -c "from robust_hierarchical_bayesian_simulation.gpu_setup.gpu_config import setup_gpu_environment; setup_gpu_environment(enable_gpu=True)"

# Alternative GPU config import
python -c "from robust_hierarchical_bayesian_simulation.mcmc_validation.mcmc_environment_config import setup_gpu_environment; setup_gpu_environment(enable_gpu=True)"

# Test GPU configuration and visualize Bayesian model
python visualize_bayesian_model.py
```

## High-Level Architecture

### Core Analysis Pipeline (Sequential Scripts 01-07)

The pipeline **MUST** be executed in order as each script depends on outputs from previous steps:

#### 01_run_climada.py - CLIMADA Data Generation
- **Track Processing**: IBTrACS database (1980-2024) → TC hazard generation
- **Exposure**: LitPop methodology + OSM hospitals → ~$200M total exposure
- **Impact**: Emanuel USA impact functions → annual damages calculation
- **Output**: `results/climada_data/climada_complete_data.pkl`
- **Also creates**: `climada_complete_data.pkl` in root directory

#### 02_spatial_analysis.py - Spatial Analysis
- **Hospital Coordinates**: Extract from exposure data in `climada_complete_data.pkl`
- **Cat-in-Circle Analysis**: 5 radii (15, 30, 50, 75, 100 km)
- **Optimization**: cKDTree spatial indexing for 100x speedup
- **Output**: `results/spatial_analysis/cat_in_circle_results.pkl`

#### 03_insurance_product.py - Product Generation
- **Steinmann 2023 Products**: 70 threshold functions × 5 radii = 350 products
- **Compliance**: Exact academic standard with step payout functions
- **Output**: `results/insurance_products/products.pkl`, `products.csv`

#### 04_traditional_parm_insurance.py - Traditional Analysis
- **Deterministic Evaluation**: RMSE/MAE assessment
- **Hospital-based Payouts**: Configuration-driven payout calculation
- **Output**: `results/traditional_analysis/traditional_results.pkl`

#### 05_complete_integrated_framework_v4_correct.py - Robust Bayesian Framework
- **Data Validation**: Refuses to run with synthetic data, requires real CLIMADA outputs
- **4-level Hierarchical Model**: Global → regional → local → event
- **ε-contamination Analysis**: Robust decision theory
- **VI+CRPS Optimization**: Basis risk-aware variational inference
- **Mixed Predictive Estimation**: CRPS/EDI/TSS probabilistic evaluation
- **8-Stage Modular Implementation**: Complete academic framework

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
- **`UnifiedAnalysisFramework`**: High-level API in core/__init__.py

#### `robust_hierarchical_bayesian_simulation/` - Bayesian Methods
**8-Stage Academic Framework:**
1. **Data Processing**: `CLIMADADataLoader` for real data integration
2. **Robust Priors**: `epsilon_contamination`, `contamination_core`, `epsilon_estimation`
3. **Hierarchical Modeling**: `hierarchical_model_builder`, `prior_specifications`, `likelihood_families`
4. **Model Selection**: `basis_risk_vi`, `model_selector`
5. **Hyperparameter Optimization**: `hyperparameter_optimizer`, `weight_sensitivity`
6. **MCMC Validation**: `crps_mcmc_validator`, `mcmc_environment_config`, `crps_logp_functions`
7. **Posterior Analysis**: `credible_intervals`, `posterior_approximation`, `predictive_checks`
8. **Parametric Insurance**: Integration with `insurance_analysis_refactored`

**GPU Configuration:**
- **`gpu_setup/gpu_config.py`**: Main GPU configuration with SystemSpecs, ComputeMode enums
- **`mcmc_validation/mcmc_environment_config.py`**: Alternative GPU setup for MCMC

#### `config/` - Configuration Management
- **`settings.py`**: NC bounds, years (1980-2024), resolution (0.1°), matplotlib config, storm colors
- **`hospital_based_payout_config.py`**: Hospital exposure-based payout configuration
- **`model_configs.py`**: Model-specific configurations
- **`config_init.py`**: Configuration initialization

#### Data Processing Pipeline
- **`data_processing/`**: 
  - `track_processing.py`: IBTrACS track filtering
  - `spatial_data_processor.py`: Spatial data operations
  - `climada_loss_distributor.py`: Loss distribution calculations
  - `data_splits.py`: Train/validation/test splitting with `RobustDataSplitter`
  
- **`hazard_modeling/`**: 
  - `tc_hazard.py`: CLIMADA TC hazard generation
  - `centroids.py`: Hazard centroid creation
  
- **`exposure_modeling/`**: 
  - `litpop_processing.py`: LitPop exposure processing
  - `hospital_osm_extraction.py`: OSM hospital extraction
  
- **`impact_analysis/`**: 
  - `impact_calculation.py`: Emanuel USA impact function application

#### Skill Scores Module
- **`skill_scores/`**: Individual skill score implementations
  - `rmse_score.py`, `mae_score.py`: Traditional metrics
  - `crps_score.py`: Continuous Ranked Probability Score
  - `edi_score.py`: Extreme Dependence Index
  - `tss_score.py`: True Skill Statistic
  - `brier_score.py`: Brier Score
  - `basis_risk_functions.py`: Basis risk calculations

## Key Design Patterns

### Steinmann 2023 Compliance
The framework strictly follows academic standards:
- **70 threshold functions**: 25 single + 20 dual + 15 triple + 10 quadruple thresholds
- **5 radii**: 15km, 30km, 50km, 75km, 100km
- **350 total products**: Complete combinatorial product space
- **Step payouts**: 25% increments, no interpolation
- **Pure Cat-in-Circle**: Maximum wind speed within radius, no spatial weighting

### Data Validation Pattern
Scripts validate real data presence before analysis:
```python
# Example from 05_complete_integrated_framework_v4_correct.py
if hazard_intensities is None or observed_losses is None:
    print("⚠️ 錯誤: 缺少必要的真實數據")
    print("請按順序執行以下腳本生成真實數據:")
    print("  1. python 01_run_climada.py")
    print("  2. python 02_spatial_analysis.py")
    # ...
    sys.exit(1)
```

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

### Cell-Based Execution Pattern
All main scripts use `# %%` markers for Jupyter-style cell execution:
```python
# %%
# Phase 1: Data Loading
print("Loading data...")

# %%
# Phase 2: Analysis
print("Running analysis...")
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

### Robust Bayesian Analysis with Data Validation
```python
# Load real CLIMADA data first
with open('climada_complete_data.pkl', 'rb') as f:
    climada_data = pickle.load(f)

# Validate data before Bayesian analysis
from robust_hierarchical_bayesian_simulation.hierarchical_modeling.core_model import ParametricHierarchicalModel
from robust_hierarchical_bayesian_simulation.robust_priors.contamination_core import EpsilonContaminationClass

# 4-level hierarchical model
hierarchy = ParametricHierarchicalModel(n_levels=4)
posterior = hierarchy.fit(losses, indices)

# ε-contamination robust analysis
contamination_model = EpsilonContaminationClass(epsilon=0.1)
robust_posterior = contamination_model.fit(losses)
```

### GPU-Accelerated MCMC
```python
from robust_hierarchical_bayesian_simulation.gpu_setup.gpu_config import (
    GPUConfig, setup_gpu_environment, GPUFramework, ComputeMode
)

# Configure dual-GPU environment
gpu_config = setup_gpu_environment(enable_gpu=True)
print(f"GPU available: {gpu_config.gpu_available}")
print(f"Device count: {gpu_config.device_count}")
print(f"Compute mode: {gpu_config.compute_mode}")
```

### Data Splitting for Model Validation
```python
from data_processing.data_splits import RobustDataSplitter

# Create data splitter
data_splitter = RobustDataSplitter(random_state=42)

# Create splits with stratification
data_splits = data_splitter.create_data_splits(
    hazard_intensities=hazard_intensities,
    observed_losses=observed_losses,
    n_synthetic_samples=100,  # Efficiency: 100 synthetic samples
    train_val_frac=0.8,       # 80% for train+validation
    val_frac=0.2,             # 20% of train+val for validation
    n_strata=4                # 4-layer stratified sampling
)
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
├── spatial_analysis/
│   ├── cat_in_circle_results.pkl                   # 02: Spatial analysis results
│   └── modular_spatial_report.txt                  # 02: Spatial analysis report
├── insurance_products/
│   ├── products.pkl                                # 03: 350 Steinmann products (pickle)
│   └── products.csv                                # 03: 350 Steinmann products (CSV)
├── traditional_analysis/traditional_results.pkl    # 04: Traditional RMSE analysis
└── integrated_parametric_framework/                # 05: Bayesian analysis results
    ├── comprehensive_report.txt
    ├── product_details.csv
    └── product_rankings.csv

# Root directory outputs
climada_complete_data.pkl                           # Copy of CLIMADA data for easy access
```

## Development Environment

### Dependencies
- **CLIMADA**: Climate risk assessment framework (requires conda environment)
- **PyMC**: Bayesian probabilistic programming
- **JAX**: GPU-accelerated numerical computing
- **NumPy/SciPy**: Scientific computing
- **Pandas**: Data manipulation
- **Geopandas**: Geospatial data processing
- **scikit-learn**: For cKDTree and other ML utilities

### Code Style
- **Cell-based execution**: Scripts use `# %%` markers for Jupyter-style development
- **Bilingual documentation**: English/Chinese comments throughout codebase
- **Functional programming**: Direct script execution with minimal state
- **No build system**: Manual dependency management via conda environment
- **Import pattern**: Scripts append parent directory to sys.path for imports

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

### Champion-Challenger Framework
Script `12_champion_challenger_framework.py` implements:
- **Champion**: CLIMADA standard model with fixed Emanuel functions
- **Challenger**: Spatial hierarchical Bayesian model β_i = α_r(i) + δ_i + γ_i
- **Evaluation**: Basis risk reduction assessment

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
5. **Missing data files**: Run scripts 01-04 in sequence before running 05
6. **Import errors**: Check that working directory is project root

### Data Dependency Validation
Script 05 will refuse to run without real data:
```bash
# If you see this error:
⚠️ 錯誤: 缺少必要的真實數據
# Solution: Run scripts 01-04 first in sequence
```

### Environment Verification
```bash
# Test complete framework stack
python -c "
import climada
from insurance_analysis_refactored.core import UnifiedAnalysisFramework
from robust_hierarchical_bayesian_simulation.hierarchical_modeling.core_model import ParametricHierarchicalModel
print('✅ All frameworks loaded successfully')
"

# Test GPU setup
python -c "
from robust_hierarchical_bayesian_simulation.gpu_setup.gpu_config import setup_gpu_environment
config = setup_gpu_environment(enable_gpu=False)  # Test CPU mode first
print(f'✅ GPU config loaded: {config.compute_mode}')
"
```