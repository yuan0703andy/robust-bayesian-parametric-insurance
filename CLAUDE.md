# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

North Carolina Tropical Cyclone Parametric Insurance Analysis system - an academic research codebase combining CLIMADA catastrophic modeling with advanced Bayesian uncertainty quantification for robust parametric insurance design. This implements the Steinmann et al. (2023) framework for parametric insurance with robust Bayesian decision theory extensions.

## Key Commands

### Sequential Workflow Execution
The analysis follows a numbered script workflow (01-07) that can be run sequentially:

```bash
# Activate CLIMADA environment first
conda activate climada_env  # Required for all scripts

# Sequential analysis workflow:
python 01_run_climada.py                    # Generate base CLIMADA data
python 02_spatial_analysis.py               # Spatial cat-in-circle analysis  
python 03_insurance_product.py              # Generate parametric products
python 04_traditional_parm_insurance.py     # Traditional RMSE-based analysis
python 05_robust_bayesian_parm_insurance.py # Robust Bayesian analysis
python 06_sensitivity_analysis.py           # Weight sensitivity analysis
python 07_technical_premium_analysis.py     # Technical premium optimization
```

### Individual Component Testing
```bash
# Test core framework components
python -c "from insurance_analysis_refactored.core import ParametricInsuranceEngine; print('Framework loaded successfully')"

# Test Bayesian components  
python -c "from bayesian import RobustBayesianAnalyzer; print('Bayesian framework ready')"

# Verify configuration
python -c "from config.settings import NC_BOUNDS; print(f'NC Bounds: {NC_BOUNDS}')"
```

### Module Development
Since there are no standard build files (requirements.txt, setup.py), development follows:
- Direct script execution for analysis workflows
- Module imports as libraries for reusable components
- Manual dependency management (requires CLIMADA environment)
- Interactive development through Jupyter-style cell execution

## High-Level Architecture

### Core Analysis Components

1. **Sequential Analysis Pipeline** (Scripts 01-07)
   - **`01_run_climada.py`**: CLIMADA data generation with IBTRACS track processing
   - **`02_spatial_analysis.py`**: Cat-in-circle spatial analysis with multiple radii
   - **`03_insurance_product.py`**: Steinmann 2023-compliant product generation (70 products × 5 radii = 350 total)
   - **`04_traditional_parm_insurance.py`**: Traditional RMSE-based parametric insurance evaluation
   - **`05_robust_bayesian_parm_insurance.py`**: 4-level hierarchical Bayesian analysis with uncertainty quantification
   - **`06_sensitivity_analysis.py`**: Weight sensitivity analysis for penalty parameters
   - **`07_technical_premium_analysis.py`**: Multi-objective premium optimization with Pareto analysis

2. **Insurance Analysis Framework** (`insurance_analysis_refactored/core/`)
   - **`ParametricInsuranceEngine`**: Core product creation and data structures
   - **`SkillScoreEvaluator`**: Comprehensive evaluation (RMSE, MAE, Brier, CRPS, EDI, TSS)
   - **`InsuranceProductManager`**: Product lifecycle management
   - **`TechnicalPremiumCalculator`**: Advanced premium calculation with VaR and Solvency II
   - **`MarketAcceptabilityAnalyzer`**: Market acceptance and product complexity analysis
   - **`MultiObjectiveOptimizer`**: Pareto frontier optimization
   - **Input Adapters**: 
     - `CLIMADAInputAdapter`: Traditional hazard-exposure-impact workflow
     - `BayesianInputAdapter`: Probabilistic simulation results processing
   - **`EnhancedCatInCircleAnalyzer`**: Multi-radius spatial analysis with cKDTree optimization

3. **Bayesian Uncertainty Framework** (`bayesian/`)
   - **`ProbabilisticLossDistributionGenerator`**: Monte Carlo uncertainty quantification
   - **`HierarchicalBayesianModel`**: 4-level hierarchical models with PyMC
   - **`RobustBayesianAnalyzer`**: Mixed Predictive Estimation (MPE)
   - **Report Generation** (`reports/`): Automated diagnostics and analysis
   - **Decision Theory**: Shifts from deterministic to probabilistic basis risk evaluation

4. **Modular Data Pipeline**
   - **`hazard_modeling/`**: TC hazard processing with CLIMADA integration
   - **`exposure_modeling/`**: LitPop methodology and hospital OSM extraction  
   - **`impact_analysis/`**: Impact calculations with Emanuel USA functions
   - **`data_processing/`**: IBTrACS track processing and filtering

5. **Comprehensive Evaluation** (`skill_scores/`)
   - **Traditional Metrics**: RMSE, MAE, correlation for deterministic evaluation
   - **Probabilistic Metrics**: CRPS, EDI, TSS for Bayesian uncertainty distributions
   - **Bootstrap Confidence**: Statistical significance testing for all metrics

### Configuration (`config/settings.py`)
```python
NC_BOUNDS = {'lon_min': -84.5, 'lon_max': -75.5, 'lat_min': 33.8, 'lat_max': 36.6}
YEAR_RANGE = (1980, 2024)
RESOLUTION = 0.1  # degrees
```

## Key Design Patterns

### Steinmann 2023 Academic Standard Compliance
- **Exact Product Generation**: 70 products = 25 single + 20 dual + 15 triple + 10 quadruple threshold functions
- **Payout Structure**: 25% incremental payouts with step functions (no weighted averaging)
- **Multi-Radius Analysis**: 5 analysis radii (15km, 30km, 50km, 75km, 100km)  
- **Total Product Space**: 350 products (5 radii × 70 threshold functions)
- **Pure Cat-in-a-Circle**: No spatial weighting, pure maximum wind speed extraction

### Dual-Track Evaluation Architecture
- **Traditional Track**: Deterministic CLIMADA workflow → RMSE/MAE/correlation evaluation
- **Probabilistic Track**: Bayesian uncertainty quantification → CRPS/EDI/TSS evaluation
- **Hybrid Analysis**: Input adapters allow seamless switching between evaluation modes
- **Comparative Framework**: Built-in method comparison and performance benchmarking

### Performance Engineering
- **Spatial Optimization**: cKDTree indexing reduces Cat-in-a-Circle from hours to minutes
- **Vectorized Operations**: NumPy/Pandas optimization throughout analysis pipeline
- **Memory Management**: Efficient data structures for large Monte Carlo simulations
- **Caching Systems**: Built-in result caching to avoid repeated expensive calculations

### Academic Research Integration
- **Bilingual Documentation**: English/Chinese comments for international collaboration
- **Reproducible Research**: Functional programming style with immediate execution
- **HPC Optimization**: Designed for high-performance computing environments
- **Modular Architecture**: Plug-and-play components for different research questions

## Usage Patterns

### Complete Analysis Workflow
```bash
# Full sequential analysis - run all scripts in order
conda activate climada_env

python 01_run_climada.py                    # ~10 minutes: Generate CLIMADA data
python 02_spatial_analysis.py               # ~30 minutes: Spatial analysis  
python 03_insurance_product.py              # ~5 minutes: Generate 350 products
python 04_traditional_parm_insurance.py     # ~45 minutes: Traditional analysis
python 05_robust_bayesian_parm_insurance.py # ~2 hours: Bayesian analysis
python 06_sensitivity_analysis.py           # ~30 minutes: Sensitivity analysis
python 07_technical_premium_analysis.py     # ~1 hour: Premium optimization
```

### Steinmann 2023 Standard Analysis
```python
# Generate exactly 70 Steinmann 2023-compliant products
from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products

# Create products for all 5 radii (total = 350 products)
all_products = generate_steinmann_2023_products()
print(f"✅ Generated {len(all_products)} products (5 radii × 70 functions)")

# Individual component usage
from insurance_analysis_refactored.core import ParametricInsuranceEngine
engine = ParametricInsuranceEngine()
product = engine.create_parametric_product(
    product_id="NC_H3_50km", 
    index_type="CAT_IN_CIRCLE",
    trigger_thresholds=[33.0, 42.0, 58.0],
    payout_amounts=[1e8, 3e8, 5e8]
)
```

### Input Adapter System
```python
from insurance_analysis_refactored.core.input_adapters import (
    CLIMADAInputAdapter, BayesianInputAdapter, HybridInputAdapter
)

# Traditional deterministic CLIMADA workflow
climada_adapter = CLIMADAInputAdapter(tc_hazard, exposure, impact_func_set)
traditional_results = framework.analyze_with_adapter(climada_adapter)

# Bayesian probabilistic uncertainty workflow  
bayesian_adapter = BayesianInputAdapter(bayesian_simulation_results)
probabilistic_results = framework.analyze_with_adapter(bayesian_adapter)

# Hybrid analysis combining both approaches
hybrid_adapter = HybridInputAdapter(climada_data, bayesian_data, blend_weight=0.7)
hybrid_results = framework.analyze_with_adapter(hybrid_adapter)
```

### Bayesian Uncertainty Quantification
```python
from bayesian.robust_bayesian_uncertainty import ProbabilisticLossDistributionGenerator
from bayesian.hierarchical_bayesian_model import MixedPredictiveEstimation

# Generate probabilistic loss distributions
loss_generator = ProbabilisticLossDistributionGenerator(
    n_monte_carlo_samples=500,
    hazard_uncertainty_std=0.15,
    exposure_uncertainty_log_std=0.20
)

probabilistic_losses = loss_generator.generate_probabilistic_distributions(
    tc_hazard, exposure, impact_functions
)

# Apply Mixed Predictive Estimation for robust analysis
mpe = MixedPredictiveEstimation()
robust_posterior = mpe.fit_ensemble_posterior(probabilistic_losses)
```

## Dependencies & Environment

### Core Requirements (Manual Installation)
- **CLIMADA**: Full catastrophic modeling framework with Emanuel USA impact functions
- **Spatial Analysis**: NumPy, Pandas, SciPy (cKDTree), GeoPandas for geographic processing  
- **Bayesian Stack**: PyMC, ArviZ, statsmodels for hierarchical modeling
- **Visualization**: Matplotlib with Chinese font support (Heiti TC configured)
- **Data Sources**: IBTrACS tropical cyclone database, OSM building extractions
- **Performance**: Optimized for HPC environments with large memory requirements

### Environment Setup
```bash
# REQUIRED: Activate CLIMADA conda environment
conda activate climada_env  # Essential for all analysis scripts

# Verify core dependencies
python -c "import climada; print('CLIMADA available')"
python -c "from insurance_analysis_refactored.core import ParametricInsuranceEngine; print('Framework ready')"
python -c "from bayesian import RobustBayesianAnalyzer; print('Bayesian framework ready')"

# Check configuration
python -c "from config.settings import NC_BOUNDS, YEAR_RANGE; print(f'Analysis: NC {YEAR_RANGE[0]}-{YEAR_RANGE[1]}')"
```

### Data Dependencies
- **North Carolina Boundaries**: `osm/osm_bpf/nc.osm.pbf` (OpenStreetMap extract)
- **Configuration**: All geographic bounds and parameters in `config/settings.py`
- **Results Storage**: Outputs saved to local `results/` directory (not in repo)

## Development Notes

### Code Organization Principles
- **Sequential Scripts**: Numbered workflow (01-07) for reproducible analysis pipeline
- **Jupyter-style Cells**: Scripts use `# %%` cell markers for interactive development
- **Modular Libraries**: `insurance_analysis_refactored/` and `bayesian/` provide reusable components
- **Bilingual Documentation**: English/Chinese throughout for international research collaboration
- **Academic Focus**: Prioritizes research correctness over production deployment
- **No Standard Packaging**: No requirements.txt, setup.py by design (research flexibility)

### Performance Characteristics
- **Main analysis runtime**: Several hours for complete 350-product analysis
- **Memory requirements**: Multi-GB for Monte Carlo simulations and spatial indexing
- **Optimization**: Built for HPC clusters, not local development machines
- **Caching**: Results cached to avoid re-computation of expensive operations

### Research Integration
- **Steinmann 2023 Compliance**: Exact reproduction of academic standard methodology
- **Extension Framework**: Bayesian decision theory layered on top of standard approach  
- **Comparative Analysis**: Built-in framework for comparing traditional vs probabilistic methods
- **Publication Ready**: Code structure supports reproducible research and academic publication