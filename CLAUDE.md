# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

North Carolina Tropical Cyclone Parametric Insurance Analysis system - an academic research codebase combining CLIMADA catastrophic modeling with advanced Bayesian uncertainty quantification for robust parametric insurance design. This implements the Steinmann et al. (2023) framework for parametric insurance with robust Bayesian decision theory extensions.

## Key Commands

### Running the Main Analysis
```bash
# Complete NC tropical cyclone analysis (29,900+ line functional script)
python nc_tc_comprehensive_functional.py

# Alternative optimized test runs
python test_new_bayesian_framework.py
```

### Testing and Validation
```bash
# Test Bayesian framework implementation
python test_bayesian_fix.py

# Import and test core modules
python -c "from insurance_analysis_refactored.core import UnifiedAnalysisFramework; print('Framework loaded successfully')"
```

### Module Development
Since there are no standard build files (requirements.txt, setup.py), development follows:
- Direct script execution for analysis workflows
- Module imports as libraries for reusable components
- Manual dependency management (requires CLIMADA environment)
- Interactive development through Jupyter-style cell execution

## High-Level Architecture

### Core Analysis Components

1. **Main Entry Point** (`nc_tc_comprehensive_functional.py`)
   - 29,900+ line comprehensive analysis script implementing complete workflow
   - Orchestrates NC tropical cyclone catastrophic risk assessment
   - Functional style with immediate cell execution and bilingual documentation
   - Generates 350 products (5 radii × 70 Steinmann functions)
   - Dual-track analysis: Traditional RMSE vs Bayesian CRPS evaluation

2. **Unified Insurance Analysis Framework** (`insurance_analysis_refactored/core/`)
   - **`UnifiedAnalysisFramework`**: High-level API integrating all components
   - **`ParametricInsuranceEngine`**: Cat-in-a-Circle index extraction with cKDTree optimization
   - **`SkillScoreEvaluator`**: Comprehensive evaluation (RMSE, MAE, Brier, CRPS, EDI, TSS)
   - **`InsuranceProductManager`**: Enterprise product lifecycle management
   - **`SaffirSimpsonProductGenerator`**: Generates exactly 70 Steinmann 2023-compliant products
   - **Input Adapters**: 
     - `CLIMADAInputAdapter`: Traditional hazard-exposure-impact workflow
     - `BayesianInputAdapter`: Probabilistic simulation results processing
   - **`EnhancedCatInCircleAnalyzer`**: Multi-radius spatial analysis with Haversine distances

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
```python
# Main comprehensive analysis - preferred entry point
python nc_tc_comprehensive_functional.py

# For programmatic access:
from insurance_analysis_refactored.core import UnifiedAnalysisFramework

framework = UnifiedAnalysisFramework()
results = framework.run_comprehensive_analysis(parametric_indices, observed_losses)
```

### Steinmann 2023 Standard Analysis
```python
# Academic research compliance - exactly 70 products
steinmann_results = framework.run_steinmann_analysis(parametric_indices, observed_losses)

# Validation check
assert len(steinmann_results.products) == 70
print("✅ Fully compliant with Steinmann et al. (2023) standard")

# Generate all 350 products (5 radii × 70 functions)
from insurance_analysis_refactored.core.saffir_simpson_products import generate_steinmann_2023_products
all_products = generate_steinmann_2023_products()
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
# Assumes CLIMADA conda environment already exists
conda activate climada_env  # or your CLIMADA environment name

# Verify key dependencies
python -c "import climada; print('CLIMADA available')"
python -c "from insurance_analysis_refactored.core import UnifiedAnalysisFramework; print('Framework ready')"
```

### Data Dependencies
- **North Carolina Boundaries**: `osm/osm_bpf/nc.osm.pbf` (OpenStreetMap extract)
- **Configuration**: All geographic bounds and parameters in `config/settings.py`
- **Results Storage**: Outputs saved to local `results/` directory (not in repo)

## Development Notes

### Code Organization Principles
- **No standard packaging**: No requirements.txt, setup.py by design (research flexibility)
- **Functional main script**: `nc_tc_comprehensive_functional.py` uses Jupyter-style cells for reproducibility
- **Modular libraries**: `insurance_analysis_refactored/` provides reusable components
- **Bilingual documentation**: English/Chinese throughout for international research collaboration
- **Academic focus**: Prioritizes research correctness over production deployment

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