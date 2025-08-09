# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

North Carolina Tropical Cyclone Parametric Insurance Analysis system - an academic research codebase combining CLIMADA catastrophic modeling with advanced Bayesian uncertainty quantification for robust parametric insurance design.

## Key Commands

### Running the Main Analysis
```bash
python nc_tc_comprehensive_functional.py
```

### Testing Bayesian Implementation
```bash
python test_bayesian_fix.py
```

### Module Execution
Since there are no test/build configuration files, development is done through:
- Direct script execution
- Module imports as libraries
- Manual dependency management

## High-Level Architecture

### Core Analysis Components

1. **Main Entry Point** (`nc_tc_comprehensive_functional.py`)
   - 29,900+ line comprehensive analysis script
   - Orchestrates complete NC tropical cyclone workflow
   - Functional style with immediate cell execution
   - Bilingual comments (English/Chinese)

2. **Refactored Insurance Analysis** (`insurance_analysis_refactored/core/`)
   - `analysis_framework.py`: High-level API for complete workflows
   - `parametric_engine.py`: Cat-in-a-Circle index extraction and payout calculations
   - `skill_evaluator.py`: Unified evaluation using RMSE, MAE, Brier, CRPS, EDI, TSS scores
   - `product_manager.py`: Enterprise product lifecycle management
   - `saffir_simpson_products.py`: Generates 70 Steinmann-compliant products
   - `input_adapters.py`: CLIMADA and Bayesian data adapters
   - `unified_probabilistic_framework_full.py`: Full probabilistic framework

3. **Bayesian Analysis** (`bayesian/`)
   - Hierarchical 4-level Bayesian models for uncertainty quantification
   - Mixed Predictive Estimation (MPE) for robust analysis
   - Automated report generation and diagnostics
   - Shifts evaluation from deterministic to probabilistic

4. **Data Processing Pipeline**
   - `hazard_modeling/`: TC hazard processing with CLIMADA
   - `exposure_modeling/`: LitPop exposure and hospital OSM extraction
   - `impact_analysis/`: Impact calculations
   - `data_processing/`: Track processing

5. **Evaluation System** (`skill_scores/`)
   - Traditional: RMSE, MAE, correlation for deterministic inputs
   - Probabilistic: CRPS, EDI, TSS for Bayesian distributions

### Configuration (`config/settings.py`)
```python
NC_BOUNDS = {'lon_min': -84.5, 'lon_max': -75.5, 'lat_min': 33.8, 'lat_max': 36.6}
YEAR_RANGE = (1980, 2024)
RESOLUTION = 0.1  # degrees
```

## Key Design Patterns

### Steinmann 2023 Compliance
- Generates exactly 70 insurance products: 25 single + 20 dual + 15 triple + 10 quadruple thresholds
- 25% incremental payout structure
- 5 radii × 70 functions = 350 total products

### Dual-Track Evaluation
- **Traditional**: Deterministic evaluation with RMSE/correlation
- **Probabilistic**: Bayesian uncertainty with CRPS/EDI/TSS scores

### Performance Optimizations
- cKDTree spatial indexing for Cat-in-a-Circle analysis
- Vectorized calculations (hours → minutes runtime)
- Built-in caching mechanisms

## Usage Patterns

### Basic Analysis
```python
from insurance_analysis_refactored.core import UnifiedAnalysisFramework

framework = UnifiedAnalysisFramework()
results = framework.run_comprehensive_analysis(parametric_indices, observed_losses)
```

### Steinmann Standard Analysis
```python
results = framework.run_steinmann_analysis(parametric_indices, observed_losses)
# Validates 70 products per Steinmann et al. (2023)
```

### Input Adapters
```python
from insurance_analysis_refactored.core.input_adapters import CLIMADAInputAdapter, BayesianInputAdapter

# For CLIMADA hazard objects
climada_adapter = CLIMADAInputAdapter(tc_hazard, exposure, impact_func_set)

# For Bayesian simulation results
bayesian_adapter = BayesianInputAdapter(bayesian_results)
```

## Dependencies

Core requirements (manual installation needed):
- CLIMADA catastrophic modeling framework
- NumPy, Pandas for data manipulation
- SciPy for spatial indexing (cKDTree)
- Matplotlib for visualization
- Bayesian analysis libraries
- OpenStreetMap data processing tools

## Important Notes

- No standard Python packaging files (requirements.txt, setup.py)
- Dependencies managed manually - ensure CLIMADA environment is properly configured
- Main entry point: `nc_tc_comprehensive_functional.py` for full NC analysis
- Results stored in `results/` directory
- Academic research code - prioritizes correctness over production deployment
- HPC environment optimized - designed for high-performance computing systems