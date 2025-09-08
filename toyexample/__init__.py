"""Toyexample package - Modularized toy example from toy_example_complete.py."""

# Main API functions
from .main import (
    main,
    run_complete_analysis,
    stage1_generate_data,
    stage2_train_test_split,
    stage3_run_model_matrix,
    stage4_analyze_results,
    stage5_stress_test,
    create_results_visualization
)

# Core components
from .core.data import ToyDataGenerator, SimulatedCLIMADAData, SimulatedSpatialData
from .core.model import UnifiedEndToEndVIModel
from .core.trainer import EndToEndTrainer

# Configuration and components
from .components.config import ModelConfiguration
from .components.prior import PriorScenario, LikelihoodFamily, PriorLikelihoodProcessor

# Analysis tools
from .analysis.stress import RobustnessStressTester

__version__ = "1.0.0"

__all__ = [
    # Main functions
    'main',
    'run_complete_analysis',
    'stage1_generate_data',
    'stage2_train_test_split', 
    'stage3_run_model_matrix',
    'stage4_analyze_results',
    'stage5_stress_test',
    'create_results_visualization',
    
    # Core components
    'ToyDataGenerator',
    'SimulatedCLIMADAData',
    'SimulatedSpatialData',
    'UnifiedEndToEndVIModel',
    'EndToEndTrainer',
    
    # Configuration
    'ModelConfiguration',
    'PriorScenario',
    'LikelihoodFamily', 
    'PriorLikelihoodProcessor',
    
    # Analysis
    'RobustnessStressTester'
]