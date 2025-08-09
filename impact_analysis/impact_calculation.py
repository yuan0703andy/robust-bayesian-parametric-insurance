"""
Impact calculation and analysis functions
"""

import numpy as np
from climada.entity import ImpactFuncSet, ImpfTropCyclone
from climada.engine import ImpactCalc


def calculate_tc_impact(hazard, exposure):
    """
    Calculate tropical cyclone impact
    
    Parameters:
    -----------
    hazard : TropCyclone
        CLIMADA tropical cyclone hazard object
    exposure : LitPop
        CLIMADA exposure object
        
    Returns:
    --------
    tuple
        (impact, impf_set) - Impact object and impact function set
    """
    
    print("Preparing impact functions...")
    # Create tropical cyclone impact function
    impf_tc = ImpfTropCyclone.from_emanuel_usa()
    impf_set = ImpactFuncSet([impf_tc])
    impf_set.check()
    
    # Get hazard type and ID
    haz_type = list(impf_set.get_hazard_types())[0]
    haz_id = list(impf_set.get_ids()[haz_type])[0]
    
    print(f"Hazard type: {haz_type}, Impact function ID: {haz_id}")
    
    # Set impact function ID for exposure
    exposure_copy = exposure.copy()
    if f"impf_{haz_type}" not in exposure_copy.gdf.columns:
        exposure_copy.gdf[f"impf_{haz_type}"] = haz_id
    exposure_copy.check()
    
    print("Calculating hazard impact...")
    # Calculate impact
    impact = ImpactCalc(exposure_copy, impf_set, hazard).impact(save_mat=False)
    
    return impact, impf_set


def analyze_impact_results(impact):
    """
    Analyze hazard impact results
    
    Parameters:
    -----------
    impact : Impact
        CLIMADA impact object
        
    Returns:
    --------
    dict
        Dictionary containing impact statistics
    """
    
    # Basic statistics
    stats = {
        'aai_agg': impact.aai_agg,
        'total_loss': impact.at_event.sum(),
        'max_event_loss': impact.at_event.max(),
        'affected_events': (impact.at_event > 0).sum()
    }
    
    # Calculate return period losses
    freq_curve = impact.calc_freq_curve()
    rp_losses = {}
    for rp in [10, 50, 100, 500]:
        idx = np.argmin(np.abs(freq_curve.return_per - rp))
        rp_losses[rp] = freq_curve.impact[idx]
    
    stats['return_period_losses'] = rp_losses
    
    return stats


def print_impact_summary(impact):
    """
    Print hazard impact analysis summary
    
    Parameters:
    -----------
    impact : Impact
        CLIMADA impact object
    """
    
    stats = analyze_impact_results(impact)
    
    print(f"\n💥 Hazard Impact Analysis Results:")
    print(f"   Annual Average Loss (AAI): ${stats['aai_agg']/1e9:.2f}B")
    print(f"   Total event loss: ${stats['total_loss']/1e9:.2f}B")
    print(f"   Maximum single event loss: ${stats['max_event_loss']/1e9:.2f}B")
    print(f"   Number of affected events: {stats['affected_events']}")
    
    print(f"\n📊 Return Period Loss Estimates:")
    for rp, loss in stats['return_period_losses'].items():
        print(f"   {rp}-year return period: ${loss/1e9:.2f}B")
    
    print(f"📈 Hazard impact calculation completed")