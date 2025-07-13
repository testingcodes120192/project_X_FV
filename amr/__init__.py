# amr/__init__.py
"""
Adaptive Mesh Refinement (AMR) module for FV Heat Diffusion Simulator.

This module provides multiple AMR backend implementations:
- Simple: Python-based block-structured AMR
- AMReX: High-performance AMR using the AMReX framework

The module uses a factory pattern to allow runtime selection of backends.
"""

from .base_amr import BaseAMR
from .amr_factory import AMRFactory, create_amr_system

# Try to import available backends
__all__ = ['BaseAMR', 'AMRFactory', 'create_amr_system']

# Attempt to register backends
try:
    from .simple_amr import SimpleAMRRefactored
    AMRFactory.register_backend('simple', SimpleAMRRefactored)
    __all__.append('SimpleAMRRefactored')
except ImportError:
    pass

try:
    from .amrex_amr import AMReXAMR
    AMRFactory.register_backend('amrex', AMReXAMR)
    __all__.append('AMReXAMR')
    
    # Also export AMReX utilities if available
    from .amrex_data_bridge import AMReXDataBridge
    from .amrex_config import AMReXConfig, setup_amrex_environment
    __all__.extend(['AMReXDataBridge', 'AMReXConfig', 'setup_amrex_environment'])
except ImportError:
    pass

# Version info
__version__ = '1.0.0'
__author__ = 'FV Heat Diffusion Team'

def get_available_backends():
    """Get list of available AMR backends."""
    return AMRFactory.get_available_backends()

def backend_info(backend=None):
    """
    Print information about available AMR backends.
    
    Parameters
    ----------
    backend : str, optional
        Specific backend to get info for. If None, shows all.
    """
    if backend is None:
        backends = get_available_backends()
        print("Available AMR backends:")
        print("-" * 40)
        
        for b in backends:
            if b == 'none':
                continue
            info = AMRFactory.get_backend_info(b)
            status = "✓" if info['available'] else "✗"
            print(f"{status} {info['name']}: {info['description']}")
            
    else:
        info = AMRFactory.get_backend_info(backend)
        print(f"\n{info['name']}")
        print("=" * len(info['name']))
        print(f"Description: {info['description']}")
        print(f"Available: {'Yes' if info['available'] else 'No'}")
        
        if 'features' in info and info['features']:
            print("\nFeatures:")
            for feature in info['features']:
                print(f"  • {feature}")
                
        if 'requirements' in info and info['requirements']:
            print("\nRequirements:")
            for req in info['requirements']:
                print(f"  • {req}")
                
        print(f"\nParallel support: {'Yes' if info.get('parallel') else 'No'}")
        print(f"GPU support: {'Yes' if info.get('gpu_support') else 'No'}")

def test_amrex_installation():
    """Test if AMReX is properly installed and functional."""
    try:
        import amrex.space3d as amr
        print("✓ AMReX (3D) is installed and importable")
        version = getattr(amr, '__version__', 'unknown')
        print(f"  Version: {version}")
        
        # Check GPU support
        if hasattr(amr.Config, 'gpu_backend'):
            print(f"  GPU backend: {amr.Config.gpu_backend}")
            
        return True
    except ImportError:
        try:
            import amrex.space2d as amr
            print("✓ AMReX (2D) is installed and importable")
            return True
        except ImportError:
            print("✗ AMReX is not installed")
            print("  To install: pip install pyamrex")
            return False