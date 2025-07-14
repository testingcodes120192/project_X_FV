# amr_factory.py
from typing import Dict, Any, Optional
import importlib
import warnings

from pyparsing import abstractmethod

from .base_amr import BaseAMR

class AMRFactory:
    """
    Factory class for creating AMR instances based on backend selection.
    
    This factory allows for runtime selection of AMR implementations
    and handles optional dependencies gracefully.
    """
    
    # Registry of available AMR backends
    _backends = {}
    _backend_modules = {
        'simple': '.simple_amr_refactored',
        'amrex': '.amrex_amr'
    }
    
    @abstractmethod
    def adapt_to_initial_condition(self):
        """
        Adapt the grid based on the initial condition set in the base solver.
        
        This method should:
        1. Sync the initial condition from the base solver
        2. Compute error indicators based on the actual data
        3. Create refined levels where needed
        4. Interpolate the solution to refined levels
        
        This is called after the initial condition has been set on the
        base solver but before time stepping begins.
        """
        pass
    
    @classmethod
    def register_backend(cls, name: str, amr_class: type):
        """
        Register an AMR backend.
        
        Parameters
        ----------
        name : str
            Name of the backend (e.g., 'simple', 'amrex')
        amr_class : type
            The AMR class that implements BaseAMR
        """
        if not issubclass(amr_class, BaseAMR):
            raise TypeError(f"{amr_class} must inherit from BaseAMR")
        cls._backends[name] = amr_class
    
    @classmethod
    def create_amr(cls, backend: str, base_solver, config: Dict[str, Any]) -> Optional[BaseAMR]:
        """
        Create an AMR instance based on the specified backend.
        
        Parameters
        ----------
        backend : str
            Name of the AMR backend ('simple', 'amrex', or None)
        base_solver : FVHeatSolver
            The base finite volume solver
        config : dict
            AMR configuration parameters
            
        Returns
        -------
        BaseAMR or None
            AMR instance, or None if backend is 'none' or unavailable
        """
        # Handle no AMR case
        if backend is None or backend.lower() == 'none':
            return None
            
        # Normalize backend name
        backend = backend.lower()
        
        # Try to load the backend if not already registered
        if backend not in cls._backends:
            if backend in cls._backend_modules:
                try:
                    # Dynamically import the module
                    module = importlib.import_module(cls._backend_modules[backend], package=__name__)
                    
                    # Auto-register if it has the expected class name
                    if backend == 'simple' and hasattr(module, 'SimpleAMRRefactored'):
                        cls.register_backend('simple', module.SimpleAMRRefactored)
                    elif backend == 'amrex' and hasattr(module, 'AMReXAMR'):
                        cls.register_backend('amrex', module.AMReXAMR)
                        
                except ImportError as e:
                    if backend == 'amrex':
                        warnings.warn(
                            f"AMReX backend not available: {e}\n"
                            "To use AMReX, install pyAMReX: pip install pyamrex\n"
                            "Falling back to simple AMR."
                        )
                        # Try to fall back to simple AMR
                        if 'simple' in cls._backends or cls._try_load_simple():
                            return cls.create_amr('simple', base_solver, config)
                    else:
                        raise ImportError(f"Failed to load AMR backend '{backend}': {e}")
        
        # Check if backend is now available
        if backend not in cls._backends:
            available = ', '.join(cls.get_available_backends())
            raise ValueError(
                f"Unknown AMR backend: '{backend}'. "
                f"Available backends: {available}"
            )
        
        # Create and return the AMR instance
        amr_class = cls._backends[backend]
        amr_instance = amr_class(base_solver, config)
        
        # IMPORTANT: Only initialize the base grid structure
        # The actual adaptation will happen after initial conditions are set
        amr_instance.initialize()
        
        return amr_instance
    
    @classmethod
    def _try_load_simple(cls) -> bool:
        """Try to load the simple AMR backend."""
        try:
            module = importlib.import_module(cls._backend_modules['simple'], package=__name__)
            if hasattr(module, 'SimpleAMRRefactored'):
                cls.register_backend('simple', module.SimpleAMRRefactored)
                return True
        except ImportError:
            pass
        return False
    
    @classmethod
    def get_available_backends(cls) -> list:
        """
        Get list of available AMR backends.
        
        Returns
        -------
        list
            List of backend names that are currently available
        """
        available = ['none']  # Always available
        
        # Check registered backends
        available.extend(cls._backends.keys())
        
        # Check for potentially available backends
        for backend, module_path in cls._backend_modules.items():
            if backend not in cls._backends:
                try:
                    # Try to import without registering
                    module = importlib.import_module(module_path, package=__name__)
                    if ((backend == 'simple' and hasattr(module, 'SimpleAMRRefactored')) or
                        (backend == 'amrex' and hasattr(module, 'AMReXAMR'))):
                        available.append(backend)
                except ImportError:
                    # Backend not available
                    pass
        
        return sorted(list(set(available)))
    
    @classmethod
    def is_backend_available(cls, backend: str) -> bool:
        """
        Check if a specific backend is available.
        
        Parameters
        ----------
        backend : str
            Name of the backend to check
            
        Returns
        -------
        bool
            True if the backend can be loaded
        """
        return backend.lower() in cls.get_available_backends()
    
    @classmethod
    def get_backend_info(cls, backend: str) -> Dict[str, Any]:
        """
        Get information about a specific backend.
        
        Parameters
        ----------
        backend : str
            Name of the backend
            
        Returns
        -------
        dict
            Information about the backend including features and requirements
        """
        info = {
            'none': {
                'name': 'No AMR',
                'description': 'Run without adaptive mesh refinement',
                'features': [],
                'requirements': [],
                'parallel': False,
                'gpu_support': False
            },
            'simple': {
                'name': 'Simple AMR',
                'description': 'Basic block-structured AMR implementation',
                'features': [
                    'Quadtree refinement',
                    'Gradient-based criteria',
                    'Time subcycling',
                    'Python-based'
                ],
                'requirements': ['numpy', 'scipy'],
                'parallel': False,
                'gpu_support': False
            },
            'amrex': {
                'name': 'AMReX',
                'description': 'High-performance AMR framework from LBL',
                'features': [
                    'Full block-structured AMR',
                    'MPI parallelization',
                    'GPU support',
                    'Load balancing',
                    'Flux correction',
                    'Checkpoint/restart',
                    'Built-in I/O'
                ],
                'requirements': ['pyamrex', 'mpi4py (optional)'],
                'parallel': True,
                'gpu_support': True
            }
        }
        
        backend = backend.lower()
        if backend in info:
            # Add availability status
            info[backend]['available'] = cls.is_backend_available(backend)
            return info[backend]
        else:
            return {
                'name': backend,
                'description': 'Unknown backend',
                'available': False
            }
    
        
    @classmethod
    def get_backend_parameters(cls, backend: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific backend.
        
        Parameters
        ----------
        backend : str
            Name of the backend
            
        Returns
        -------
        dict
            Default configuration parameters for the backend
        """
        # Common parameters for all AMR backends
        common_params = {
            'max_levels': 3,
            'refinement_ratio': 2,
            'regrid_interval': 10,
            # Workflow parameters
            'initial_levels': 1,
            'adapt_after_ic': True,
            'show_before_adapt': False,
            'temp_threshold': 500.0,
            'show_error_indicator': False
        }
        
        # Backend-specific parameters
        backend_params = {
            'simple': {
                'refine_threshold': 100.0,
                'coarsen_threshold': 10.0,
                'proper_nesting_buffer': 1
            },
            'amrex': {
                'max_grid_size': 32,
                'blocking_factor': 8,
                'n_error_buf': 2,
                'grid_eff': 0.7,
                'refine_threshold': 100.0,  # Changed from 0.5 to match gradient scale
                'coarsen_threshold': 10.0,
                'n_cell_coarsen': 2,
                'subcycling': True,
                'interpolation_type': 'pc_interp'  # piecewise constant
            }
        }
        
        params = common_params.copy()
        if backend in backend_params:
            params.update(backend_params[backend])
            
        return params


# Helper function for backward compatibility
def create_amr_system(base_solver, enable_amr: bool = False, backend: str = 'simple',
                     **kwargs) -> Optional[BaseAMR]:
    """
    Create an AMR system (backward compatibility helper).
    
    Parameters
    ----------
    base_solver : FVHeatSolver
        Base finite volume solver
    enable_amr : bool
        Whether to enable AMR
    backend : str
        AMR backend to use
    **kwargs
        Additional AMR parameters
        
    Returns
    -------
    BaseAMR or None
        AMR system instance or None if disabled
    """
    if not enable_amr:
        return None
        
    # Get default parameters for the backend
    config = AMRFactory.get_backend_parameters(backend)
    
    # Override with provided parameters
    config.update(kwargs)
    
    return AMRFactory.create_amr(backend, base_solver, config)