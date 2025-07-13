# amrex_config.py
"""
Configuration handler for AMReX parameters.

This module provides utilities for:
- Converting Python configuration to AMReX ParmParse format
- Validating AMReX parameters
- Managing AMReX runtime options
- Handling MPI and GPU settings
"""

from typing import Dict, Any, List, Optional, Union
import os

try:
    import amrex.space3d as amr
    AMREX_AVAILABLE = True
except ImportError:
    try:
        import amrex.space2d as amr
        AMREX_AVAILABLE = True
    except ImportError:
        AMREX_AVAILABLE = False


class AMReXConfig:
    """Handler for AMReX configuration and parameters."""
    
    # Default AMReX parameters
    DEFAULT_PARAMS = {
        # Grid parameters
        'amr.max_level': 2,
        'amr.ref_ratio': 2,
        'amr.max_grid_size': 32,
        'amr.blocking_factor': 8,
        'amr.grid_eff': 0.7,
        'amr.n_error_buf': 2,
        'amr.n_proper': 1,
        
        # Regridding
        'amr.regrid_int': 10,
        'amr.regrid_on_restart': 1,
        
        # Tagging
        'amr.refinement_indicators': 'gradT',
        'amr.gradT.max_level': 2,
        'amr.gradT.adjacent_difference_greater': 100.0,
        'amr.gradT.field_name': 'temperature',
        
        # I/O
        'amr.plot_int': -1,
        'amr.plot_file': 'plt',
        'amr.checkpoint_int': -1,
        'amr.checkpoint_file': 'chk',
        
        # Verbosity
        'amr.v': 0,
        'amrex.v': 0,
        'amrex.verbose': 0,
        
        # Performance
        'amrex.use_gpu_aware_mpi': 0,
        'tiny_profiler.v': 0,
        'tiny_profiler.device_synchronize_around_region': 0,
        
        # Particles (if needed)
        'particles.do_tiling': 1,
        
        # FabArray
        'fabarray.mfiter_tile_size': '1024000 8 8',
        
        # Memory
        'amrex.the_arena_init_size': 0,
        'amrex.the_device_arena_init_size': 8388608,
    }
    
    @classmethod
    def create_parmparse(cls, config: Dict[str, Any]) -> None:
        """
        Create AMReX ParmParse entries from Python config.
        
        Parameters
        ----------
        config : dict
            Python configuration dictionary
        """
        if not AMREX_AVAILABLE:
            raise ImportError("AMReX not available")
            
        # Merge with defaults
        params = cls.DEFAULT_PARAMS.copy()
        
        # Map Python config to AMReX parameters
        param_mapping = {
            'max_levels': 'amr.max_level',
            'refinement_ratio': 'amr.ref_ratio',
            'max_grid_size': 'amr.max_grid_size',
            'blocking_factor': 'amr.blocking_factor',
            'grid_eff': 'amr.grid_eff',
            'n_error_buf': 'amr.n_error_buf',
            'regrid_interval': 'amr.regrid_int',
            'refine_threshold': 'amr.gradT.adjacent_difference_greater',
            'verbosity': 'amr.v',
        }
        
        # Convert configuration
        for py_key, amrex_key in param_mapping.items():
            if py_key in config:
                value = config[py_key]
                if py_key == 'max_levels':
                    value = value - 1  # AMReX uses max_level (0-based)
                params[amrex_key] = value
                
        # Create ParmParse entries
        for key, value in params.items():
            namespace, param = key.rsplit('.', 1)
            pp = amr.ParmParse(namespace)
            
            if isinstance(value, bool):
                pp.add(param, int(value))
            elif isinstance(value, int):
                pp.add(param, value)
            elif isinstance(value, float):
                pp.add(param, value)
            elif isinstance(value, str):
                pp.add(param, value)
            elif isinstance(value, list):
                if all(isinstance(v, int) for v in value):
                    pp.addarr(param, value)
                elif all(isinstance(v, float) for v in value):
                    pp.addarr(param, value)
                    
    @classmethod
    def setup_mpi(cls, use_mpi: bool = True, nprocs: Optional[int] = None) -> None:
        """
        Setup MPI configuration for AMReX.
        
        Parameters
        ----------
        use_mpi : bool
            Whether to use MPI
        nprocs : int, optional
            Number of MPI processes (for validation)
        """
        if not AMREX_AVAILABLE:
            return
            
        pp = amr.ParmParse("amrex")
        
        if use_mpi:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
                
                if nprocs is not None and size != nprocs:
                    if rank == 0:
                        print(f"Warning: Requested {nprocs} processes but running with {size}")
                        
                # Set MPI-related parameters
                pp.add("use_gpu_aware_mpi", 0)  # Disable by default
                
                if rank == 0:
                    print(f"AMReX MPI initialized with {size} processes")
                    
            except ImportError:
                print("MPI requested but mpi4py not available")
                pp.add("use_gpu_aware_mpi", 0)
        else:
            # Disable MPI features
            pp.add("use_gpu_aware_mpi", 0)
            
    @classmethod
    def setup_gpu(cls, use_gpu: bool = False, gpu_id: Optional[int] = None) -> None:
        """
        Setup GPU configuration for AMReX.
        
        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration
        gpu_id : int, optional
            Specific GPU device ID to use
        """
        if not AMREX_AVAILABLE:
            return
            
        pp = amr.ParmParse("amrex")
        device_pp = amr.ParmParse("device")
        
        if use_gpu:
            # Check if AMReX was built with GPU support
            if hasattr(amr.Config, 'gpu_backend'):
                backend = amr.Config.gpu_backend
                if backend != "DISABLED":
                    pp.add("use_gpu_aware_mpi", 0)  # Usually disable for safety
                    
                    if gpu_id is not None:
                        device_pp.add("device_id", gpu_id)
                        
                    # Set GPU-specific parameters
                    pp.add("the_device_arena_init_size", 8388608)  # 8MB
                    
                    print(f"AMReX GPU support enabled ({backend})")
                    if gpu_id is not None:
                        print(f"Using GPU device {gpu_id}")
                else:
                    print("AMReX was not built with GPU support")
            else:
                print("Cannot determine if AMReX has GPU support")
                
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        Validate AMReX configuration parameters.
        
        Parameters
        ----------
        config : dict
            Configuration to validate
            
        Returns
        -------
        list
            List of validation warnings/errors
        """
        warnings = []
        
        # Check max_grid_size vs blocking_factor
        max_grid = config.get('max_grid_size', 32)
        blocking = config.get('blocking_factor', 8)
        
        if max_grid % blocking != 0:
            warnings.append(
                f"max_grid_size ({max_grid}) should be divisible by "
                f"blocking_factor ({blocking})"
            )
            
        # Check refinement ratio
        ref_ratio = config.get('refinement_ratio', 2)
        if ref_ratio not in [2, 4]:
            warnings.append(
                f"refinement_ratio {ref_ratio} is unusual. "
                "Standard values are 2 or 4"
            )
            
        # Check grid efficiency
        grid_eff = config.get('grid_eff', 0.7)
        if grid_eff < 0.5:
            warnings.append(
                f"grid_eff {grid_eff} is very low. "
                "This may lead to many small grids"
            )
        elif grid_eff > 0.95:
            warnings.append(
                f"grid_eff {grid_eff} is very high. "
                "This may lead to poor load balancing"
            )
            
        # Check regrid interval
        regrid_int = config.get('regrid_interval', 10)
        if regrid_int < 1:
            warnings.append("regrid_interval should be >= 1")
            
        # Check error buffer
        n_error_buf = config.get('n_error_buf', 2)
        if n_error_buf < 1:
            warnings.append(
                "n_error_buf should be >= 1 for proper nesting"
            )
            
        return warnings
        
    @classmethod
    def create_inputs_file(cls, config: Dict[str, Any], filename: str = "inputs.amrex") -> None:
        """
        Create an AMReX inputs file from configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        filename : str
            Output filename
        """
        # Merge with defaults
        params = cls.DEFAULT_PARAMS.copy()
        
        # Map Python config to AMReX parameters
        param_mapping = {
            'max_levels': 'amr.max_level',
            'refinement_ratio': 'amr.ref_ratio',
            'max_grid_size': 'amr.max_grid_size',
            'blocking_factor': 'amr.blocking_factor',
            'grid_eff': 'amr.grid_eff',
            'n_error_buf': 'amr.n_error_buf',
            'regrid_interval': 'amr.regrid_int',
            'refine_threshold': 'amr.gradT.adjacent_difference_greater',
        }
        
        for py_key, amrex_key in param_mapping.items():
            if py_key in config:
                value = config[py_key]
                if py_key == 'max_levels':
                    value = value - 1  # AMReX uses max_level (0-based)
                params[amrex_key] = value
                
        # Write inputs file
        with open(filename, 'w') as f:
            f.write("# AMReX inputs file generated by FV Heat Diffusion Simulator\n\n")
            
            # Group parameters by namespace
            namespaces = {}
            for key, value in sorted(params.items()):
                namespace = key.split('.')[0]
                if namespace not in namespaces:
                    namespaces[namespace] = []
                namespaces[namespace].append((key, value))
                
            # Write grouped parameters
            for namespace, params_list in namespaces.items():
                f.write(f"# {namespace} parameters\n")
                for key, value in params_list:
                    if isinstance(value, bool):
                        f.write(f"{key} = {int(value)}\n")
                    elif isinstance(value, (int, float)):
                        f.write(f"{key} = {value}\n")
                    elif isinstance(value, str):
                        f.write(f"{key} = {value}\n")
                    elif isinstance(value, list):
                        f.write(f"{key} = {' '.join(map(str, value))}\n")
                f.write("\n")
                
        print(f"AMReX inputs file written to {filename}")
        
    @classmethod
    def get_runtime_info(cls) -> Dict[str, Any]:
        """
        Get AMReX runtime information.
        
        Returns
        -------
        dict
            Runtime information including version, GPU support, etc.
        """
        if not AMREX_AVAILABLE:
            return {"available": False}
            
        info = {
            "available": True,
            "version": getattr(amr, '__version__', 'unknown'),
        }
        
        # Check GPU support
        if hasattr(amr.Config, 'gpu_backend'):
            info['gpu_backend'] = amr.Config.gpu_backend
            info['gpu_enabled'] = amr.Config.gpu_backend != "DISABLED"
        else:
            info['gpu_backend'] = 'unknown'
            info['gpu_enabled'] = False
            
        # Check MPI support
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            info['mpi_enabled'] = True
            info['mpi_size'] = comm.Get_size()
            info['mpi_rank'] = comm.Get_rank()
        except ImportError:
            info['mpi_enabled'] = False
            info['mpi_size'] = 1
            info['mpi_rank'] = 0
            
        # Memory info
        if hasattr(amr, 'Arena'):
            info['arena_info'] = {
                'initialized': True,
                # Additional arena info could go here
            }
            
        return info
        
    @classmethod
    def print_config_summary(cls, config: Dict[str, Any]) -> None:
        """
        Print a summary of AMReX configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        print("\nAMReX Configuration Summary")
        print("=" * 40)
        
        # Basic parameters
        print(f"Max levels: {config.get('max_levels', 3)}")
        print(f"Refinement ratio: {config.get('refinement_ratio', 2)}")
        print(f"Max grid size: {config.get('max_grid_size', 32)}")
        print(f"Blocking factor: {config.get('blocking_factor', 8)}")
        print(f"Grid efficiency: {config.get('grid_eff', 0.7)}")
        
        # Regridding
        print(f"\nRegridding:")
        print(f"  Interval: {config.get('regrid_interval', 10)} steps")
        print(f"  Error buffer: {config.get('n_error_buf', 2)} cells")
        print(f"  Refinement threshold: {config.get('refine_threshold', 100.0)}")
        
        # Runtime info
        runtime_info = cls.get_runtime_info()
        print(f"\nRuntime:")
        print(f"  AMReX available: {runtime_info.get('available', False)}")
        if runtime_info.get('available'):
            print(f"  Version: {runtime_info.get('version', 'unknown')}")
            print(f"  GPU support: {runtime_info.get('gpu_enabled', False)}")
            if runtime_info.get('gpu_enabled'):
                print(f"  GPU backend: {runtime_info.get('gpu_backend', 'unknown')}")
            print(f"  MPI enabled: {runtime_info.get('mpi_enabled', False)}")
            if runtime_info.get('mpi_enabled'):
                print(f"  MPI processes: {runtime_info.get('mpi_size', 1)}")
                
        # Validation warnings
        warnings = cls.validate_config(config)
        if warnings:
            print(f"\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")
                
        print("=" * 40)


def setup_amrex_environment(config: Dict[str, Any], 
                          use_mpi: bool = False,
                          use_gpu: bool = False,
                          verbose: bool = False) -> None:
    """
    Setup AMReX environment with given configuration.
    
    Parameters
    ----------
    config : dict
        AMReX configuration
    use_mpi : bool
        Enable MPI support
    use_gpu : bool
        Enable GPU support
    verbose : bool
        Enable verbose output
    """
    if not AMREX_AVAILABLE:
        raise ImportError("AMReX not available")
        
    # Initialize AMReX if not already done
    if not amr.initialized():
        # Prepare initialization arguments
        init_args = []
        if verbose:
            init_args.extend(['amrex.v=1', 'amr.v=1'])
            
        amr.initialize(init_args)
        
    # Setup configuration
    AMReXConfig.create_parmparse(config)
    
    # Setup MPI if requested
    if use_mpi:
        AMReXConfig.setup_mpi(use_mpi)
        
    # Setup GPU if requested
    if use_gpu:
        AMReXConfig.setup_gpu(use_gpu)
        
    # Print summary if verbose
    if verbose:
        AMReXConfig.print_config_summary(config)