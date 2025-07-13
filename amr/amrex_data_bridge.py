# amrex_data_bridge.py - Fixed version with complete implementations
"""
Data bridge for converting between NumPy arrays and AMReX data structures.

This module provides utilities for:
- Converting NumPy arrays to AMReX MultiFabs
- Extracting AMReX MultiFab data to NumPy arrays
- Interpolating between different grid resolutions
- Handling ghost cells appropriately
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any

try:
    import amrex.space3d as amr
except ImportError:
    try:
        import amrex.space2d as amr
    except ImportError:
        amr = None


class AMReXDataBridge:
    """Utilities for data conversion between NumPy and AMReX."""
    
    @staticmethod
    def numpy_to_multifab(array: np.ndarray, mf: 'amr.MultiFab', 
                         component: int = 0, include_ghost: bool = True) -> None:
        """
        Copy data from NumPy array to AMReX MultiFab.
        
        Parameters
        ----------
        array : np.ndarray
            Source numpy array (shape: [ny, nx] or [ny, nx, ncomp])
        mf : amr.MultiFab
            Target MultiFab
        component : int
            Component index in MultiFab
        include_ghost : bool
            Whether array includes ghost cells
        """
        if amr is None:
            raise ImportError("AMReX not available")
            
        # Handle different array shapes
        if array.ndim == 2:
            ny, nx = array.shape
            ncomp = 1
        elif array.ndim == 3:
            ny, nx, ncomp = array.shape
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
            
        # Iterate over boxes in MultiFab
        for mfi in mf:
            # Get box (valid region)
            bx = mfi.validbox()
            
            # Get array view with ghost cells
            if include_ghost:
                arr_mf = mf.array(mfi)
            else:
                # Get only valid region
                arr_mf = mf[mfi].array()
                
            # Get bounds
            lo = bx.small_end
            hi = bx.big_end
            
            # Copy data
            for j in range(lo[1], hi[1] + 1):
                for i in range(lo[0], hi[0] + 1):
                    # Map indices appropriately
                    if include_ghost:
                        # Account for ghost cells in source array
                        j_src = j + mf.n_grow_vect[1]
                        i_src = i + mf.n_grow_vect[0]
                    else:
                        j_src = j
                        i_src = i
                        
                    # Bounds check
                    if 0 <= i_src < nx and 0 <= j_src < ny:
                        if ncomp == 1:
                            arr_mf[i, j, 0, component] = array[j_src, i_src]
                        else:
                            for c in range(min(ncomp, mf.n_comp - component)):
                                arr_mf[i, j, 0, component + c] = array[j_src, i_src, c]
                                
    @staticmethod
    def multifab_to_numpy(mf: 'amr.MultiFab', component: int = 0,
                         include_ghost: bool = False,
                         domain_only: bool = True) -> np.ndarray:
        """
        Extract data from AMReX MultiFab to NumPy array.
        
        Parameters
        ----------
        mf : amr.MultiFab
            Source MultiFab
        component : int
            Component to extract
        include_ghost : bool
            Include ghost cells in output
        domain_only : bool
            Only extract data within the problem domain
            
        Returns
        -------
        np.ndarray
            Extracted data array
        """
        if amr is None:
            raise ImportError("AMReX not available")
            
        # Get domain box
        domain = mf.box_array.minimal_box()
        
        if include_ghost:
            # Expand domain to include ghost cells
            grow_vect = mf.n_grow_vect
            lo = [domain.small_end[i] - grow_vect[i] for i in range(2)]
            hi = [domain.big_end[i] + grow_vect[i] for i in range(2)]
        else:
            lo = [domain.small_end[i] for i in range(2)]
            hi = [domain.big_end[i] for i in range(2)]
            
        # Create output array
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        array = np.zeros((ny, nx))
        
        # Extract data from all boxes
        for mfi in mf:
            if include_ghost:
                bx = mfi.growntilebox()
            else:
                bx = mfi.validbox()
                
            arr_mf = mf.array(mfi)
            
            # Get intersection with domain
            bx_lo = bx.small_end
            bx_hi = bx.big_end
            
            # Copy data
            for j in range(max(bx_lo[1], lo[1]), min(bx_hi[1], hi[1]) + 1):
                for i in range(max(bx_lo[0], lo[0]), min(bx_hi[0], hi[0]) + 1):
                    j_dst = j - lo[1]
                    i_dst = i - lo[0]
                    
                    if 0 <= i_dst < nx and 0 <= j_dst < ny:
                        array[j_dst, i_dst] = arr_mf[i, j, 0, component]
                        
        return array
        
    @staticmethod
    def interpolate_to_uniform_grid(levels: List['AMReXLevel'],
                                   nx_out: int, ny_out: int,
                                   component: int = 0,
                                   domain_lo: Optional[List[float]] = None,
                                   domain_hi: Optional[List[float]] = None) -> np.ndarray:
        """
        Interpolate AMR hierarchy to uniform grid.
        
        Parameters
        ----------
        levels : List[AMReXLevel]
            List of AMR levels
        nx_out : int
            Output grid resolution in x
        ny_out : int
            Output grid resolution in y
        component : int
            Component to extract
        domain_lo : List[float], optional
            Lower bounds of physical domain
        domain_hi : List[float], optional
            Upper bounds of physical domain
            
        Returns
        -------
        np.ndarray
            Interpolated data on uniform grid
        """
        # Get domain bounds from first level if not provided
        if domain_lo is None or domain_hi is None:
            geom0 = levels[0].geom
            if hasattr(geom0, 'ProbLo'):
                domain_lo = [geom0.ProbLo()[0], geom0.ProbLo()[1]]
                domain_hi = [geom0.ProbHi()[0], geom0.ProbHi()[1]]
            else:
                # Fallback
                domain_lo = [0.0, 0.0]
                domain_hi = [1.0, 1.0]
        
        # Create output array
        output = np.zeros((ny_out, nx_out))
        
        # Output grid spacing
        dx_out = (domain_hi[0] - domain_lo[0]) / nx_out
        dy_out = (domain_hi[1] - domain_lo[1]) / ny_out
        
        # Process from coarse to fine (fine overwrites coarse)
        for level_idx, level in enumerate(levels):
            # Get MultiFab
            if hasattr(level, 'temperature'):
                mf = level.temperature
            else:
                continue
                
            # Get level geometry
            geom = level.geom
            
            # Process each box in this level
            for mfi in mf:
                bx = mfi.validbox()
                arr = mf.array(mfi)
                
                lo = bx.small_end
                hi = bx.big_end
                
                # Get cell size for this level
                if hasattr(geom, 'CellSize'):
                    dx_level = geom.data().CellSize()[0]
                    dy_level = geom.data().CellSize()[1]
                else:
                    # Calculate from domain
                    domain = geom.domain
                    dx_level = (domain_hi[0] - domain_lo[0]) / domain.size()[0]
                    dy_level = (domain_hi[1] - domain_lo[1]) / domain.size()[1]
                
                # Map each level cell to output grid
                for j in range(lo[1], hi[1] + 1):
                    for i in range(lo[0], hi[0] + 1):
                        # Physical coordinates of cell center
                        x_center = domain_lo[0] + (i + 0.5) * dx_level
                        y_center = domain_lo[1] + (j + 0.5) * dy_level
                        
                        # Find output cells that overlap this AMR cell
                        i_out_min = int((x_center - 0.5*dx_level - domain_lo[0]) / dx_out)
                        i_out_max = int((x_center + 0.5*dx_level - domain_lo[0]) / dx_out)
                        j_out_min = int((y_center - 0.5*dy_level - domain_lo[1]) / dy_out)
                        j_out_max = int((y_center + 0.5*dy_level - domain_lo[1]) / dy_out)
                        
                        # Clip to output bounds
                        i_out_min = max(0, i_out_min)
                        i_out_max = min(nx_out - 1, i_out_max)
                        j_out_min = max(0, j_out_min)
                        j_out_max = min(ny_out - 1, j_out_max)
                        
                        # Fill output cells
                        value = arr[i, j, 0, component]
                        for j_out in range(j_out_min, j_out_max + 1):
                            for i_out in range(i_out_min, i_out_max + 1):
                                output[j_out, i_out] = value
                        
        return output
        
    @staticmethod
    def create_multifab_from_function(geom: 'amr.Geometry',
                                     grids: 'amr.BoxArray',
                                     dmap: 'amr.DistributionMapping',
                                     ncomp: int, nghost: int,
                                     func: callable) -> 'amr.MultiFab':
        """
        Create a MultiFab initialized with a function.
        
        Parameters
        ----------
        geom : amr.Geometry
            Geometry object
        grids : amr.BoxArray
            Box array
        dmap : amr.DistributionMapping
            Distribution mapping
        ncomp : int
            Number of components
        nghost : int
            Number of ghost cells
        func : callable
            Function f(x, y) -> value
            
        Returns
        -------
        amr.MultiFab
            Initialized MultiFab
        """
        if amr is None:
            raise ImportError("AMReX not available")
            
        # Create MultiFab
        mf = amr.MultiFab(grids, dmap, ncomp, nghost)
        
        # Get cell size and problem bounds
        if hasattr(geom, 'CellSize'):
            dx = geom.data().CellSize()
        else:
            domain = geom.domain
            prob_lo = geom.prob_lo() if hasattr(geom, 'prob_lo') else [0.0, 0.0]
            prob_hi = geom.prob_hi() if hasattr(geom, 'prob_hi') else [1.0, 1.0]
            dx = [(prob_hi[0] - prob_lo[0]) / domain.size()[0],
                  (prob_hi[1] - prob_lo[1]) / domain.size()[1]]

        prob_lo = geom.prob_lo() if hasattr(geom, 'prob_lo') else [0.0, 0.0]

        # Initialize with function
        for mfi in mf:
            bx = mfi.tilebox()
            arr = mf.array(mfi)
            
            lo = bx.small_end
            hi = bx.big_end
            
            for j in range(lo[1], hi[1] + 1):
                for i in range(lo[0], hi[0] + 1):
                    # Physical coordinates (cell center)
                    x = prob_lo[0] + (i + 0.5) * dx[0]
                    y = prob_lo[1] + (j + 0.5) * dx[1]
                    
                    # Evaluate function
                    for comp in range(ncomp):
                        arr[i, j, 0, comp] = func(x, y) if comp == 0 else 0.0
                        
        return mf
        
    @staticmethod
    def compare_with_numpy(mf: 'amr.MultiFab', np_array: np.ndarray,
                          component: int = 0) -> Dict[str, float]:
        """
        Compare MultiFab data with NumPy array.
        
        Parameters
        ----------
        mf : amr.MultiFab
            AMReX MultiFab
        np_array : np.ndarray
            NumPy array to compare with
        component : int
            Component to compare
            
        Returns
        -------
        dict
            Comparison statistics (max_diff, mean_diff, etc.)
        """
        # Extract MultiFab to numpy
        mf_array = AMReXDataBridge.multifab_to_numpy(mf, component=component)
        
        # Ensure same shape
        if mf_array.shape != np_array.shape:
            # Resize or interpolate as needed
            min_shape = (min(mf_array.shape[0], np_array.shape[0]),
                        min(mf_array.shape[1], np_array.shape[1]))
            mf_array = mf_array[:min_shape[0], :min_shape[1]]
            np_array = np_array[:min_shape[0], :min_shape[1]]
            
        # Compute differences
        diff = np.abs(mf_array - np_array)
        
        return {
            'max_diff': np.max(diff),
            'mean_diff': np.mean(diff),
            'rms_diff': np.sqrt(np.mean(diff**2)),
            'max_relative_diff': np.max(diff / (np.abs(np_array) + 1e-10))
        }
        
    @staticmethod
    def copy_with_averaging(src_mf: 'amr.MultiFab', dst_mf: 'amr.MultiFab',
                           src_comp: int = 0, dst_comp: int = 0, 
                           num_comp: int = 1, refinement_ratio: int = 2) -> None:
        """
        Copy data between MultiFabs with different resolutions using averaging.
        
        This implements conservative averaging from fine to coarse grids.
        
        Parameters
        ----------
        src_mf : amr.MultiFab
            Source MultiFab (fine)
        dst_mf : amr.MultiFab  
            Destination MultiFab (coarse)
        src_comp : int
            Starting component in source
        dst_comp : int
            Starting component in destination
        num_comp : int
            Number of components to copy
        refinement_ratio : int
            Refinement ratio between grids
        """
        if amr is None:
            raise ImportError("AMReX not available")
            
        # Use AMReX's built-in average_down if available
        if hasattr(amr, 'average_down'):
            amr.average_down(src_mf, dst_mf, src_comp, num_comp, refinement_ratio)
        else:
            # Manual averaging implementation
            # Process each destination (coarse) box
            for mfi in dst_mf:
                dst_bx = mfi.validbox()
                dst_arr = dst_mf.array(mfi)
                
                dst_lo = dst_bx.small_end
                dst_hi = dst_bx.big_end
                
                # For each coarse cell, average the corresponding fine cells
                for j in range(dst_lo[1], dst_hi[1] + 1):
                    for i in range(dst_lo[0], dst_hi[0] + 1):
                        # Initialize sum for averaging
                        for comp in range(num_comp):
                            sum_val = 0.0
                            count = 0
                            
                            # Sum over fine cells that cover this coarse cell
                            for jj in range(refinement_ratio):
                                for ii in range(refinement_ratio):
                                    # Fine cell indices
                                    i_fine = i * refinement_ratio + ii
                                    j_fine = j * refinement_ratio + jj
                                    
                                    # Find which box in src_mf contains this fine cell
                                    for src_mfi in src_mf:
                                        src_bx = src_mfi.validbox()
                                        src_lo = src_bx.small_end
                                        src_hi = src_bx.big_end
                                        
                                        # Check if fine cell is in this box
                                        if (src_lo[0] <= i_fine <= src_hi[0] and
                                            src_lo[1] <= j_fine <= src_hi[1]):
                                            src_arr = src_mf.array(src_mfi)
                                            sum_val += src_arr[i_fine, j_fine, 0, src_comp + comp]
                                            count += 1
                                            break
                            
                            # Average and store in coarse cell
                            if count > 0:
                                dst_arr[i, j, 0, dst_comp + comp] = sum_val / count
                            else:
                                # No fine data found - keep existing value
                                pass
                                
    @staticmethod
    def inject_fine_to_coarse(src_mf: 'amr.MultiFab', dst_mf: 'amr.MultiFab',
                             src_comp: int = 0, dst_comp: int = 0,
                             num_comp: int = 1, refinement_ratio: int = 2) -> None:
        """
        Inject fine data to coarse grid (no averaging, just direct copy).
        
        This is useful for error estimation where you want the exact fine value
        at coarse cell centers.
        
        Parameters
        ----------
        src_mf : amr.MultiFab
            Source MultiFab (fine)
        dst_mf : amr.MultiFab
            Destination MultiFab (coarse)
        src_comp : int
            Starting component in source
        dst_comp : int
            Starting component in destination
        num_comp : int
            Number of components to copy
        refinement_ratio : int
            Refinement ratio between grids
        """
        if amr is None:
            raise ImportError("AMReX not available")
            
        # Process each destination (coarse) box
        for mfi in dst_mf:
            dst_bx = mfi.validbox()
            dst_arr = dst_mf.array(mfi)
            
            dst_lo = dst_bx.small_end
            dst_hi = dst_bx.big_end
            
            # For each coarse cell, inject from center fine cell
            for j in range(dst_lo[1], dst_hi[1] + 1):
                for i in range(dst_lo[0], dst_hi[0] + 1):
                    # Fine cell at center of coarse cell
                    i_fine = i * refinement_ratio + refinement_ratio // 2
                    j_fine = j * refinement_ratio + refinement_ratio // 2
                    
                    # Find which box in src_mf contains this fine cell
                    for src_mfi in src_mf:
                        src_bx = src_mfi.validbox()
                        src_lo = src_bx.small_end
                        src_hi = src_bx.big_end
                        
                        # Check if fine cell is in this box
                        if (src_lo[0] <= i_fine <= src_hi[0] and
                            src_lo[1] <= j_fine <= src_hi[1]):
                            src_arr = src_mf.array(src_mfi)
                            
                            # Copy components
                            for comp in range(num_comp):
                                dst_arr[i, j, 0, dst_comp + comp] = \
                                    src_arr[i_fine, j_fine, 0, src_comp + comp]
                            break