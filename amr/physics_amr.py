# physics_based_refinement.py
"""
Physics-based refinement criteria for interface-capturing AMR.
This module provides shape-agnostic interface detection based on solution properties.
"""

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
from typing import Dict, Tuple, Optional


class PhysicsBasedRefinement:
    """
    Implements physics-based refinement criteria that automatically
    detect interfaces in the solution field.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize physics-based refinement.
        
        Parameters
        ----------
        config : dict
            Configuration with refinement parameters
        """
        # Threshold parameters
        self.gradient_factor = config.get('gradient_factor', 2.0)  # Std devs above mean
        self.curvature_threshold = config.get('curvature_threshold', 0.5)
        self.jump_threshold = config.get('jump_threshold', 0.3)
        self.coherence_threshold = config.get('coherence_threshold', 0.8)
        
        # Combination weights
        self.use_gradient = config.get('use_gradient', True)
        self.use_curvature = config.get('use_curvature', True)
        self.use_jump = config.get('use_jump', True)
        self.use_coherence = config.get('use_coherence', False)
        
        # Buffer settings
        self.interface_buffer = config.get('interface_buffer', 2)  # cells
        
    def compute_refinement_indicator(self, solver) -> np.ndarray:
        """
        Compute physics-based refinement indicator.
        
        Parameters
        ----------
        solver : FVHeatSolver
            The solver instance with current solution
            
        Returns
        -------
        np.ndarray
            Refinement indicator field (0-1 normalized)
        """
        # Extract interior solution
        T_interior = solver.mesh.extract_interior(solver.T)
        
        # Initialize combined indicator
        indicator = np.zeros_like(T_interior)
        
        # 1. Gradient-based indicator with adaptive threshold
        if self.use_gradient:
            grad_indicator = self._compute_adaptive_gradient_indicator(
                solver.T, solver.mesh
            )
            indicator = np.maximum(indicator, grad_indicator)
            
        # 2. Curvature-based indicator
        if self.use_curvature:
            curv_indicator = self._compute_curvature_indicator(
                solver.T, solver.mesh
            )
            indicator = np.maximum(indicator, curv_indicator)
            
        # 3. Gradient jump indicator
        if self.use_jump:
            jump_indicator = self._compute_gradient_jump_indicator(
                solver.T, solver.mesh, solver.weno
            )
            indicator = np.maximum(indicator, jump_indicator)
            
        # 4. Gradient coherence indicator (optional, more expensive)
        if self.use_coherence:
            coherence_indicator = self._compute_coherence_indicator(
                solver.T, solver.mesh
            )
            indicator = np.maximum(indicator, coherence_indicator)
            
        # Apply interface buffer to ensure adequate refinement width
        if self.interface_buffer > 0:
            indicator = self._apply_interface_buffer(indicator)
            
        return indicator
    
    def _compute_adaptive_gradient_indicator(self, T: np.ndarray, mesh) -> np.ndarray:
        """
        Compute gradient-based indicator with adaptive thresholding.
        
        This identifies regions where gradients are statistical outliers,
        indicating interfaces rather than smooth variations.
        """
        # Compute gradients using WENO for accuracy
        from weno import WENOReconstructor
        weno = WENOReconstructor(order=2)  # Use 2nd order for gradient computation
        
        # Get derivatives
        dTdx = weno.compute_derivatives_x(T, mesh)
        dTdy = weno.compute_derivatives_y(T, mesh)
        
        # Gradient magnitude
        grad_mag = np.sqrt(dTdx**2 + dTdy**2)
        
        # Compute statistics for adaptive thresholding
        grad_mean = np.mean(grad_mag)
        grad_std = np.std(grad_mag)
        
        # Normalize temperature range for scale-independent threshold
        T_interior = mesh.extract_interior(T)
        T_range = np.max(T_interior) - np.min(T_interior)
        
        if T_range > 0:
            # Adaptive threshold: mean + factor * std
            threshold = grad_mean + self.gradient_factor * grad_std
            
            # Normalize by temperature range
            grad_mag_normalized = grad_mag / T_range
            threshold_normalized = threshold / T_range
            
            # Compute indicator (0-1 range)
            indicator = np.clip(grad_mag_normalized / threshold_normalized, 0, 1)
        else:
            indicator = np.zeros_like(grad_mag)
            
        return indicator
    
    def _compute_curvature_indicator(self, T: np.ndarray, mesh) -> np.ndarray:
        """
        Compute curvature-based indicator.
        
        High curvature (|∇²T|) relative to gradient indicates interfaces.
        """
        g = mesh.ghost_cells
        T_interior = T[g:-g, g:-g]
        dx, dy = mesh.dx, mesh.dy
        
        # Compute second derivatives (central differences)
        # d²T/dx²
        Txx = (T[g:-g, g+1:-g+1] - 2*T_interior + T[g:-g, g-1:-g-1]) / dx**2
        
        # d²T/dy²
        Tyy = (T[g+1:-g+1, g:-g] - 2*T_interior + T[g-1:-g-1, g:-g]) / dy**2
        
        # Laplacian magnitude
        laplacian_mag = np.abs(Txx + Tyy)
        
        # Compute gradient magnitude for normalization
        dTdx = (T[g:-g, g+1:-g+1] - T[g:-g, g-1:-g-1]) / (2*dx)
        dTdy = (T[g+1:-g+1, g:-g] - T[g-1:-g-1, g:-g]) / (2*dy)
        grad_mag = np.sqrt(dTdx**2 + dTdy**2) + 1e-10  # Avoid division by zero
        
        # Curvature indicator: high |∇²T|/|∇T| indicates interface
        curvature = laplacian_mag / grad_mag
        
        # Normalize to 0-1 range
        if np.max(curvature) > 0:
            indicator = np.clip(curvature / (self.curvature_threshold * np.max(curvature)), 0, 1)
        else:
            indicator = np.zeros_like(curvature)
            
        return indicator
    
    def _compute_gradient_jump_indicator(self, T: np.ndarray, mesh, weno) -> np.ndarray:
        """
        Compute gradient jump indicator.
        
        This detects cells where gradients change rapidly, indicating
        that the cell contains an interface.
        """
        g = mesh.ghost_cells
        indicator = np.zeros((mesh.ny, mesh.nx))
        
        # For each cell, compute gradient variation
        for j in range(1, mesh.ny-1):
            for i in range(1, mesh.nx-1):
                # Get temperature values in 3x3 stencil
                stencil = T[j+g-1:j+g+2, i+g-1:i+g+2]
                
                # Compute gradients at cell faces
                grad_left = (stencil[1, 1] - stencil[1, 0]) / mesh.dx
                grad_right = (stencil[1, 2] - stencil[1, 1]) / mesh.dx
                grad_bottom = (stencil[1, 1] - stencil[0, 1]) / mesh.dy
                grad_top = (stencil[2, 1] - stencil[1, 1]) / mesh.dy
                
                # Compute jumps
                jump_x = abs(grad_right - grad_left)
                jump_y = abs(grad_top - grad_bottom)
                
                # Normalize by local gradient scale
                local_grad = np.sqrt(
                    ((grad_left + grad_right)/2)**2 + 
                    ((grad_bottom + grad_top)/2)**2
                ) + 1e-10
                
                # Jump indicator
                jump_mag = np.sqrt(jump_x**2 + jump_y**2) / local_grad
                indicator[j, i] = np.clip(jump_mag / self.jump_threshold, 0, 1)
                
        return indicator
    
    def _compute_coherence_indicator(self, T: np.ndarray, mesh) -> np.ndarray:
        """
        Compute gradient direction coherence indicator.
        
        At interfaces, gradients point consistently perpendicular to the interface.
        """
        g = mesh.ghost_cells
        indicator = np.zeros((mesh.ny, mesh.nx))
        
        # Window size for coherence computation
        window = 3
        
        for j in range(window//2, mesh.ny - window//2):
            for i in range(window//2, mesh.nx - window//2):
                # Collect gradient vectors in window
                grad_vectors = []
                
                for dj in range(-window//2, window//2 + 1):
                    for di in range(-window//2, window//2 + 1):
                        jj = j + dj + g
                        ii = i + di + g
                        
                        # Gradient at this point
                        gx = (T[jj, ii+1] - T[jj, ii-1]) / (2*mesh.dx)
                        gy = (T[jj+1, ii] - T[jj-1, ii]) / (2*mesh.dy)
                        
                        grad_mag = np.sqrt(gx**2 + gy**2)
                        if grad_mag > 1e-10:
                            # Normalized gradient direction
                            grad_vectors.append([gx/grad_mag, gy/grad_mag])
                
                if len(grad_vectors) > 1:
                    # Compute coherence as alignment of gradient directions
                    grad_array = np.array(grad_vectors)
                    mean_direction = np.mean(grad_array, axis=0)
                    mean_direction /= np.linalg.norm(mean_direction) + 1e-10
                    
                    # Compute alignment with mean direction
                    alignments = np.dot(grad_array, mean_direction)
                    coherence = np.mean(np.abs(alignments))
                    
                    # Only flag if gradient magnitude is significant
                    center_gx = (T[j+g, i+g+1] - T[j+g, i+g-1]) / (2*mesh.dx)
                    center_gy = (T[j+g+1, i+g] - T[j+g-1, i+g]) / (2*mesh.dy)
                    center_grad_mag = np.sqrt(center_gx**2 + center_gy**2)
                    
                    T_range = np.max(T[g:-g, g:-g]) - np.min(T[g:-g, g:-g])
                    if T_range > 0 and center_grad_mag / T_range > 0.1:
                        indicator[j, i] = coherence
                        
        # Threshold coherence
        indicator = np.where(indicator > self.coherence_threshold, indicator, 0)
        
        return indicator
    
    def _apply_interface_buffer(self, indicator: np.ndarray) -> np.ndarray:
        """
        Apply buffer around detected interfaces to ensure adequate refinement width.
        """
        # Use maximum filter to expand high-indicator regions
        from scipy.ndimage import maximum_filter
        
        # Apply maximum filter with specified buffer size
        size = 2 * self.interface_buffer + 1
        buffered = maximum_filter(indicator, size=size)
        
        # Smooth the edges slightly
        from scipy.ndimage import gaussian_filter
        buffered = gaussian_filter(buffered, sigma=0.5)
        
        return np.clip(buffered, 0, 1)
    

# Integration with AMReX - add this method to AMReXAMR class
def _compute_error_indicator_from_data_physics_based(self, level: int):
    """
    Compute error indicator using physics-based interface detection.
    
    This replaces the gradient-based indicator with sophisticated
    physics-based criteria that work for any interface shape.
    """
    level_data = self.levels[level]
    solver = self.level_solvers[level]
    
    # Create physics-based refinement object
    refine_config = {
        'gradient_factor': self.config.get('gradient_factor', 2.0),
        'curvature_threshold': self.config.get('curvature_threshold', 0.5),
        'jump_threshold': self.config.get('jump_threshold', 0.3),
        'use_gradient': True,
        'use_curvature': True,
        'use_jump': True,
        'use_coherence': self.config.get('use_coherence', False),
        'interface_buffer': self.config.get('interface_buffer', 2)
    }
    
    refiner = PhysicsBasedRefinement(refine_config)
    
    # Compute refinement indicator
    error_indicator = refiner.compute_refinement_indicator(solver)
    
    # Convert to boolean flags
    # Use adaptive threshold based on indicator statistics
    threshold = self.config.get('refine_threshold', 0.3)
    
    # Alternative: use Otsu's method for automatic thresholding
    if self.config.get('auto_threshold', False):
        from skimage.filters import threshold_otsu
        if np.max(error_indicator) > np.min(error_indicator):
            threshold = threshold_otsu(error_indicator)
    
    # Flag cells for refinement
    flags = error_indicator > threshold
    
    # Convert flags to tagged boxes
    tagged_boxes = self._flags_to_boxes(flags)
    
    # Visualize if requested
    if self.config.get('show_error_indicator', False):
        self._visualize_physics_indicator(level, error_indicator, flags)
    
    return error_indicator, tagged_boxes


def _visualize_physics_indicator(self, level: int, indicator: np.ndarray, flags: np.ndarray):
    """Visualize the physics-based error indicator and refinement flags."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get solution for reference
    solver = self.level_solvers[level]
    T_interior = solver.mesh.extract_interior(solver.T)
    
    # Plot temperature
    im1 = ax1.imshow(T_interior, origin='lower', cmap='hot')
    ax1.set_title(f'Temperature (Level {level})')
    plt.colorbar(im1, ax=ax1)
    
    # Plot error indicator
    im2 = ax2.imshow(indicator, origin='lower', cmap='viridis')
    ax2.set_title('Physics-Based Error Indicator')
    plt.colorbar(im2, ax=ax2)
    
    # Plot refinement flags
    im3 = ax3.imshow(flags, origin='lower', cmap='RdBu', vmin=0, vmax=1)
    ax3.set_title('Refinement Flags')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()


# Usage example - configuration for your main.py
def get_physics_based_amr_config():
    """Get configuration for physics-based interface-capturing AMR."""
    return {
        # Basic AMR settings
        'max_levels': 2,
        'refinement_ratio': 2,
        'max_grid_size': 16,
        'blocking_factor': 4,
        'n_error_buf': 1,
        'regrid_interval': 10,
        
        # Physics-based refinement settings
        'refine_mode': 'physics_based',
        'gradient_factor': 2.0,      # Std devs above mean for gradient
        'curvature_threshold': 0.5,  # Relative curvature threshold
        'jump_threshold': 0.3,       # Gradient jump threshold
        'use_coherence': False,      # Enable for better interface detection (slower)
        'interface_buffer': 2,       # Buffer cells around interfaces
        'refine_threshold': 0.3,     # Overall refinement threshold
        'auto_threshold': True,      # Use automatic thresholding
        
        # Visualization
        'show_error_indicator': True
    }