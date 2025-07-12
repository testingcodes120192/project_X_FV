# simple_amr.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class AMRPatch:
    """
    Represents a refined patch in the AMR hierarchy.
    """
    level: int                    # Refinement level (0 = base)
    parent_indices: Tuple[int, int]  # (j, i) indices in parent level
    refinement_ratio: int         # Typically 2 or 4
    nx: int                      # Number of cells in x
    ny: int                      # Number of cells in y
    x_min: float                 # Physical bounds
    x_max: float
    y_min: float
    y_max: float
    data: np.ndarray            # Temperature field
    reaction_data: Optional[np.ndarray] = None  # Reaction progress


class SimpleAMR:
    """
    Simple block-structured Adaptive Mesh Refinement for FV solver.
    
    This implementation uses a quadtree-like structure where cells
    can be refined into 2x2 or 4x4 sub-cells.
    """
    
    def __init__(self, base_solver, max_levels=3, refinement_ratio=2,
                 refine_threshold=100.0, coarsen_threshold=10.0):
        """
        Initialize AMR system.
        
        Parameters
        ----------
        base_solver : FVHeatSolver
            Base level solver
        max_levels : int
            Maximum refinement levels (including base)
        refinement_ratio : int
            Refinement ratio (2 or 4)
        refine_threshold : float
            Threshold for refinement criterion
        coarsen_threshold : float
            Threshold for coarsening criterion
        """
        self.base_solver = base_solver
        self.max_levels = max_levels
        self.refinement_ratio = refinement_ratio
        self.refine_threshold = refine_threshold
        self.coarsen_threshold = coarsen_threshold
        
        # Store patches by level
        self.patches = {level: [] for level in range(max_levels)}
        
        # Refinement flags for each level
        self.refinement_flags = {
            level: np.zeros((base_solver.mesh.ny // (refinement_ratio**level),
                           base_solver.mesh.nx // (refinement_ratio**level)), 
                          dtype=bool)
            for level in range(max_levels)
        }
        
        # Time synchronization
        self.time_ratios = [refinement_ratio**level for level in range(max_levels)]
        
    def compute_error_indicator(self, field, mesh):
        """
        Compute error indicator for refinement criterion.
        
        Uses gradient-based indicator with scaling.
        
        Parameters
        ----------
        field : ndarray
            Temperature field
        mesh : FVMesh
            Mesh object
            
        Returns
        -------
        ndarray
            Error indicator at each cell
        """
        # Compute gradient magnitude
        grad_mag = mesh.compute_gradient_magnitude(field)
        
        # Scale by cell size (smaller cells need less refinement)
        cell_size = np.sqrt(mesh.dx * mesh.dy)
        scaled_error = grad_mag * cell_size
        
        # Also consider the temperature magnitude
        T_interior = mesh.extract_interior(field)
        T_scale = np.maximum(T_interior - 300.0, 1.0)  # Above background
        
        # Combined indicator
        error_indicator = scaled_error * np.sqrt(T_scale / 1000.0)
        
        return error_indicator
    
    def flag_cells_for_refinement(self):
        """
        Flag cells that need refinement based on error indicator.
        
        Updates self.refinement_flags for all levels.
        """
        # Start with base level
        error = self.compute_error_indicator(self.base_solver.T, self.base_solver.mesh)
        
        # Reset flags
        for level in range(self.max_levels):
            self.refinement_flags[level].fill(False)
            
        # Flag base level cells
        self.refinement_flags[0] = error > self.refine_threshold
        
        # Ensure proper nesting (refined cells must have refined neighbors)
        self._enforce_proper_nesting()
        
        # Count flagged cells
        for level in range(self.max_levels):
            n_flagged = np.sum(self.refinement_flags[level])
            if n_flagged > 0:
                print(f"Level {level}: {n_flagged} cells flagged for refinement")
                
    def _enforce_proper_nesting(self):
        """
        Ensure proper nesting of refinement levels.
        
        If a cell is refined, its neighbors must be refined to at least
        one level coarser.
        """
        for level in range(self.max_levels - 1):
            flags = self.refinement_flags[level]
            ny, nx = flags.shape
            
            # Expand refinement by one cell in each direction
            expanded = np.zeros_like(flags)
            for j in range(ny):
                for i in range(nx):
                    if flags[j, i]:
                        # Flag neighbors
                        for dj in [-1, 0, 1]:
                            for di in [-1, 0, 1]:
                                jj = j + dj
                                ii = i + di
                                if 0 <= jj < ny and 0 <= ii < nx:
                                    expanded[jj, ii] = True
                                    
            self.refinement_flags[level] = expanded
            
    def create_refined_patches(self):
        """
        Create refined patches based on refinement flags.
        
        This creates new patch objects but doesn't transfer data yet.
        """
        # Clear existing patches (except base level)
        for level in range(1, self.max_levels):
            self.patches[level] = []
            
        # Create patches for each flagged region
        for level in range(self.max_levels - 1):
            flags = self.refinement_flags[level]
            
            # Find connected regions (simple approach - treat each cell separately)
            for j in range(flags.shape[0]):
                for i in range(flags.shape[1]):
                    if flags[j, i]:
                        # Create a patch covering this cell
                        patch = self._create_patch(level, j, i)
                        self.patches[level + 1].append(patch)
                        
    def _create_patch(self, parent_level, parent_j, parent_i):
        """
        Create a refined patch for a parent cell.
        
        Parameters
        ----------
        parent_level : int
            Level of parent cell
        parent_j : int
            J-index of parent cell
        parent_i : int
            I-index of parent cell
            
        Returns
        -------
        AMRPatch
            New patch object
        """
        # Get parent mesh info
        if parent_level == 0:
            parent_mesh = self.base_solver.mesh
            dx_parent = parent_mesh.dx
            dy_parent = parent_mesh.dy
            
            # Physical bounds of parent cell
            x_min = parent_i * dx_parent
            x_max = (parent_i + 1) * dx_parent
            y_min = parent_j * dy_parent
            y_max = (parent_j + 1) * dy_parent
        else:
            # Would need to track patch meshes for higher levels
            raise NotImplementedError("Multi-level refinement not fully implemented")
            
        # Create refined patch
        patch = AMRPatch(
            level=parent_level + 1,
            parent_indices=(parent_j, parent_i),
            refinement_ratio=self.refinement_ratio,
            nx=self.refinement_ratio,
            ny=self.refinement_ratio,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            data=np.zeros((self.refinement_ratio + 6, self.refinement_ratio + 6))  # Include ghost cells
        )
        
        if self.base_solver.enable_reactions:
            patch.reaction_data = np.zeros_like(patch.data)
            
        return patch
        
    def interpolate_to_fine(self, coarse_data, coarse_j, coarse_i, patch):
        """
        Interpolate from coarse cell to fine patch.
        
        Parameters
        ----------
        coarse_data : ndarray
            Coarse level data
        coarse_j : int
            J-index in coarse level
        coarse_i : int
            I-index in coarse level
        patch : AMRPatch
            Fine level patch to fill
        """
        # Simple approach: use bilinear interpolation
        # Get 3x3 stencil from coarse level
        stencil = coarse_data[coarse_j-1:coarse_j+2, coarse_i-1:coarse_i+2]
        
        # For now, just copy the center value
        # TODO: Implement proper interpolation
        g = 3  # ghost cells
        patch.data[g:-g, g:-g] = stencil[1, 1]
        
    def restrict_to_coarse(self, patch, coarse_data, coarse_j, coarse_i):
        """
        Restrict (average) from fine patch to coarse cell.
        
        Parameters
        ----------
        patch : AMRPatch
            Fine level patch
        coarse_data : ndarray
            Coarse level data to update
        coarse_j : int
            J-index in coarse level
        coarse_i : int
            I-index in coarse level
        """
        # Average fine cells
        g = 3  # ghost cells
        fine_interior = patch.data[g:-g, g:-g]
        coarse_data[coarse_j, coarse_i] = np.mean(fine_interior)
        
    def advance_amr(self, dt_coarse):
        """
        Advance the AMR hierarchy by one coarse time step.
        
        Uses sub-cycling for finer levels.
        
        Parameters
        ----------
        dt_coarse : float
            Time step for coarsest level
        """
        # Advance each level with appropriate time step
        for level in range(self.max_levels):
            dt_level = dt_coarse / self.time_ratios[level]
            n_substeps = self.time_ratios[level]
            
            if level == 0:
                # Advance base solver
                for _ in range(n_substeps):
                    self.base_solver.advance(dt_level)
            else:
                # Advance patches at this level
                for patch in self.patches[level]:
                    for _ in range(n_substeps):
                        self._advance_patch(patch, dt_level, level)
                        
        # Synchronize levels (average down, interpolate up)
        self._synchronize_levels()
        
    def _advance_patch(self, patch, dt, level):
        """
        Advance a single patch.
        
        Parameters
        ----------
        patch : AMRPatch
            Patch to advance
        dt : float
            Time step
        level : int
            Refinement level
        """
        # This is a simplified version
        # In practice, would need to:
        # 1. Fill ghost cells from coarser level
        # 2. Apply boundary conditions
        # 3. Compute fluxes
        # 4. Update solution
        
        # For now, just apply simple diffusion
        g = 3  # ghost cells
        T = patch.data[g:-g, g:-g]
        
        # Simple explicit diffusion (not recommended for production)
        dx = (patch.x_max - patch.x_min) / patch.nx
        dy = (patch.y_max - patch.y_min) / patch.ny
        
        alpha = self.base_solver.alpha
        
        # Compute Laplacian (simple 2nd order)
        laplacian = (
            (patch.data[g:-g, g+1:-g+1] - 2*T + patch.data[g:-g, g-1:-g-1]) / dx**2 +
            (patch.data[g+1:-g+1, g:-g] - 2*T + patch.data[g-1:-g-1, g:-g]) / dy**2
        )
        
        # Update
        patch.data[g:-g, g:-g] += dt * alpha * laplacian
        
    def _synchronize_levels(self):
        """
        Synchronize solution between levels.
        
        Average down from fine to coarse, interpolate ghost cells.
        """
        # Average down from fine to coarse
        for level in range(self.max_levels - 1, 0, -1):
            for patch in self.patches[level]:
                parent_j, parent_i = patch.parent_indices
                
                if level == 1:
                    # Restrict to base level
                    self.restrict_to_coarse(patch, self.base_solver.T, 
                                          parent_j + self.base_solver.mesh.ghost_cells,
                                          parent_i + self.base_solver.mesh.ghost_cells)
                else:
                    # Would need to handle patch-to-patch restriction
                    pass
                    
    def get_composite_solution(self, nx_plot, ny_plot):
        """
        Get composite solution on uniform grid for plotting.
        
        Parameters
        ----------
        nx_plot : int
            Number of plotting points in x
        ny_plot : int
            Number of plotting points in y
            
        Returns
        -------
        x_plot : ndarray
            X coordinates
        y_plot : ndarray
            Y coordinates
        T_plot : ndarray
            Temperature on plotting grid
        level_plot : ndarray
            Refinement level at each point
        """
        # Create plotting grid
        x_plot = np.linspace(0, self.base_solver.mesh.plate_length, nx_plot)
        y_plot = np.linspace(0, self.base_solver.mesh.plate_width, ny_plot)
        X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
        
        T_plot = np.zeros((ny_plot, nx_plot))
        level_plot = np.zeros((ny_plot, nx_plot), dtype=int)
        
        # Start with base level
        from scipy.interpolate import RegularGridInterpolator
        
        T_interior = self.base_solver.mesh.extract_interior(self.base_solver.T)
        interp_base = RegularGridInterpolator(
            (self.base_solver.mesh.y_centers, self.base_solver.mesh.x_centers),
            T_interior,
            method='linear',
            bounds_error=False,
            fill_value=300.0
        )
        
        # Interpolate base level
        points = np.column_stack([Y_plot.ravel(), X_plot.ravel()])
        T_plot = interp_base(points).reshape(ny_plot, nx_plot)
        
        # Overlay finer levels
        for level in range(1, self.max_levels):
            for patch in self.patches[level]:
                # Find plotting points within this patch
                mask_x = (X_plot >= patch.x_min) & (X_plot <= patch.x_max)
                mask_y = (Y_plot >= patch.y_min) & (Y_plot <= patch.y_max)
                mask = mask_x & mask_y
                
                if np.any(mask):
                    # Interpolate patch data to these points
                    # (simplified - just use patch center value)
                    g = 3
                    patch_value = np.mean(patch.data[g:-g, g:-g])
                    T_plot[mask] = patch_value
                    level_plot[mask] = level
                    
        return x_plot, y_plot, T_plot, level_plot
    
    def estimate_effective_resolution(self):
        """
        Estimate the effective resolution of the AMR grid.
        
        Returns
        -------
        dict
            Resolution statistics
        """
        base_cells = self.base_solver.mesh.nx * self.base_solver.mesh.ny
        total_cells = base_cells
        
        cell_counts = {0: base_cells}
        
        for level in range(1, self.max_levels):
            level_cells = len(self.patches[level]) * (self.refinement_ratio**2)
            cell_counts[level] = level_cells
            total_cells += level_cells
            
        # Equivalent uniform resolution
        equiv_resolution = int(np.sqrt(total_cells))
        
        return {
            'base_cells': base_cells,
            'total_cells': total_cells,
            'cell_counts': cell_counts,
            'equivalent_resolution': equiv_resolution,
            'compression_ratio': base_cells / total_cells if total_cells > 0 else 1.0
        }


def plot_amr_grid(amr_system, ax=None):
    """
    Visualize the AMR grid structure.
    
    Parameters
    ----------
    amr_system : SimpleAMR
        AMR system to visualize
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
    # Define colors for each level
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Plot base grid
    mesh = amr_system.base_solver.mesh
    
    # Draw base level cells where not refined
    for j in range(mesh.ny):
        for i in range(mesh.nx):
            if not amr_system.refinement_flags[0][j, i]:
                x = i * mesh.dx
                y = j * mesh.dy
                rect = mpatches.Rectangle((x, y), mesh.dx, mesh.dy,
                                        fill=False, edgecolor=colors[0],
                                        linewidth=0.5)
                ax.add_patch(rect)
                
    # Draw refined patches
    for level in range(1, amr_system.max_levels):
        for patch in amr_system.patches[level]:
            # Draw patch boundary
            rect = mpatches.Rectangle((patch.x_min, patch.y_min),
                                    patch.x_max - patch.x_min,
                                    patch.y_max - patch.y_min,
                                    fill=False, edgecolor=colors[level],
                                    linewidth=2)
            ax.add_patch(rect)
            
            # Draw fine cells within patch
            dx_fine = (patch.x_max - patch.x_min) / patch.nx
            dy_fine = (patch.y_max - patch.y_min) / patch.ny
            
            for j in range(patch.ny):
                for i in range(patch.nx):
                    x = patch.x_min + i * dx_fine
                    y = patch.y_min + j * dy_fine
                    rect = mpatches.Rectangle((x, y), dx_fine, dy_fine,
                                            fill=False, edgecolor=colors[level],
                                            linewidth=0.5, alpha=0.5)
                    ax.add_patch(rect)
                    
    # Labels and formatting
    ax.set_xlim(0, mesh.plate_length)
    ax.set_ylim(0, mesh.plate_width)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('AMR Grid Structure')
    
    # Legend
    legend_elements = [mpatches.Patch(edgecolor=colors[i], fill=False,
                                    label=f'Level {i}')
                      for i in range(min(amr_system.max_levels, len(colors)))]
    ax.legend(handles=legend_elements)
    
    return ax