# simple_amr_refactored.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from .base_amr import BaseAMR

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


class SimpleAMRRefactored(BaseAMR):
    """
    Simple block-structured Adaptive Mesh Refinement for FV solver.
    
    Refactored version that conforms to the BaseAMR interface.
    This implementation uses a quadtree-like structure where cells
    can be refined into 2x2 or 4x4 sub-cells.
    """
    
    def __init__(self, base_solver, config: Dict[str, Any]):
        """Initialize Simple AMR system."""
        super().__init__(base_solver, config)
        
        # Extract Simple AMR specific parameters
        self.refine_threshold = config.get('refine_threshold', 100.0)
        self.coarsen_threshold = config.get('coarsen_threshold', 10.0)
        self.proper_nesting_buffer = config.get('proper_nesting_buffer', 1)
        
        # Store patches by level
        self.patches = {level: [] for level in range(self.max_levels)}
        
        # Refinement flags for each level
        self.refinement_flags = {}
        
        # Time synchronization
        self.time_ratios = [self.refinement_ratio**level for level in range(self.max_levels)]
        
        # Workflow control
        self.initial_levels = config.get('initial_levels', 1)
        self.adapt_after_ic = config.get('adapt_after_ic', True)
        self.base_initialized = False
        self.adapted = False
        
    def initialize(self):
        """Initialize only the base grid structure."""
        # Initialize refinement flags for base level
        self.refinement_flags = {
            level: np.zeros((self.base_solver.mesh.ny // (self.refinement_ratio**level),
                           self.base_solver.mesh.nx // (self.refinement_ratio**level)), 
                          dtype=bool)
            for level in range(self.max_levels)
        }
        
        # Initially no refined patches
        self.patches = {level: [] for level in range(self.max_levels)}
        
        # Mark as initialized but not adapted
        self.base_initialized = True
        self.adapted = False
        
        print(f"Simple AMR initialized with base level only ({self.base_solver.mesh.nx}×{self.base_solver.mesh.ny} cells)")
        print("Ready for initial condition setup")
            
    def initial_refinement(self):
        """Perform initial refinement based on initial conditions."""
        # Flag cells based on initial solution
        for level in range(self.max_levels - 1):
            flags = self.flag_cells_for_refinement(level)
            if np.any(flags):
                self.regrid(level)
                
    def flag_cells_for_refinement(self, level: int) -> np.ndarray:
        """Flag cells that need refinement at a given level."""
        if level >= self.max_levels - 1:
            return np.zeros((1, 1), dtype=bool)
            
        # Compute error indicator
        if level == 0:
            # Base level - use solver's field
            error = self.compute_error_indicator(self.base_solver.T, self.base_solver.mesh)
        else:
            # Higher levels - need to compute from patches
            # Simplified for now - just return no refinement
            return np.zeros(self.refinement_flags[level].shape, dtype=bool)
            
        # Apply threshold
        flags = error > self.refine_threshold
        
        # Store flags
        self.refinement_flags[level] = flags
        
        # Enforce proper nesting
        if self.proper_nesting_buffer > 0:
            self._enforce_proper_nesting(level)
            
        return self.refinement_flags[level]
    
    def adapt_to_initial_condition(self):
        """Adapt the grid based on the current solution in the base solver."""
        if not self.base_initialized:
            raise RuntimeError("Base grid not initialized. Call initialize() first.")
            
        if self.adapted:
            print("Grid already adapted")
            return
            
        print("\n" + "="*60)
        print("Adapting Simple AMR grid to initial condition")
        print("="*60)
        
        # Print initial statistics
        T_interior = self.base_solver.mesh.extract_interior(self.base_solver.T)
        print(f"\nBase level temperature statistics:")
        print(f"  Min: {np.min(T_interior):.1f} K")
        print(f"  Max: {np.max(T_interior):.1f} K")
        print(f"  Mean: {np.mean(T_interior):.1f} K")
        
        # Perform initial refinement based on actual data
        levels_created = 0
        
        for level in range(self.max_levels - 1):
            print(f"\nChecking level {level} for refinement...")
            
            # Flag cells based on actual temperature data
            flags = self.flag_cells_for_refinement(level)
            
            if np.any(flags):
                flagged_count = np.sum(flags)
                print(f"  Flagged {flagged_count} cells for refinement")
                
                # Create patches
                self.regrid(level)
                levels_created += 1
                
                # Print patch statistics
                patch_count = len(self.patches[level + 1])
                print(f"  Created {patch_count} patches at level {level + 1}")
            else:
                print(f"  No refinement needed at level {level}")
                break
                
        self.adapted = True
        print(f"\nGrid adaptation complete!")
        print(f"  Created {levels_created} refined levels")
        
        # Print summary
        self._print_adaptation_summary()
    
    def _print_adaptation_summary(self):
        """Print summary of the adapted grid."""
        print("\n" + "-"*60)
        print("Simple AMR Grid Adaptation Summary")
        print("-"*60)
        
        total_cells = 0
        for level in range(self.max_levels):
            level_cells = self.get_level_cell_count(level)
            if level_cells > 0:
                total_cells += level_cells
                print(f"Level {level}: {level_cells:,} cells")
                if level > 0:
                    print(f"  Patches: {len(self.patches[level])}")
        
        # Compute efficiency
        base_cells = self.base_solver.mesh.nx * self.base_solver.mesh.ny
        uniform_fine_cells = base_cells * (self.refinement_ratio ** (2 * (self.max_levels - 1)))
        efficiency = (uniform_fine_cells - total_cells) / uniform_fine_cells * 100 if uniform_fine_cells > 0 else 0
        
        print(f"\nTotal cells: {total_cells:,}")
        print(f"Equivalent uniform fine grid: {uniform_fine_cells:,} cells")
        print(f"Memory savings: {efficiency:.1f}%")
        print("-"*60)
    
    def compute_error_indicator(self, field, mesh):
        """
        Compute error indicator for refinement criterion.
        
        Uses gradient-based indicator with scaling.
        """
        # Compute gradient magnitude
        grad_mag = mesh.compute_gradient_magnitude(field)
        
        # Also check temperature magnitude
        T_interior = mesh.extract_interior(field)
        
        # Only compute error where temperature is significant
        temp_threshold = self.config.get('temp_threshold', 500.0)
        significant = T_interior > temp_threshold
        
        # Set error to 0 where temperature is low
        error_indicator = np.where(significant, grad_mag, 0.0)
        
        return error_indicator
    
    def _enforce_proper_nesting(self, level: int):
        """Ensure proper nesting of refinement levels."""
        flags = self.refinement_flags[level]
        ny, nx = flags.shape
        
        # Expand refinement by buffer cells in each direction
        expanded = np.zeros_like(flags)
        
        for j in range(ny):
            for i in range(nx):
                if flags[j, i]:
                    # Flag neighbors within buffer distance
                    for dj in range(-self.proper_nesting_buffer, self.proper_nesting_buffer + 1):
                        for di in range(-self.proper_nesting_buffer, self.proper_nesting_buffer + 1):
                            jj = j + dj
                            ii = i + di
                            if 0 <= jj < ny and 0 <= ii < nx:
                                expanded[jj, ii] = True
                                
        self.refinement_flags[level] = expanded
        
    def regrid(self, level: int):
        """Regrid the hierarchy starting from the given level."""
        # Clear existing patches at finer levels
        for lev in range(level + 1, self.max_levels):
            self.patches[lev] = []
            
        # Create new patches based on flags
        if level < self.max_levels - 1:
            self._create_patches_from_flags(level)
            
    def _create_patches_from_flags(self, level: int):
        """Create refined patches based on refinement flags."""
        flags = self.refinement_flags[level]
        
        # Clear patches at next level
        self.patches[level + 1] = []
        
        # Create a patch for each flagged cell (simplified approach)
        for j in range(flags.shape[0]):
            for i in range(flags.shape[1]):
                if flags[j, i]:
                    patch = self._create_patch(level, j, i)
                    self.patches[level + 1].append(patch)
                    
                    # Initialize patch data by interpolation
                    self._interpolate_to_patch(level, j, i, patch)
                    
    def _create_patch(self, parent_level: int, parent_j: int, parent_i: int) -> AMRPatch:
        """Create a refined patch for a parent cell."""
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
            # For higher levels, would need to track patch meshes
            # Simplified for now
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
        
    def _interpolate_to_patch(self, parent_level: int, parent_j: int, parent_i: int, patch: AMRPatch):
        """Interpolate from parent level to patch."""
        if parent_level == 0:
            # Interpolate from base solver
            # Simple approach: copy parent value to all children
            g = self.base_solver.mesh.ghost_cells
            parent_value = self.base_solver.T[parent_j + g, parent_i + g]
            
            # Fill patch interior (accounting for patch ghost cells)
            patch_g = 3
            patch.data[patch_g:-patch_g, patch_g:-patch_g] = parent_value
        else:
            # Would need to handle patch-to-patch interpolation
            pass
            
    def advance_hierarchy(self, dt: float):
        """Advance the entire AMR hierarchy by one coarse time step."""
        # Advance each level with appropriate time step
        for level in range(self.max_levels):
            dt_level = dt / self.time_ratios[level]
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
                        
        # Synchronize levels after advance
        self.synchronize_levels()
        
    def _advance_patch(self, patch: AMRPatch, dt: float, level: int):
        """Advance a single patch (simplified)."""
        # This is a simplified version for demonstration
        # In practice, would need proper finite volume update
        
        g = 3  # ghost cells
        T = patch.data[g:-g, g:-g]
        
        # Simple explicit diffusion
        dx = (patch.x_max - patch.x_min) / patch.nx
        dy = (patch.y_max - patch.y_min) / patch.ny
        
        alpha = self.base_solver.alpha
        
        # Compute Laplacian
        laplacian = (
            (patch.data[g:-g, g+1:-g+1] - 2*T + patch.data[g:-g, g-1:-g-1]) / dx**2 +
            (patch.data[g+1:-g+1, g:-g] - 2*T + patch.data[g-1:-g-1, g:-g]) / dy**2
        )
        
        # Update
        patch.data[g:-g, g:-g] += dt * alpha * laplacian
        
    def synchronize_levels(self):
        """Synchronize data between AMR levels."""
        # Average down from fine to coarse
        for level in range(self.max_levels - 1, 0, -1):
            for patch in self.patches[level]:
                self._restrict_patch_to_parent(patch, level)
                
    def _restrict_patch_to_parent(self, patch: AMRPatch, level: int):
        """Restrict (average) patch data to parent level."""
        parent_j, parent_i = patch.parent_indices
        
        if level == 1:
            # Restrict to base level
            g = 3  # patch ghost cells
            patch_interior = patch.data[g:-g, g:-g]
            avg_value = np.mean(patch_interior)
            
            # Update parent cell
            solver_g = self.base_solver.mesh.ghost_cells
            self.base_solver.T[parent_j + solver_g, parent_i + solver_g] = avg_value
        else:
            # Would need patch-to-patch restriction
            pass
            
    def get_composite_solution(self, field_name: str = 'T') -> Dict[str, np.ndarray]:
        """Get composite solution across all AMR levels."""
        # For simplicity, just return base level solution
        # Full implementation would overlay finer levels
        
        mesh = self.base_solver.mesh
        T_interior = mesh.extract_interior(self.base_solver.T)
        
        # Create result on base grid (can be refined later)
        result = {
            'data': T_interior.copy(),
            'x': mesh.x_centers,
            'y': mesh.y_centers,
            'level_map': np.zeros_like(T_interior, dtype=int)
        }
        
        # Overlay patches (simplified - just marks where patches exist)
        for level in range(1, self.max_levels):
            for patch in self.patches[level]:
                parent_j, parent_i = patch.parent_indices
                if parent_j < mesh.ny and parent_i < mesh.nx:
                    result['level_map'][parent_j, parent_i] = level
                    
        return result
        
    def get_level_data(self, level: int, field_name: str = 'T') -> Dict[str, Any]:
        """Get data from a specific AMR level."""
        if level == 0:
            # Base level
            return {
                'patches': [(0, 0, self.base_solver.mesh.ny, self.base_solver.mesh.nx)],
                'data': self.base_solver.T,
                'mesh': self.base_solver.mesh
            }
        else:
            # Higher levels
            return {
                'patches': self.patches[level],
                'count': len(self.patches[level])
            }
            
    def get_level_cell_count(self, level: int) -> int:
        """Get the number of cells at a specific level."""
        if level == 0:
            return self.base_solver.mesh.nx * self.base_solver.mesh.ny
        else:
            return len(self.patches[level]) * (self.refinement_ratio ** 2)
            
    def compute_refinement_indicators(self, level: int) -> np.ndarray:
        """Compute error indicators for refinement criteria."""
        if level == 0:
            return self.compute_error_indicator(self.base_solver.T, self.base_solver.mesh)
        else:
            # Would need to handle patches
            return np.zeros((1, 1))
            
    def plot_grid_structure(self, ax=None, show_levels=True):
        """Visualize the AMR grid structure."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
        # Define colors for each level
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        # Plot base grid
        mesh = self.base_solver.mesh
        
        # Draw base level cells where not refined
        for j in range(mesh.ny):
            for i in range(mesh.nx):
                if not (level < len(self.refinement_flags) and 
                       self.refinement_flags[0][j, i] if 0 in self.refinement_flags else False):
                    x = i * mesh.dx
                    y = j * mesh.dy
                    rect = mpatches.Rectangle((x, y), mesh.dx, mesh.dy,
                                            fill=False, edgecolor=colors[0],
                                            linewidth=0.5)
                    ax.add_patch(rect)
                    
        # Draw refined patches
        for level in range(1, self.max_levels):
            for patch in self.patches[level]:
                # Draw patch boundary
                rect = mpatches.Rectangle((patch.x_min, patch.y_min),
                                        patch.x_max - patch.x_min,
                                        patch.y_max - patch.y_min,
                                        fill=False, edgecolor=colors[level % len(colors)],
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
                                                fill=False, edgecolor=colors[level % len(colors)],
                                                linewidth=0.5, alpha=0.5)
                        ax.add_patch(rect)
                        
        # Set limits and labels
        ax.set_xlim(0, mesh.plate_length)
        ax.set_ylim(0, mesh.plate_width)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('AMR Grid Structure')
        
        # Legend
        if show_levels:
            legend_elements = [mpatches.Patch(edgecolor=colors[i % len(colors)], fill=False,
                                            label=f'Level {i}')
                              for i in range(min(self.max_levels, len(colors)))]
            ax.legend(handles=legend_elements)
            
        return ax
        
    def save_checkpoint(self, filename: str):
        """Save AMR state to checkpoint file."""
        import pickle
        
        checkpoint = {
            'patches': self.patches,
            'refinement_flags': self.refinement_flags,
            'current_time': self.current_time,
            'time_steps_since_regrid': self.time_steps_since_regrid,
            'config': self.config
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
            
    def load_checkpoint(self, filename: str):
        """Load AMR state from checkpoint file."""
        import pickle
        
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
            
        self.patches = checkpoint['patches']
        self.refinement_flags = checkpoint['refinement_flags']
        self.current_time = checkpoint['current_time']
        self.time_steps_since_regrid = checkpoint['time_steps_since_regrid']
        # Note: config is not updated to preserve current settings