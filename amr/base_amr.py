# base_amr.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class BaseAMR(ABC):
    """
    Abstract base class for Adaptive Mesh Refinement implementations.
    
    This class defines the interface that all AMR implementations must follow,
    allowing for different backends (Simple, AMReX, etc.) to be used interchangeably.
    """
    
    def __init__(self, base_solver, config: Dict[str, Any]):
        """
        Initialize the AMR system.
        
        Parameters
        ----------
        base_solver : FVHeatSolver
            Base level finite volume solver
        config : dict
            Configuration parameters for the AMR system
        """
        self.base_solver = base_solver
        self.config = config
        
        # Common AMR parameters
        self.max_levels = config.get('max_levels', 3)
        self.refinement_ratio = config.get('refinement_ratio', 2)
        self.regrid_interval = config.get('regrid_interval', 10)
        
        # Time tracking
        self.current_time = 0.0
        self.time_steps_since_regrid = 0
        
        # Statistics
        self.stats = {
            'total_cells': 0,
            'cells_per_level': {},
            'efficiency': 1.0
        }
        
    @abstractmethod
    def initialize(self):
        """
        Initialize the AMR hierarchy.
        
        This method should set up the initial grid structure and
        prepare all necessary data structures.
        """
        pass
    
    @abstractmethod
    def flag_cells_for_refinement(self, level: int) -> np.ndarray:
        """
        Flag cells that need refinement at a given level.
        
        Parameters
        ----------
        level : int
            The level to check for refinement (0 = base level)
            
        Returns
        -------
        np.ndarray
            Boolean array indicating which cells need refinement
        """
        pass
    
    @abstractmethod
    def regrid(self, level: int):
        """
        Regrid the hierarchy starting from the given level.
        
        This method should:
        1. Create new grids based on flagged cells
        2. Transfer data from old to new grids
        3. Ensure proper nesting
        
        Parameters
        ----------
        level : int
            Starting level for regridding
        """
        pass
    
    @abstractmethod
    def advance_hierarchy(self, dt: float):
        """
        Advance the entire AMR hierarchy by one coarse time step.
        
        This method should handle subcycling for finer levels
        and maintain synchronization between levels.
        
        Parameters
        ----------
        dt : float
            Time step for the coarsest level
        """
        pass
    
    @abstractmethod
    def get_composite_solution(self, field_name: str = 'T') -> Dict[str, np.ndarray]:
        """
        Get the composite solution across all AMR levels.
        
        Parameters
        ----------
        field_name : str
            Name of the field to retrieve ('T' for temperature)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'data': Composite solution on finest available grid
            - 'x': X coordinates
            - 'y': Y coordinates  
            - 'level_map': Which AMR level each point comes from
        """
        pass
    
    @abstractmethod
    def get_level_data(self, level: int, field_name: str = 'T') -> Dict[str, Any]:
        """
        Get data from a specific AMR level.
        
        Parameters
        ----------
        level : int
            AMR level (0 = coarsest)
        field_name : str
            Name of the field to retrieve
            
        Returns
        -------
        dict
            Level-specific data including grid structure and field values
        """
        pass
    
    @abstractmethod
    def synchronize_levels(self):
        """
        Synchronize data between AMR levels.
        
        This includes:
        - Restriction (fine to coarse averaging)
        - Flux correction at coarse-fine interfaces
        """
        pass
    
    @abstractmethod
    def compute_refinement_indicators(self, level: int) -> np.ndarray:
        """
        Compute error indicators for refinement criteria.
        
        Parameters
        ----------
        level : int
            Level to compute indicators for
            
        Returns
        -------
        np.ndarray
            Error indicator values for each cell
        """
        pass
    
    # Common methods that can be shared across implementations
    
    def should_regrid(self) -> bool:
        """
        Determine if regridding is needed.
        
        Returns
        -------
        bool
            True if regridding should be performed
        """
        return self.time_steps_since_regrid >= self.regrid_interval
    
    def update_statistics(self):
        """Update AMR statistics."""
        total_cells = 0
        base_cells = self.base_solver.mesh.nx * self.base_solver.mesh.ny
        
        for level in range(self.max_levels):
            level_cells = self.get_level_cell_count(level)
            self.stats['cells_per_level'][level] = level_cells
            total_cells += level_cells
            
        self.stats['total_cells'] = total_cells
        
        # Efficiency: ratio of actual cells to uniform fine grid
        equivalent_fine_cells = base_cells * (self.refinement_ratio ** (2 * (self.max_levels - 1)))
        self.stats['efficiency'] = total_cells / equivalent_fine_cells if equivalent_fine_cells > 0 else 1.0
    
    @abstractmethod
    def get_level_cell_count(self, level: int) -> int:
        """
        Get the number of cells at a specific level.
        
        Parameters
        ----------
        level : int
            AMR level
            
        Returns
        -------
        int
            Number of cells at this level
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get AMR statistics.
        
        Returns
        -------
        dict
            Dictionary of statistics including cell counts and efficiency
        """
        self.update_statistics()
        return self.stats.copy()
    
    @abstractmethod
    def plot_grid_structure(self, ax=None, show_levels=True):
        """
        Visualize the AMR grid structure.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        show_levels : bool
            Color-code by refinement level
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object
        """
        pass
    
    def advance_with_regrid(self, dt: float):
        """
        Advance the hierarchy with automatic regridding.
        
        Parameters
        ----------
        dt : float
            Time step for coarsest level
        """
        # Check if regridding is needed
        if self.should_regrid():
            # Flag cells and regrid all levels
            for level in range(self.max_levels - 1):
                flags = self.flag_cells_for_refinement(level)
                if np.any(flags):
                    self.regrid(level)
            
            self.time_steps_since_regrid = 0
        
        # Advance the hierarchy
        self.advance_hierarchy(dt)
        
        # Update counters
        self.current_time += dt
        self.time_steps_since_regrid += 1
    
    @abstractmethod
    def save_checkpoint(self, filename: str):
        """
        Save AMR state to checkpoint file.
        
        Parameters
        ----------
        filename : str
            Path to checkpoint file
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, filename: str):
        """
        Load AMR state from checkpoint file.
        
        Parameters
        ----------
        filename : str
            Path to checkpoint file
        """
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage of the AMR hierarchy.
        
        Returns
        -------
        dict
            Memory usage statistics in MB
        """
        # Base implementation - derived classes can override for more accuracy
        bytes_per_cell = 8  # double precision
        fields_per_cell = 2 if self.base_solver.enable_reactions else 1
        
        total_bytes = 0
        for level in range(self.max_levels):
            cells = self.get_level_cell_count(level)
            total_bytes += cells * bytes_per_cell * fields_per_cell
            
        return {
            'total_mb': total_bytes / (1024 * 1024),
            'per_level_mb': {
                level: self.get_level_cell_count(level) * bytes_per_cell * fields_per_cell / (1024 * 1024)
                for level in range(self.max_levels)
            }
        }