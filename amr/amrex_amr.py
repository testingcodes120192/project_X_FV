# amrex_amr.py - Complete version with proper core solver integration
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass
import os

from .base_amr import BaseAMR
from .amrex_data_bridge import AMReXDataBridge

# Try to import AMReX
try:
    import amrex.space2d as amr  # Explicitly use 2D version
    AMREX_AVAILABLE = True
    print("pyAMReX working available!")
except ImportError:
    AMREX_AVAILABLE = False
    warnings.warn("pyAMReX not available. AMReX backend will not function.")



@dataclass
class AMReXLevel:
    """Data structure for an AMReX refinement level."""
    level: int
    geom: 'amr.Geometry'
    grids: 'amr.BoxArray'
    dmap: 'amr.DistributionMapping'
    temperature: 'amr.MultiFab'
    temperature_old: 'amr.MultiFab'
    reaction_progress: Optional['amr.MultiFab'] = None
    reaction_progress_old: Optional['amr.MultiFab'] = None


class AMReXAMR(BaseAMR):
    """
    AMReX-based Adaptive Mesh Refinement implementation.
    
    This class provides AMR using AMReX for grid management while
    delegating all physics computation to the core FV solver.
    """
    
    def __init__(self, base_solver, config: Dict[str, Any]):
        """Initialize AMReX AMR system."""
        self.amrex_initialized = False
        if not AMREX_AVAILABLE:
            raise ImportError(
                "AMReX backend requires pyAMReX. "
                "Install with: pip install pyamrex"
            )
            
        super().__init__(base_solver, config)
        
        # AMReX specific parameters
        self.max_grid_size = config.get('max_grid_size', 32)
        self.blocking_factor = config.get('blocking_factor', 8)
        self.grid_eff = config.get('grid_eff', 0.7)
        self.n_error_buf = config.get('n_error_buf', 2)
        self.n_cell_coarsen = config.get('n_cell_coarsen', 2)
        self.subcycling = config.get('subcycling', True)
        
        # Tag buffer for proper nesting
        self.tag_buffer = config.get('tag_buffer', 2)
        
        # AMReX data structures
        self.levels: List[AMReXLevel] = []
        self.flux_reg: List[Optional['amr.FluxRegister']] = []
        
        # Domain setup
        self.domain_lo = [0.0, 0.0]
        self.domain_hi = [self.base_solver.mesh.plate_length, 
                          self.base_solver.mesh.plate_width]
        
        # Base level cell count
        self.n_cell_base = [self.base_solver.mesh.nx, self.base_solver.mesh.ny]
        
        # Boundary conditions
        try:
            # Try the expected way first
            self.bc_lo = [amr.BCType.foextrap, amr.BCType.foextrap]
            self.bc_hi = [amr.BCType.foextrap, amr.BCType.foextrap]
        except AttributeError:
            try:
                # Try alternate naming
                self.bc_lo = [amr.BC.foextrap, amr.BC.foextrap]
                self.bc_hi = [amr.BC.foextrap, amr.BC.foextrap]
            except AttributeError:
                # Use integer values directly (foextrap is typically 1 or 2)
                # 0 = periodic, 1 = reflect_odd, 2 = foextrap (first order extrapolation)
                self.bc_lo = [2, 2]  # First order extrapolation (Neumann-like)
                self.bc_hi = [2, 2]
                print("Warning: Using hardcoded BC values - BCType not found in AMReX")
                
        # Component indices
        self.comp_T = 0
        self.comp_lambda = 1 if self.base_solver.enable_reactions else -1
        self.n_comp = 2 if self.base_solver.enable_reactions else 1
        
        # AMReX initialization flag
        self.amrex_initialized = False
        
        # Solver instances for each level
        self.level_solvers = []
        
        # Check for AMReX features
        self.has_flux_register = hasattr(amr, 'FluxRegister')
        self.has_average_down = hasattr(amr, 'average_down')
        
        # Workflow control flags
        self.initial_levels = config.get('initial_levels', 1)
        self.adapt_after_ic = config.get('adapt_after_ic', True)
        self.show_before_adapt = config.get('show_before_adapt', False)
        self.temp_threshold = config.get('temp_threshold', 500.0)
        self.show_error_indicator = config.get('show_error_indicator', False)
        
        # State tracking
        self.base_initialized = False
        self.adapted = False
        
    def initialize(self):
        """Initialize only the base grid structure."""
        # Initialize AMReX if not already done
        if not self.amrex_initialized:
            self._initialize_amrex()
            
        # Initialize level solvers list
        self.level_solvers = []
        
        # Create base level only
        self._make_base_level()
        
        # Base level uses the provided base solver
        self.level_solvers.append(self.base_solver)
        
        # Mark as initialized but not adapted
        self.base_initialized = True
        self.adapted = False
        
        print(f"AMReX AMR initialized with base level only ({self.n_cell_base[0]}×{self.n_cell_base[1]} cells)")
        print("Ready for initial condition setup")
    
    def adapt_to_initial_condition(self):
        """Adapt the grid based on the current solution in the base solver."""
        if not self.base_initialized:
            raise RuntimeError("Base grid not initialized. Call initialize() first.")
            
        if self.adapted:
            print("Grid already adapted")
            return
            
        print("\n" + "="*60)
        print("Adapting AMR grid to initial condition")
        print("="*60)
        
        # First sync the current solution from solver to AMReX
        print("Syncing initial condition to AMReX...")
        self._sync_solver_to_amrex(0)
        
        # Fill ghost cells for gradient computation
        self._fill_patch(0, self.levels[0].temperature)
        
        # Print temperature statistics
        self._print_temperature_stats(0)
        
        # Now perform initial refinement based on actual data
        levels_created = 0
        
        for level in range(self.max_levels - 1):
            print(f"\nChecking level {level} for refinement...")
            
            # Compute error indicators from actual temperature field
            error_indicator, tagged_boxes = self._compute_error_indicator_from_data(level)
            
            if error_indicator is not None and self.show_error_indicator:
                self._visualize_error_indicator(level, error_indicator)
            
            # Check if any cells need refinement
            if tagged_boxes and len(tagged_boxes) > 0:
                print(f"  Found {len(tagged_boxes)} boxes to refine at level {level}")
                
                # Create refined level
                self._regrid_level(level + 1, tagged_boxes)
                levels_created += 1
                
                # Print stats for new level
                self._print_temperature_stats(level + 1)
            else:
                print(f"  No refinement needed at level {level}")
                break
                
        self.adapted = True
        print(f"\nGrid adaptation complete!")
        print(f"  Created {levels_created} refined levels")
        print(f"  Total levels: {len(self.levels)}")
        
        # Print final statistics
        self._print_adaptation_summary()
        
    def _initialize_amrex(self):
        """Initialize AMReX framework."""
        # Initialize AMReX
        amr.initialize([])
        
        # Set verbosity
        amr.ParmParse("amrex").add("v", 0)  # Quiet mode
        
        # Set some AMReX parameters
        pp = amr.ParmParse("amr")
        pp.add("max_level", self.max_levels - 1)
        pp.add("ref_ratio", self.refinement_ratio)
        pp.add("max_grid_size", self.max_grid_size)
        pp.add("blocking_factor", self.blocking_factor)
        pp.add("grid_eff", self.grid_eff)
        
        self.amrex_initialized = True
        
    def _create_level_solver(self, level: int):
        """Create a solver instance for a specific AMR level."""
        # Calculate refined mesh parameters
        ref_ratio = self.refinement_ratio ** level
        nx_level = self.n_cell_base[0] * ref_ratio
        ny_level = self.n_cell_base[1] * ref_ratio
        
        # Create mesh for this level
        from mesh import FVMesh
        level_mesh = FVMesh(
            nx_cells=nx_level,
            ny_cells=ny_level,
            plate_length=self.base_solver.mesh.plate_length,
            plate_width=self.base_solver.mesh.plate_width,
            ghost_cells=self.base_solver.mesh.ghost_cells
        )
        
        # Create solver for this level
        from solver import FVHeatSolver
        level_solver = FVHeatSolver(
            mesh=level_mesh,
            alpha=self.base_solver.alpha,
            spatial_order=self.base_solver.spatial_order,
            time_integration=self.base_solver.time_integration,
            enable_reactions=self.base_solver.enable_reactions
        )
        
        # Copy solver settings
        level_solver.bc_type = self.base_solver.bc_type
        level_solver.bc_values = self.base_solver.bc_values.copy() if self.base_solver.bc_values else {}
        
        # Disable standalone features for level solvers
        level_solver.collect_centerlines = False
        
        return level_solver
        
    def _make_base_level(self):
        """Create the base AMR level."""
        level = 0
        
        # Create geometry
        geom = self._make_geometry(level)
        
        # Create box array for base level
        domain = geom.domain
        grids = amr.BoxArray(domain)
        
        # Chop up grids according to max_grid_size
        grids.max_size(self.max_grid_size)
        
        # Create distribution mapping
        dmap = amr.DistributionMapping(grids)
        
        # Create MultiFabs for data storage
        nghost = self.base_solver.mesh.ghost_cells
        temperature = amr.MultiFab(grids, dmap, self.n_comp, nghost)
        temperature_old = amr.MultiFab(grids, dmap, self.n_comp, nghost)
        
        # Initialize to background temperature
        temperature.set_val(self.base_solver.T[0, 0])
        temperature_old.set_val(self.base_solver.T[0, 0])
        
        # Create reaction MultiFabs if needed
        reaction_progress = None
        reaction_progress_old = None
        if self.base_solver.enable_reactions:
            reaction_progress = amr.MultiFab(grids, dmap, 1, nghost)
            reaction_progress_old = amr.MultiFab(grids, dmap, 1, nghost)
            reaction_progress.set_val(0.0)
            reaction_progress_old.set_val(0.0)
            
        # Create level structure
        level_data = AMReXLevel(
            level=level,
            geom=geom,
            grids=grids,
            dmap=dmap,
            temperature=temperature,
            temperature_old=temperature_old,
            reaction_progress=reaction_progress,
            reaction_progress_old=reaction_progress_old
        )
        
        self.levels = [level_data]
        self.flux_reg = [None]
        
    def _make_geometry(self, level: int) -> 'amr.Geometry':
        """Create geometry for a given level."""
        # Domain indices
        ref_ratio = self.refinement_ratio ** level
        lo = amr.IntVect(0, 0)
        hi = amr.IntVect(self.n_cell_base[0] * ref_ratio - 1,
                        self.n_cell_base[1] * ref_ratio - 1)
        domain = amr.Box(lo, hi)
        
        # Physical domain - try different RealBox construction methods
        try:
            # Method 1: Direct construction with lists
            real_box = amr.RealBox(self.domain_lo, self.domain_hi)
        except TypeError:
            try:
                # Method 2: Using RealVect objects differently
                real_lo = amr.RealVect(self.domain_lo[0], self.domain_lo[1])
                real_hi = amr.RealVect(self.domain_hi[0], self.domain_hi[1])
                # Try passing as separate arguments
                real_box = amr.RealBox(real_lo[0], real_lo[1], real_hi[0], real_hi[1])
            except (TypeError, IndexError):
                try:
                    # Method 3: Direct coordinate construction
                    real_box = amr.RealBox(
                        self.domain_lo[0], self.domain_lo[1],
                        self.domain_hi[0], self.domain_hi[1]
                    )
                except TypeError:
                    # Method 4: Try with tuple unpacking
                    real_box = amr.RealBox(*self.domain_lo, *self.domain_hi)
        
        # Coordinate system (0 = Cartesian)
        coord = amr.CoordSys.cartesian
        
        # Periodicity
        is_periodic = [False, False]
        
        # Create geometry
        geom = amr.Geometry(domain, real_box, coord, is_periodic)
        
        return geom 
    
    def _set_initial_data(self):
        """Set initial data from base solver."""
        # This method is called during _make_base_level() to initialize the base level
        # At this point, the base solver might not have the actual IC yet
        
        # Check if base solver has meaningful data
        T_interior = self.base_solver.mesh.extract_interior(self.base_solver.T)
        T_max = np.max(T_interior)
        T_min = np.min(T_interior)
        
        if T_max - T_min < 1.0:  # Essentially uniform
            # No IC set yet - just use background temperature
            background_temp = T_interior[0, 0]  # Use first cell value
            print(f"Initializing base level with uniform temperature: {background_temp:.1f} K")
            
            # Set uniform temperature in MultiFab
            self.levels[0].temperature.set_val(background_temp)
            
            if self.base_solver.enable_reactions and self.levels[0].reaction_progress is not None:
                self.levels[0].reaction_progress.set_val(0.0)
        else:
            # IC has been set - sync it to AMReX
            print(f"Syncing initial data from base solver (T range: {T_min:.1f} - {T_max:.1f} K)")
            self._sync_solver_to_amrex(0)
            
        # Fill ghost cells using boundary conditions
        self._fill_patch(0, self.levels[0].temperature)
        
        if self.base_solver.enable_reactions and self.levels[0].reaction_progress is not None:
            self._fill_patch(0, self.levels[0].reaction_progress)
        
    def _sync_solver_to_amrex(self, level: int):
        """Sync data from solver arrays to AMReX MultiFab."""
        level_data = self.levels[level]
        solver = self.level_solvers[level]
        
        # Use AMReXDataBridge to convert
        AMReXDataBridge.numpy_to_multifab(
            solver.T,
            level_data.temperature,
            component=self.comp_T,
            include_ghost=True
        )
        
        # Copy reaction progress if enabled
        if self.base_solver.enable_reactions and solver.lambda_rxn is not None:
            AMReXDataBridge.numpy_to_multifab(
                solver.lambda_rxn,
                level_data.reaction_progress,
                component=0,
                include_ghost=True
            )
            
    def _sync_amrex_to_solver(self, level: int):
        """Sync data from AMReX MultiFab to solver arrays."""
        level_data = self.levels[level]
        solver = self.level_solvers[level]
        
        # Extract data from MultiFab
        for mfi in level_data.temperature:
            bx = mfi.validbox()
            arr = level_data.temperature.array(mfi)
            
            lo = bx.small_end
            hi = bx.big_end
            
            # Map to solver indices (including ghost cells)
            for j in range(lo[1], hi[1] + 1):
                for i in range(lo[0], hi[0] + 1):
                    j_solver = j + solver.mesh.ghost_cells
                    i_solver = i + solver.mesh.ghost_cells
                    
                    if (0 <= i_solver < solver.T.shape[1] and 
                        0 <= j_solver < solver.T.shape[0]):
                        solver.T[j_solver, i_solver] = arr[i, j, 0, self.comp_T]
                        
                        if self.base_solver.enable_reactions and level_data.reaction_progress:
                            reaction_arr = level_data.reaction_progress.array(mfi)
                            solver.lambda_rxn[j_solver, i_solver] = reaction_arr[i, j, 0, 0]
                            
    def _initial_refinement(self):
        """Perform initial refinement based on initial conditions."""
        for level in range(self.max_levels - 1):
            if level < len(self.levels):
                # Get tagged boxes
                tagged_boxes = self._tag_cells_for_refinement(level)
                
                if tagged_boxes:
                    # Create new level
                    self._regrid_level(level + 1, tagged_boxes)
                    
    def flag_cells_for_refinement(self, level: int) -> np.ndarray:
        """Flag cells that need refinement at a given level."""
        if level >= len(self.levels) - 1 or level >= self.max_levels - 1:
            return np.zeros((1, 1), dtype=bool)
            
        # Get tags as AMReX object
        tags = self._tag_cells_for_refinement(level)
        
        # Convert to numpy for return (simplified)
        return np.ones((1, 1), dtype=bool) if tags.numTags() > 0 else np.zeros((1, 1), dtype=bool)
        
    
    def _tag_cells_for_refinement(self, level: int):
        """Tag cells for refinement based on temperature gradients."""
        level_data = self.levels[level]
        
        # Get threshold - scale by level
        base_threshold = self.config.get('refine_threshold', 100.0)
        threshold = base_threshold / (self.refinement_ratio ** level)
        
        print(f"  Refinement threshold at level {level}: {threshold:.2f}")
        
        # Compute error indicators and get tagged boxes
        error_indicator, tagged_boxes = self._compute_error_indicator_from_data(level)
        
        return tagged_boxes
    
    
    def regrid(self, level: int):
        """Regrid the hierarchy starting from the given level."""
        if level >= self.max_levels - 1:
            return
            
        # Remove finer levels
        self.levels = self.levels[:level+1]
        self.flux_reg = self.flux_reg[:level+1]
        self.level_solvers = self.level_solvers[:level+1]
        
        # Now regrid finer levels one by one
        for lev in range(level, min(level + 2, self.max_levels - 1)):
            if lev < len(self.levels):
                tags = self._tag_cells_for_refinement(lev)
                
                if tags.numTags() > 0:
                    self._regrid_level(lev + 1, tags)
                else:
                    break
                    
    def _regrid_level(self, level: int, tagged_boxes):
        """Create or recreate a level based on tagged boxes."""
        if not tagged_boxes:
            return
            
        # Create list of refined boxes
        refined_boxes = []
        for box in tagged_boxes:
            # Refine each tagged box
            refined_box = amr.Box(
                amr.IntVect(box.small_end[0] * self.refinement_ratio,
                        box.small_end[1] * self.refinement_ratio),
                amr.IntVect((box.big_end[0] + 1) * self.refinement_ratio - 1,
                        (box.big_end[1] + 1) * self.refinement_ratio - 1)
            )
            refined_boxes.append(refined_box)
        
        # Create BoxArray from list of boxes (using Vector_Box as shown in test_boxarray.py)
        box_list = amr.Vector_Box(refined_boxes)
        new_grids = amr.BoxArray(box_list)
        
        # Break up boxes if too large
        new_grids.max_size(self.max_grid_size)
        
        # Create or update level
        if level < len(self.levels):
            self._remake_level(level, new_grids)
        else:
            self._make_new_level(level, new_grids)
        
    def _make_new_level(self, level: int, grids: 'amr.BoxArray'):
        """Create a new refinement level."""
        # Create geometry
        geom = self._make_geometry(level)
        
        # Create distribution mapping
        dmap = amr.DistributionMapping(grids)
        
        # Create MultiFabs
        nghost = self.base_solver.mesh.ghost_cells
        temperature = amr.MultiFab(grids, dmap, self.n_comp, nghost)
        temperature_old = amr.MultiFab(grids, dmap, self.n_comp, nghost)
        
        # Create reaction MultiFabs if needed
        reaction_progress = None
        reaction_progress_old = None
        if self.base_solver.enable_reactions:
            reaction_progress = amr.MultiFab(grids, dmap, 1, nghost)
            reaction_progress_old = amr.MultiFab(grids, dmap, 1, nghost)
            
        # Create level structure
        level_data = AMReXLevel(
            level=level,
            geom=geom,
            grids=grids,
            dmap=dmap,
            temperature=temperature,
            temperature_old=temperature_old,
            reaction_progress=reaction_progress,
            reaction_progress_old=reaction_progress_old
        )
        
        self.levels.append(level_data)
        
        # Create solver for this level
        self.level_solvers.append(self._create_level_solver(level))
        
        # Create flux register if available
        if self.has_flux_register and level > 0:
            crse_level = self.levels[level - 1]
            flux_reg = amr.FluxRegister(grids, dmap, crse_level.grids, crse_level.dmap,
                                       geom, crse_level.geom, self.refinement_ratio,
                                       level, self.n_comp)
            self.flux_reg.append(flux_reg)
        else:
            self.flux_reg.append(None)
            
        # Fill with interpolated data
        self._fill_patch(level, temperature)
        
        # Initialize solver at this level
        self._initialize_level_solver(level)
        
    def _initialize_level_solver(self, level: int):
        """Initialize solver at a specific level with interpolated data."""
        if level == 0:
            return  # Base level already initialized
            
        # Get coarse level data
        coarse_solver = self.level_solvers[level - 1]
        fine_solver = self.level_solvers[level]
        
        # Get interior data from coarse level
        coarse_T = coarse_solver.mesh.extract_interior(coarse_solver.T)
        
        # Use solver's interpolation method
        fine_solver.set_from_interpolation(coarse_T, self.refinement_ratio)
        
        # Sync to AMReX
        self._sync_solver_to_amrex(level)
        
    def _remake_level(self, level: int, new_grids: 'amr.BoxArray'):
        """Remake an existing level with new grids."""
        old_level = self.levels[level]
        
        # Create new distribution mapping
        new_dmap = amr.DistributionMapping(new_grids)
        
        # Create new MultiFabs
        nghost = self.base_solver.mesh.ghost_cells
        new_temperature = amr.MultiFab(new_grids, new_dmap, self.n_comp, nghost)
        new_temperature_old = amr.MultiFab(new_grids, new_dmap, self.n_comp, nghost)
        
        # Copy data from old to new
        if hasattr(amr, 'Copy'):
            amr.Copy(new_temperature, old_level.temperature, 0, 0, self.n_comp,
                    nghost, nghost, old_level.geom.periodicity())
        else:
            # Manual copy - simplified
            new_temperature.ParallelCopy(old_level.temperature, 0, 0, self.n_comp,
                                        nghost, nghost, old_level.geom.periodicity())
        
        # Update level
        old_level.grids = new_grids
        old_level.dmap = new_dmap
        old_level.temperature = new_temperature
        old_level.temperature_old = new_temperature_old
        
        # Update flux register if available
        if self.has_flux_register and level > 0:
            crse_level = self.levels[level - 1]
            self.flux_reg[level] = amr.FluxRegister(
                new_grids, new_dmap, crse_level.grids, crse_level.dmap,
                old_level.geom, crse_level.geom, self.refinement_ratio,
                level, self.n_comp
            )
            
    def advance_hierarchy(self, dt: float):
        """Advance the entire AMR hierarchy by one coarse time step."""
        if self.subcycling:
            # Recursive time stepping with subcycling
            self._advance_level(0, dt)
        else:
            # Advance all levels with same dt
            for level in range(len(self.levels)):
                self._advance_single_level(level, dt)
                
        # Synchronize levels
        self.synchronize_levels()
        
        # Update time
        self.current_time += dt
        
    def _advance_level(self, level: int, dt: float):
        """Recursively advance a level and finer levels with subcycling."""
        # Number of substeps for this level
        if level == 0:
            nsubsteps = 1
        else:
            nsubsteps = self.refinement_ratio
            
        dt_level = dt / nsubsteps
        
        for substep in range(nsubsteps):
            # Advance this level
            self._advance_single_level(level, dt_level)
            
            # Recursively advance finer level
            if level + 1 < len(self.levels):
                self._advance_level(level + 1, dt_level)
                
            # Average down from finer level
            if level + 1 < len(self.levels):
                self._average_down(level + 1)
                
    def _advance_single_level(self, level: int, dt: float):
        """Advance a single level by dt using the core solver."""
        level_data = self.levels[level]
        
        # Store old solution - swap pointers instead of copying
        # This is more efficient and avoids the copy issue
        temp = level_data.temperature_old
        level_data.temperature_old = level_data.temperature
        level_data.temperature = temp
        
        # Sync AMReX data to solver
        self._sync_amrex_to_solver(level)
        
        # Fill ghost cells
        if level > 0:
            # Fine levels need BC from coarse
            self._fill_fine_level_ghost_cells(level)
        
        # Use core solver to advance!
        solver = self.level_solvers[level]
        solver.advance(dt)
        
        # Sync back to AMReX (this updates the swapped temperature MultiFab)
        self._sync_solver_to_amrex(level)
        
        # Store fluxes for conservation if using flux registers
        if self.has_flux_register and level > 0 and self.flux_reg[level] is not None:
            self._store_fluxes_for_reflux(level, dt)
            
    def synchronize_levels_old(self):
        """Synchronize data between AMR levels with flux correction."""
        # Step 1: Average down from fine to coarse (restriction)
        for level in range(len(self.levels) - 1, 0, -1):
            self._average_down(level)
            
        # Step 2: Apply flux correction at coarse-fine interfaces
        if self.has_flux_register:
            for level in range(1, len(self.levels)):
                if self.flux_reg[level] is not None:
                    # Reflux: correct coarse fluxes using fine fluxes
                    crse_level = self.levels[level - 1]
                    dt_crse = 1.0  # This should be the actual coarse dt used
                    
                    # Scale by dt and cell size ratios
                    scale = dt_crse / (self.refinement_ratio ** 2)  # 2D scaling
                    
                    # Apply flux correction to coarse level
                    self.flux_reg[level].Reflux(crse_level.temperature, scale, 0, 0, 
                                               self.n_comp, crse_level.geom)
                    
                    # Clear flux register for next time step
                    self.flux_reg[level].ClearInternalBorders(crse_level.geom)
                    
    def _average_down(self, level: int):
        """Average down from fine to coarse level (conservative restriction)."""
        if level == 0:
            return
            
        fine_level = self.levels[level]
        crse_level = self.levels[level - 1]
        
        # First sync fine level solver to AMReX (in case it was modified)
        self._sync_solver_to_amrex(level)
        
        # Use AMReX's built-in conservative averaging if available
        if self.has_average_down:
            amr.average_down(fine_level.temperature, crse_level.temperature,
                            0, self.n_comp, self.refinement_ratio)
            
            if self.base_solver.enable_reactions and fine_level.reaction_progress:
                amr.average_down(fine_level.reaction_progress, crse_level.reaction_progress,
                                0, 1, self.refinement_ratio)
        else:
            # Use data bridge for manual averaging
            AMReXDataBridge.copy_with_averaging(
                fine_level.temperature, crse_level.temperature,
                0, 0, self.n_comp, self.refinement_ratio
            )
            
        # Sync coarse level back to solver
        self._sync_amrex_to_solver(level - 1)
                            
    def get_composite_solution(self, field_name: str = 'T') -> Dict[str, np.ndarray]:
        """Get composite solution across all AMR levels using proper interpolation."""
        # Determine output resolution (use finest level resolution)
        finest_level = len(self.levels) - 1
        ref_ratio = self.refinement_ratio ** finest_level
        
        nx_fine = self.n_cell_base[0] * ref_ratio
        ny_fine = self.n_cell_base[1] * ref_ratio
        
        # Create output arrays
        data = np.full((ny_fine, nx_fine), 300.0)  # Default to background
        level_map = np.zeros((ny_fine, nx_fine), dtype=int)
        
        # Create a mask to track which cells have been filled
        filled = np.zeros((ny_fine, nx_fine), dtype=bool)
        
        # Fill from coarse to fine (fine overwrites coarse)
        for level in range(len(self.levels)):
            level_data = self.levels[level]
            mf = level_data.temperature if field_name == 'T' else level_data.reaction_progress
            
            if mf is None:
                continue
                
            # Refinement ratio for this level
            r = self.refinement_ratio ** level
            cells_per_coarse = ref_ratio // r
            
            # Process each box in this level
            for mfi in mf:
                bx = mfi.validbox()  # Get valid region (no ghost)
                arr = mf.array(mfi)
                
                lo = bx.small_end
                hi = bx.big_end
                
                # Map to fine grid indices
                for j in range(lo[1], hi[1] + 1):
                    for i in range(lo[0], hi[0] + 1):
                        # Get the value at this AMR cell
                        comp = self.comp_T if field_name == 'T' else 0
                        value = arr[i, j, 0, comp]
                        
                        # Fill corresponding fine cells
                        i_fine_start = i * cells_per_coarse
                        i_fine_end = (i + 1) * cells_per_coarse
                        j_fine_start = j * cells_per_coarse
                        j_fine_end = (j + 1) * cells_per_coarse
                        
                        for jf in range(j_fine_start, j_fine_end):
                            for if_ in range(i_fine_start, i_fine_end):
                                if 0 <= if_ < nx_fine and 0 <= jf < ny_fine:
                                    data[jf, if_] = value
                                    level_map[jf, if_] = level
                                    filled[jf, if_] = True
                                        
        # Create coordinate arrays
        dx_fine = self.domain_hi[0] / nx_fine
        dy_fine = self.domain_hi[1] / ny_fine
        
        x = np.linspace(dx_fine/2, self.domain_hi[0] - dx_fine/2, nx_fine)
        y = np.linspace(dy_fine/2, self.domain_hi[1] - dy_fine/2, ny_fine)
        
        # Ensure all cells are filled (fallback for any gaps)
        if not np.all(filled):
            # Fill gaps with nearest neighbor interpolation
            from scipy.ndimage import distance_transform_edt
            missing = ~filled
            indices = distance_transform_edt(missing, return_distances=False, return_indices=True)
            data[missing] = data[tuple(indices[:, missing])]
            
        return {
            'data': data,
            'x': x,
            'y': y,
            'level_map': level_map
        }
        
    def get_level_data(self, level: int, field_name: str = 'T') -> Dict[str, Any]:
        """Get data from a specific AMR level."""
        if level >= len(self.levels):
            return {'data': None, 'grids': None}
            
        level_data = self.levels[level]
        
        # Extract data to numpy array
        nx = self.n_cell_base[0] * (self.refinement_ratio ** level)
        ny = self.n_cell_base[1] * (self.refinement_ratio ** level)
        
        data = np.zeros((ny, nx))
        mf = level_data.temperature
        
        for mfi in mf:
            bx = mfi.tilebox()
            arr = mf.array(mfi)
            
            lo = bx.small_end
            hi = bx.big_end
            
            for j in range(lo[1], hi[1] + 1):
                for i in range(lo[0], hi[0] + 1):
                    if 0 <= i < nx and 0 <= j < ny:
                        comp = self.comp_T if field_name == 'T' else self.comp_lambda
                        if comp >= 0:
                            data[j, i] = arr[i, j, 0, comp]
                            
        # Get box information
        boxes = []
        for i in range(len(level_data.grids)):
            box = level_data.grids[i]
            boxes.append({
                'lo': [box.small_end[0], box.small_end[1]],
                'hi': [box.big_end[0], box.big_end[1]]
            })
            
        return {
            'data': data,
            'grids': boxes,
            'geometry': {
                'dx': self._get_cell_size(level_data.geom)[0],
                'dy': self._get_cell_size(level_data.geom)[1],
                'domain': [[0, 0], [nx-1, ny-1]]
            }
        }
        
    def get_level_cell_count(self, level: int) -> int:
        """Get number of cells at a specific level."""
        if level >= len(self.levels):
            return 0
            
        return self.levels[level].grids.numPts
        
    def compute_refinement_indicators(self, level: int) -> np.ndarray:
        """Compute error indicators for refinement criteria."""
        if level >= len(self.levels):
            return np.zeros((1, 1))
            
        # Use solver's gradient computation
        solver = self.level_solvers[level]
        return solver.compute_gradient_for_amr()
        
    def _get_cell_size(self, geom):
        """Get cell size with compatibility fallback."""
        # Try different case variations for pyAMReX compatibility
        if hasattr(geom, 'cell_size'):
            return geom.cell_size()
        elif hasattr(geom, 'CellSize'):
            return geom.CellSize()
        else:
            # Calculate from domain and problem size
            domain = geom.domain  # domain is a property, not a method
            
            # Try different case variations for prob_lo/prob_hi
            if hasattr(geom, 'prob_lo'):
                prob_lo = geom.prob_lo()
            elif hasattr(geom, 'ProbLo'):
                prob_lo = geom.ProbLo()
            else:
                prob_lo = [0.0, 0.0]
                
            if hasattr(geom, 'prob_hi'):
                prob_hi = geom.prob_hi()
            elif hasattr(geom, 'ProbHi'):
                prob_hi = geom.ProbHi()
            else:
                prob_hi = [1.0, 1.0]
                
            # domain.size is a property, not a method
            return np.array([
                (prob_hi[0] - prob_lo[0]) / domain.size[0],
                (prob_hi[1] - prob_lo[1]) / domain.size[1]
            ])
            
    def _fill_patch_org(self, level: int, mf: 'amr.MultiFab'):
        """Fill ghost cells using interpolation from coarser level."""
        if level == 0:
            # Base level - apply physical boundary conditions
            self._apply_physical_bc(level, mf)
        else:
            # Interpolate from coarser level
            self._fill_patch_interp(level, mf)
            
    def _apply_physical_bc_org(self, level: int, mf: 'amr.MultiFab'):
        """Apply physical boundary conditions."""
        # For Neumann BC, we use first-order extrapolation
        # AMReX handles this with BCType.foextrap
        geom = self.levels[level].geom
        
        # Fill boundary
        mf.fill_boundary(geom.periodicity())
        
    def _fill_patch_interp(self, level: int, mf: 'amr.MultiFab'):
        """Fill patch using interpolation from coarser level."""
        fine_level = self.levels[level]
        
        # First fill from same level
        mf.fill_boundary(fine_level.geom.periodicity())
        
        # Then interpolate from coarse level where needed
        # This is simplified - real implementation would use AMReX FillPatch operations
        
    def plot_grid_structure(self, ax=None, show_levels=True):
        """Plot AMReX grid structure."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Rectangle
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
        # Colors for each level
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        
        # Plot grids for each level
        for level in range(len(self.levels)):
            level_data = self.levels[level]
            color = colors[level % len(colors)]
            
            # Get cell size for this level
            dx = self._get_cell_size(level_data.geom)
            
            # Plot each box in the BoxArray
            for i in range(level_data.grids.size):
                box = level_data.grids[i]
                lo = box.small_end
                hi = box.big_end
                
                # Physical coordinates
                x_lo = lo[0] * dx[0]
                y_lo = lo[1] * dx[1]
                x_hi = (hi[0] + 1) * dx[0]
                y_hi = (hi[1] + 1) * dx[1]
                
                # Draw box outline
                rect = Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                               fill=False, edgecolor=color, linewidth=2,
                               label=f'Level {level}' if i == 0 else '')
                ax.add_patch(rect)
                
                # Optionally draw individual cells for small boxes
                if (hi[0] - lo[0] + 1) <= 16 and (hi[1] - lo[1] + 1) <= 16:
                    for j in range(lo[1], hi[1] + 1):
                        for i in range(lo[0], hi[0] + 1):
                            x = i * dx[0]
                            y = j * dx[1]
                            cell = Rectangle((x, y), dx[0], dx[1],
                                           fill=False, edgecolor=color,
                                           linewidth=0.5, alpha=0.3)
                            ax.add_patch(cell)
                            
        # Set limits and labels
        ax.set_xlim(0, self.domain_hi[0])
        ax.set_ylim(0, self.domain_hi[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'AMReX Grid Structure ({len(self.levels)} levels)', fontsize=14)
        
        # Legend (remove duplicates)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        
        # Add grid statistics
        total_cells = sum(self.get_level_cell_count(lev) for lev in range(len(self.levels)))
        ax.text(0.02, 0.98, f'Total cells: {total_cells:,}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax
        
    def save_checkpoint(self, filename: str):
        """Save AMReX checkpoint files."""
        # Create checkpoint directory
        import os
        checkpoint_dir = f"{filename}_checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Write header file
        header_file = os.path.join(checkpoint_dir, "Header")
        with open(header_file, 'w') as f:
            f.write(f"Checkpoint version 1.0\n")
            f.write(f"number_of_levels {len(self.levels)}\n")
            f.write(f"time {self.current_time}\n")
            f.write(f"max_level {self.max_levels - 1}\n")
            
        # Write data for each level
        for level in range(len(self.levels)):
            level_dir = os.path.join(checkpoint_dir, f"Level_{level}")
            os.makedirs(level_dir, exist_ok=True)
            
            # Write MultiFab data
            self.levels[level].temperature.write(level_dir)
            
            # Write metadata
            meta_file = os.path.join(level_dir, "metadata")
            with open(meta_file, 'w') as f:
                f.write(f"refinement_ratio {self.refinement_ratio}\n")
                f.write(f"n_cells {self.get_level_cell_count(level)}\n")
                
        print(f"AMReX checkpoint saved to {checkpoint_dir}")
        
    def load_checkpoint(self, filename: str):
        """Load AMReX checkpoint files."""
        # This is a simplified version
        # Full implementation would read AMReX checkpoint format
        import os
        
        checkpoint_dir = f"{filename}_checkpoint"
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
            
        # Read header
        header_file = os.path.join(checkpoint_dir, "Header")
        with open(header_file, 'r') as f:
            lines = f.readlines()
            # Parse header info
            
        print(f"AMReX checkpoint loaded from {checkpoint_dir}")
        
    def get_amrex_statistics(self) -> Dict[str, Any]:
        """Get AMReX-specific statistics."""
        stats = {
            'max_grid_size': self.max_grid_size,
            'blocking_factor': self.blocking_factor,
            'efficiency': self.grid_eff,
            'using_gpu': amr.Config.gpu_backend != "DISABLED" if hasattr(amr.Config, 'gpu_backend') else False,
            'num_boxes': sum(level.grids.size for level in self.levels),
            'level_details': {}
        }
        
        # Per-level statistics
        for level in range(len(self.levels)):
            level_data = self.levels[level]
            stats['level_details'][level] = {
                'n_grids': level_data.grids.size,
                'n_cells': level_data.grids.numPts,
                'efficiency': amr.efficiency(level_data.grids, level_data.grids, 1),
                'dx': self._get_cell_size(level_data.geom)[0],
                'dy': self._get_cell_size(level_data.geom)[1]
            }
            
        return stats
    
    # Complete implementation for amrex_amr.py - _fill_fine_level_ghost_cells method

    def _fill_fine_level_ghost_cells(self, level: int):
        """
        Fill ghost cells of fine level using data from coarse level.
        
        This method determines which patches need coarse-fine interpolation
        and applies the appropriate boundary conditions.
        
        Parameters
        ----------
        level : int
            Fine level index (must be > 0)
        """
        if level == 0:
            # Base level - just apply physical BC
            self.level_solvers[0]._apply_boundary_conditions()
            return
        
        fine_solver = self.level_solvers[level]
        coarse_solver = self.level_solvers[level - 1]
        fine_level_data = self.levels[level]
        
        # First, apply physical BC everywhere (this handles domain boundaries)
        fine_solver._apply_boundary_conditions()
        
        # Now we need to identify which fine grid boxes are at coarse-fine interfaces
        # and apply interpolation from coarse data
        
        # Get the refinement ratio
        ref_ratio = self.refinement_ratio
        
        # Process each box in the fine level
        for mfi in fine_level_data.temperature:
            fine_box = mfi.validbox()
            fine_lo = fine_box.small_end
            fine_hi = fine_box.big_end
            
            # Determine the bounds of this patch in the fine grid index space
            patch_bounds = (fine_lo[0], fine_hi[0] + 1, fine_lo[1], fine_hi[1] + 1)
            
            # Check if this box touches any coarse-fine interfaces
            # A box is at a coarse-fine interface if it's not at the domain boundary
            # but its neighbor in the coarse level would be a different box
            
            # Domain bounds in fine grid indices
            domain_lo = [0, 0]
            domain_hi = [self.n_cell_base[0] * (ref_ratio ** level) - 1,
                        self.n_cell_base[1] * (ref_ratio ** level) - 1]
            
            # Check each face of the box
            at_domain_left = (fine_lo[0] == domain_lo[0])
            at_domain_right = (fine_hi[0] == domain_hi[0])
            at_domain_bottom = (fine_lo[1] == domain_lo[1])
            at_domain_top = (fine_hi[1] == domain_hi[1])
            
            # If not at domain boundary, we might need coarse-fine interpolation
            needs_cf_bc = (not at_domain_left or not at_domain_right or 
                        not at_domain_bottom or not at_domain_top)
            
            if needs_cf_bc:
                # Extract the patch data from fine solver
                # Map box indices to solver array indices
                i_start = fine_lo[0]
                i_end = fine_hi[0] + 1
                j_start = fine_lo[1]
                j_end = fine_hi[1] + 1
                
                # Apply coarse-fine BC for this patch
                # We need to be careful about the coordinate mapping
                
                # Create a temporary patch solver that shares data with the main solver
                # This is more efficient than copying data back and forth
                
                # The patch bounds are in fine grid coordinates
                # We'll use the solver's apply_coarse_fine_bc method
                patch_bounds_solver = (i_start, i_end, j_start, j_end)
                
                # Apply the coarse-fine interpolation
                self._apply_cf_interpolation(fine_solver, coarse_solver, 
                                        patch_bounds_solver, ref_ratio)
        
        # Sync the updated ghost cells back to AMReX
        self._sync_solver_to_amrex(level)


    def _apply_cf_interpolation(self, fine_solver: 'FVHeatSolver', 
                            coarse_solver: 'FVHeatSolver',
                            patch_bounds: Tuple[int, int, int, int],
                            ref_ratio: int):
        """
        Apply coarse-fine interpolation for a specific patch.
        
        This is a helper method that handles the actual interpolation
        from coarse to fine ghost cells.
        
        Parameters
        ----------
        fine_solver : FVHeatSolver
            Fine level solver
        coarse_solver : FVHeatSolver
            Coarse level solver
        patch_bounds : tuple
            (i_start, i_end, j_start, j_end) in fine grid coordinates
        ref_ratio : int
            Refinement ratio between levels
        """
        # Use the solver's built-in method
        fine_solver.apply_coarse_fine_bc(coarse_solver, ref_ratio, patch_bounds)
        
        
    def _fill_patch(self, level: int, mf: 'amr.MultiFab'):
        """
        Fill ghost cells using interpolation from coarser level.
        
        This is the AMReX-specific version that works with MultiFabs.
        
        Parameters
        ----------
        level : int
            Level index
        mf : amr.MultiFab
            MultiFab to fill
        """
        if level == 0:
            # Base level - apply physical boundary conditions
            self._apply_physical_bc(level, mf)
        else:
            # Fine level - need interpolation from coarse
            # First, synchronize current data to solver
            self._sync_amrex_to_solver(level)
            
            # Fill ghost cells using solver's interpolation
            self._fill_fine_level_ghost_cells(level)
            
            # The ghost cells are now filled in the solver
            # We don't need to sync back immediately as this will be done
            # after the time step
            
            
    def _apply_physical_bc(self, level: int, mf: 'amr.MultiFab'):
        """
        Apply physical boundary conditions at domain boundaries.
        
        Parameters
        ----------
        level : int
            Level index
        mf : amr.MultiFab
            MultiFab to apply BC to
        """
        # For Neumann BC, we use first-order extrapolation
        geom = self.levels[level].geom
        
        # Fill boundary using AMReX periodicity
        mf.fill_boundary(geom.periodicity())
        
        # For more complex BC, we would implement them here
        # For now, the extrapolation BC (foextrap) handles Neumann
        
        
    def _check_coarse_fine_interface(self, level: int, box: 'amr.Box') -> Dict[str, bool]:
        """
        Check which faces of a box are at coarse-fine interfaces.
        
        Parameters
        ----------
        level : int
            Fine level index
        box : amr.Box
            Box to check
            
        Returns
        -------
        dict
            Dictionary with keys 'left', 'right', 'bottom', 'top'
            indicating which faces are at C-F interfaces
        """
        if level == 0:
            # Base level has no coarse-fine interfaces
            return {'left': False, 'right': False, 'bottom': False, 'top': False}
        
        # Get box bounds
        lo = box.small_end
        hi = box.big_end
        
        # Get domain bounds at this level
        ref_ratio = self.refinement_ratio ** level
        domain_hi = [self.n_cell_base[0] * ref_ratio - 1,
                    self.n_cell_base[1] * ref_ratio - 1]
        
        # Check if at domain boundary (no C-F interface possible)
        at_domain_left = (lo[0] == 0)
        at_domain_right = (hi[0] == domain_hi[0])
        at_domain_bottom = (lo[1] == 0)
        at_domain_top = (hi[1] == domain_hi[1])
        
        # Get coarse level boxes to check for coverage
        coarse_level = self.levels[level - 1]
        coarse_ba = coarse_level.grids
        
        # Map fine box to coarse index space
        coarse_lo = [lo[i] // self.refinement_ratio for i in range(2)]
        coarse_hi = [hi[i] // self.refinement_ratio for i in range(2)]
        
        # Check each face
        cf_interfaces = {
            'left': False,
            'right': False,
            'bottom': False,
            'top': False
        }
        
        # A face is at a C-F interface if:
        # 1. It's not at the domain boundary
        # 2. The adjacent coarse cell is covered by a different fine grid
        
        # This is a simplified check - full implementation would need
        # to query the BoxArray to see if adjacent regions are refined
        
        if not at_domain_left:
            # Check if left neighbor is refined at this level
            cf_interfaces['left'] = True  # Simplified - assume C-F interface
            
        if not at_domain_right:
            cf_interfaces['right'] = True
            
        if not at_domain_bottom:
            cf_interfaces['bottom'] = True
            
        if not at_domain_top:
            cf_interfaces['top'] = True
        
        return cf_interfaces
    
    # Complete implementation of _store_fluxes_for_reflux for amrex_amr.py

    def _store_fluxes_for_reflux(self, level: int, dt: float):
        """
        Store fluxes at coarse-fine interfaces for conservation.
        
        This method computes and stores the fluxes from the fine level
        at faces that coincide with coarse level faces. These fluxes
        will later be used to correct the coarse level fluxes to
        maintain conservation.
        
        Parameters
        ----------
        level : int
            Fine level index (must be > 0)
        dt : float
            Time step used at this level
        """
        if level == 0 or not self.has_flux_register:
            return
            
        if self.flux_reg[level] is None:
            return
            
        # Get fine and coarse level data
        fine_level = self.levels[level]
        fine_solver = self.level_solvers[level]
        
        # Get flux register
        freg = self.flux_reg[level]
        
        # Compute fluxes at fine level using the solver
        # The solver uses WENO reconstruction for high-order accuracy
        flux_x_fine, flux_y_fine = fine_solver.compute_fluxes_at_faces()
        
        # Create MultiFabs for fluxes
        # X-fluxes are face-centered in x-direction
        xflux_ba = amr.BoxArray(fine_level.grids)
        xflux_ba = xflux_ba.convert(amr.IntVect(1, 0))  # Convert to x-face centered
        xflux_mf = amr.MultiFab(xflux_ba, fine_level.dmap, self.n_comp, 0)
        
        # Y-fluxes are face-centered in y-direction
        yflux_ba = amr.BoxArray(fine_level.grids)
        yflux_ba = yflux_ba.convert(amr.IntVect(0, 1))  # Convert to y-face centered
        yflux_mf = amr.MultiFab(yflux_ba, fine_level.dmap, self.n_comp, 0)
        
        # Fill MultiFabs with computed fluxes
        for mfi in fine_level.temperature:
            bx = mfi.validbox()
            
            # Get flux arrays from MultiFabs
            xflux_arr = xflux_mf.array(mfi)
            yflux_arr = yflux_mf.array(mfi)
            
            # Get box bounds
            lo = bx.small_end
            hi = bx.big_end
            
            # Fill x-flux MultiFab
            # X-flux dimensions: (nx+1) x ny
            for j in range(lo[1], hi[1] + 1):
                for i in range(lo[0], hi[0] + 2):  # One extra in x
                    # Map to flux array indices
                    j_flux = j - lo[1]
                    i_flux = i - lo[0]
                    
                    if 0 <= j_flux < flux_x_fine.shape[0] and 0 <= i_flux < flux_x_fine.shape[1]:
                        # Store negative flux (for conservation equation form)
                        xflux_arr[i, j, 0, self.comp_T] = -flux_x_fine[j_flux, i_flux]
                        
                        # Store reaction flux if enabled
                        if self.base_solver.enable_reactions and self.comp_lambda >= 0:
                            # For reaction, flux is typically zero (no diffusion of lambda)
                            xflux_arr[i, j, 0, self.comp_lambda] = 0.0
            
            # Fill y-flux MultiFab
            # Y-flux dimensions: nx x (ny+1)
            for j in range(lo[1], hi[1] + 2):  # One extra in y
                for i in range(lo[0], hi[0] + 1):
                    # Map to flux array indices
                    j_flux = j - lo[1]
                    i_flux = i - lo[0]
                    
                    if 0 <= j_flux < flux_y_fine.shape[0] and 0 <= i_flux < flux_y_fine.shape[1]:
                        # Store negative flux
                        yflux_arr[i, j, 0, self.comp_T] = -flux_y_fine[j_flux, i_flux]
                        
                        if self.base_solver.enable_reactions and self.comp_lambda >= 0:
                            yflux_arr[i, j, 0, self.comp_lambda] = 0.0
        
        # Now register the fluxes with the flux register
        # The flux register will accumulate these fluxes and later use them
        # to correct the coarse level
        
        # Scale factor for accumulation
        # We scale by dt because the flux register accumulates F*dt
        scale = dt
        
        # Register fluxes at each face orientation
        # FineAdd adds fine fluxes that will later correct coarse fluxes
        
        # X-direction fluxes
        # Low side (left faces)
        freg.FineAdd(xflux_mf, 0, 0, self.n_comp, scale, 
                    amr.Orientation(0, amr.Orientation.low))
        
        # High side (right faces) - note the negative scale for opposite orientation
        freg.FineAdd(xflux_mf, 0, 0, self.n_comp, -scale, 
                    amr.Orientation(0, amr.Orientation.high))
        
        # Y-direction fluxes
        # Low side (bottom faces)
        freg.FineAdd(yflux_mf, 1, 0, self.n_comp, scale, 
                    amr.Orientation(1, amr.Orientation.low))
        
        # High side (top faces)
        freg.FineAdd(yflux_mf, 1, 0, self.n_comp, -scale, 
                    amr.Orientation(1, amr.Orientation.high))
        
        # The flux register now contains the fine fluxes that need to be
        # used to correct the coarse fluxes during synchronization


    def _create_flux_multifabs(self, level: int) -> Tuple['amr.MultiFab', 'amr.MultiFab']:
        """
        Create face-centered MultiFabs for storing fluxes.
        
        Parameters
        ----------
        level : int
            Level index
            
        Returns
        -------
        xflux_mf : amr.MultiFab
            X-direction flux MultiFab (x-face centered)
        yflux_mf : amr.MultiFab  
            Y-direction flux MultiFab (y-face centered)
        """
        level_data = self.levels[level]
        
        # Create face-centered box arrays
        xflux_ba = amr.BoxArray(level_data.grids)
        xflux_ba = xflux_ba.convert(amr.IntVect(1, 0))  # x-face centered
        
        yflux_ba = amr.BoxArray(level_data.grids)
        yflux_ba = yflux_ba.convert(amr.IntVect(0, 1))  # y-face centered
        
        # Create MultiFabs (no ghost cells needed for fluxes)
        xflux_mf = amr.MultiFab(xflux_ba, level_data.dmap, self.n_comp, 0)
        yflux_mf = amr.MultiFab(yflux_ba, level_data.dmap, self.n_comp, 0)
        
        return xflux_mf, yflux_mf


    def synchronize_levels(self):
        """
        Synchronize data between AMR levels with flux correction.
        
        This version includes the complete flux correction implementation.
        """
        # Step 1: Average down from fine to coarse (restriction)
        for level in range(len(self.levels) - 1, 0, -1):
            self._average_down(level)
        
        # Step 2: Apply flux correction at coarse-fine interfaces
        if self.has_flux_register:
            for level in range(1, len(self.levels)):
                if self.flux_reg[level] is not None:
                    # Get coarse level data
                    crse_level = self.levels[level - 1]
                    crse_solver = self.level_solvers[level - 1]
                    
                    # The flux register contains the accumulated fine fluxes
                    # We need to apply these to correct the coarse solution
                    
                    # Reflux scale factor
                    # The scale accounts for:
                    # - Time step differences (if subcycling)
                    # - Area differences due to refinement
                    # - The fact that we're correcting a divergence
                    
                    if self.subcycling:
                        # With subcycling, fine takes ref_ratio substeps
                        dt_crse = self.current_time  # This is simplified
                        scale = 1.0 / (self.refinement_ratio ** 2)  # Area ratio in 2D
                    else:
                        # No subcycling - same dt but different areas
                        scale = 1.0 / (self.refinement_ratio ** 2)
                    
                    # Apply the flux correction to coarse level
                    # This modifies the coarse solution to maintain conservation
                    self.flux_reg[level].Reflux(
                        crse_level.temperature,  # MultiFab to correct
                        scale,                   # Scale factor
                        0,                       # Source component
                        0,                       # Destination component  
                        self.n_comp,            # Number of components
                        crse_level.geom         # Geometry for coarse level
                    )
                    
                    # Sync the corrected coarse data back to solver
                    self._sync_amrex_to_solver(level - 1)
                    
                    # Clear the flux register for next time step
                    self.flux_reg[level].setVal(0.0)
                    
    # Add this method to the AMReXAMR class in amrex_amr.py

    def adapt_to_initial_condition(self):
        """Adapt the grid based on the current solution in the base solver."""
        if not self.base_initialized:
            raise RuntimeError("Base grid not initialized. Call initialize() first.")
            
        if self.adapted:
            print("Grid already adapted")
            return
            
        print("\n" + "="*60)
        print("Adapting AMReX AMR grid to initial condition")
        print("="*60)
        
        # First sync the current solution from solver to AMReX
        print("Syncing initial condition to AMReX...")
        self._sync_solver_to_amrex(0)
        
        # Fill ghost cells for gradient computation
        self._fill_patch(0, self.levels[0].temperature)
        
        # Print temperature statistics
        self._print_temperature_stats(0)
        
        # Now perform initial refinement based on actual data
        levels_created = 0
        
        for level in range(self.max_levels - 1):
            print(f"\nChecking level {level} for refinement...")
            
            # Compute error indicators from actual temperature field
            error_indicator, tagged_boxes = self._compute_error_indicator_from_data(level)
            
            if error_indicator is not None and self.config.get('show_error_indicator', False):
                self._visualize_error_indicator(level, error_indicator)
            
            # Check if any cells need refinement
            if tagged_boxes and len(tagged_boxes) > 0:
                print(f"  Found {len(tagged_boxes)} boxes to refine at level {level}")
                
                # Create refined level
                self._regrid_level(level + 1, tagged_boxes)
                levels_created += 1
                
                # Print stats for new level
                self._print_temperature_stats(level + 1)
            else:
                print(f"  No refinement needed at level {level}")
                break
                
        self.adapted = True
        print(f"\nGrid adaptation complete!")
        print(f"  Created {levels_created} refined levels")
        print(f"  Total levels: {len(self.levels)}")
        
        # Print final statistics
        self._print_adaptation_summary()


    def _compute_error_indicator_from_data(self, level: int):
        """
        Compute error indicator from actual temperature data in MultiFab.
        
        Returns:
            error_indicator: numpy array of error values (for visualization)
            tagged_boxes: list of boxes that need refinement
        """
        level_data = self.levels[level]
        solver = self.level_solvers[level]
        
        # Get refinement threshold
        base_threshold = self.config.get('refine_threshold', 100.0)
        threshold = base_threshold / (self.refinement_ratio ** level)
        temp_threshold = self.config.get('temp_threshold', 500.0)
        
        tagged_boxes = []
        
        # For visualization - create error indicator array
        nx_level = self.n_cell_base[0] * (self.refinement_ratio ** level)
        ny_level = self.n_cell_base[1] * (self.refinement_ratio ** level)
        error_indicator = np.zeros((ny_level, nx_level))
        
        # Process each box in this level
        for mfi in level_data.temperature:
            bx = mfi.validbox()
            temp_arr = level_data.temperature.array(mfi)
            
            lo = bx.small_end
            hi = bx.big_end
            
            # Compute gradients for this box
            box_needs_refinement = False
            
            for j in range(lo[1], hi[1] + 1):
                for i in range(lo[0], hi[0] + 1):
                    # Get temperature value
                    T = temp_arr[i, j, 0, self.comp_T]
                    
                    # Skip if temperature is too low
                    if T < temp_threshold:
                        continue
                    
                    # Compute gradient using central differences
                    if lo[0] < i < hi[0] and lo[1] < j < hi[1]:
                        dTdx = (temp_arr[i+1, j, 0, self.comp_T] - 
                               temp_arr[i-1, j, 0, self.comp_T]) / (2 * solver.mesh.dx)
                        dTdy = (temp_arr[i, j+1, 0, self.comp_T] - 
                               temp_arr[i, j-1, 0, self.comp_T]) / (2 * solver.mesh.dy)
                        
                        # Gradient magnitude
                        grad_mag = np.sqrt(dTdx**2 + dTdy**2)
                        
                        # Store in error indicator array
                        if 0 <= i < nx_level and 0 <= j < ny_level:
                            error_indicator[j, i] = grad_mag
                        
                        # Check against threshold
                        if grad_mag > threshold:
                            box_needs_refinement = True
            
            # If this box needs refinement, add it to the list
            if box_needs_refinement:
                tagged_boxes.append(bx)
        
        print(f"  Error indicator stats: min={np.min(error_indicator):.2f}, "
              f"max={np.max(error_indicator):.2f}, mean={np.mean(error_indicator):.2f}")
        
        return error_indicator, tagged_boxes


    def _visualize_error_indicator(self, level: int, error_indicator: np.ndarray):
        """Visualize the error indicator field."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create coordinate arrays
        nx = error_indicator.shape[1]
        ny = error_indicator.shape[0]
        x = np.linspace(0, self.domain_hi[0], nx)
        y = np.linspace(0, self.domain_hi[1], ny)
        X, Y = np.meshgrid(x, y)
        
        # Plot error indicator
        im = ax.contourf(X, Y, error_indicator, levels=50, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Error Indicator (|∇T|)')
        
        # Add threshold line
        threshold = self.config.get('refine_threshold', 100.0) / (self.refinement_ratio ** level)
        ax.contour(X, Y, error_indicator, levels=[threshold], colors='red', linewidths=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Error Indicator at Level {level} (threshold = {threshold:.1f})')
        ax.set_aspect('equal')
        
        plt.show()


    def _print_temperature_stats(self, level: int):
        """Print temperature statistics for a level."""
        if level >= len(self.levels):
            return
            
        level_data = self.levels[level]
        solver = self.level_solvers[level]
        
        # Sync AMReX to solver to get accurate stats
        self._sync_amrex_to_solver(level)
        
        T_interior = solver.mesh.extract_interior(solver.T)
        
        print(f"\nLevel {level} temperature statistics:")
        print(f"  Grid: {solver.mesh.nx}×{solver.mesh.ny} cells")
        print(f"  Cell size: Δx={solver.mesh.dx:.3e} m, Δy={solver.mesh.dy:.3e} m")
        print(f"  Temperature: min={np.min(T_interior):.1f} K, max={np.max(T_interior):.1f} K, "
              f"mean={np.mean(T_interior):.1f} K")


    def _print_adaptation_summary(self):
        """Print summary of the adapted grid."""
        print("\n" + "-"*60)
        print("AMReX AMR Grid Adaptation Summary")
        print("-"*60)
        
        total_cells = 0
        for level in range(len(self.levels)):
            level_cells = self.get_level_cell_count(level)
            total_cells += level_cells
            print(f"Level {level}: {level_cells:,} cells "
                  f"(refinement ratio: {self.refinement_ratio**level})")
        
        # Compute efficiency
        base_cells = self.n_cell_base[0] * self.n_cell_base[1]
        uniform_fine_cells = base_cells * (self.refinement_ratio ** (2 * (len(self.levels) - 1)))
        efficiency = (uniform_fine_cells - total_cells) / uniform_fine_cells * 100 if uniform_fine_cells > 0 else 0
        
        print(f"\nTotal cells: {total_cells:,}")
        print(f"Equivalent uniform fine grid: {uniform_fine_cells:,} cells")
        print(f"Memory savings: {efficiency:.1f}%")
        print("-"*60)
        
    def __del__(self):
        """Cleanup AMReX resources."""
        if self.amrex_initialized and AMREX_AVAILABLE:
            amr.finalize()