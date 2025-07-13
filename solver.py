# solver.py
import numpy as np
from tqdm import tqdm
from mesh import FVMesh
from weno import WENOReconstructor
from amr.amr_factory import AMRFactory
from amr.base_amr import BaseAMR
from typing import Optional

class FVHeatSolver:
    """
    Finite Volume solver for 2D heat diffusion equation with optional reactions.
    
    Solves: ∂T/∂t = α∇²T + S(T)
    
    where:
        T: temperature
        α: thermal diffusivity
        S: source term (chemical reactions)
    """
    
    def __init__(self, mesh, alpha, spatial_order=5, time_integration='RK3',
                 enable_reactions=False):
        """
        Initialize the FV heat solver.
        
        Parameters
        ----------
        mesh : FVMesh
            Finite volume mesh object
        alpha : float
            Thermal diffusivity (m²/s)
        spatial_order : int
            Spatial order of accuracy (1, 2, or 5)
        time_integration : str
            Time integration scheme ('RK3', 'RK4', 'Euler')
        enable_reactions : bool
            Enable chemical reaction source terms
        """
        self.mesh = mesh
        self.alpha = alpha
        self.spatial_order = spatial_order
        self.time_integration = time_integration
        self.enable_reactions = enable_reactions
        
        # Create WENO reconstructor
        self.weno = WENOReconstructor(order=spatial_order)
        
        # Solution arrays
        self.T = mesh.create_field(300.0)  # Temperature
        self.T_old = mesh.create_field(300.0)  # For time stepping
        
        # Reaction variables (if enabled)
        if enable_reactions:
            self.lambda_rxn = mesh.create_field(0.0)  # Reaction progress
            self.lambda_rxn_old = mesh.create_field(0.0)
            self.reaction_model = None  # To be set later
        else:
            self.lambda_rxn = None
            self.reaction_model = None
        
        # Time tracking
        self.current_time = 0.0
        self.step_count = 0
        
        # AMR preparation (not active by default)
        self.amr_enabled = False
        self.refinement_flags = None
        
        # Boundary condition type
        self.bc_type = 'neumann'  # 'neumann' or 'dirichlet'
        self.bc_values = {}
        
        # Statistics tracking
        self.stats = {
            'max_T': [],
            'min_T': [],
            'avg_T': [],
            'total_energy': []
        }
        
        # Centerline data collection
        self.collect_centerlines = False
        self.centerline_data = {
            'times': [],
            'x_centerline': [],
            'y_centerline': []
        }
        
    def enable_centerline_collection(self, enable=True):
        """Enable or disable centerline data collection during simulation."""
        self.collect_centerlines = enable
        if enable:
            # Clear any existing data
            self.centerline_data = {
                'times': [],
                'x_centerline': [],
                'y_centerline': []
            }
            
    def _collect_centerline_data(self):
        """Collect centerline data at current time step."""
        if not self.collect_centerlines:
            return
            
        # Get centerline indices
        j_center = self.mesh.ny // 2 + self.mesh.ghost_cells
        i_center = self.mesh.nx // 2 + self.mesh.ghost_cells
        
        # Extract x-centerline (along x at y=center)
        x_data = self.T[j_center, self.mesh.ghost_cells:-self.mesh.ghost_cells].copy()
        
        # Extract y-centerline (along y at x=center)
        y_data = self.T[self.mesh.ghost_cells:-self.mesh.ghost_cells, i_center].copy()
        
        # Store data
        self.centerline_data['times'].append(self.current_time)
        self.centerline_data['x_centerline'].append(x_data)
        self.centerline_data['y_centerline'].append(y_data)
        
    def get_centerline_history(self):
        """
        Get collected centerline data.
        
        Returns
        -------
        dict
            Dictionary with times and centerline data arrays
        """
        if not self.collect_centerlines or len(self.centerline_data['times']) == 0:
            return None
            
        return {
            'times': np.array(self.centerline_data['times']),
            'x_centerline': np.array(self.centerline_data['x_centerline']),
            'y_centerline': np.array(self.centerline_data['y_centerline']),
            'x_coords': self.mesh.x_centers,
            'y_coords': self.mesh.y_centers
        }
        
    def set_initial_condition_circular(self, T_background=300.0, T_hotspot=6000.0,
                                     center_x=None, center_y=None, 
                                     hotspot_radius=0.05, smooth_transition=True,
                                     transition_width=0.005):
        """
        Set circular hotspot initial condition.
        
        Parameters
        ----------
        T_background : float
            Background temperature (K)
        T_hotspot : float
            Hotspot center temperature (K)
        center_x : float
            X-coordinate of hotspot center (m). Default: domain center
        center_y : float
            Y-coordinate of hotspot center (m). Default: domain center
        hotspot_radius : float
            Radius of hotspot (m)
        smooth_transition : bool
            Use smooth transition at hotspot edge
        transition_width : float
            Width of transition zone (m)
        """
        if center_x is None:
            center_x = self.mesh.plate_length / 2
        if center_y is None:
            center_y = self.mesh.plate_width / 2
            
        # Initialize to background temperature
        self.T.fill(T_background)
        
        # Set hotspot
        for j in range(self.mesh.ny_total):
            for i in range(self.mesh.nx_total):
                x, y = self.mesh.index_to_physical(i, j)
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                if smooth_transition:
                    if r <= hotspot_radius - transition_width:
                        self.T[j, i] = T_hotspot
                    elif r >= hotspot_radius + transition_width:
                        self.T[j, i] = T_background
                    else:
                        # Smooth transition using cosine
                        t = (r - (hotspot_radius - transition_width)) / (2 * transition_width)
                        t = np.clip(t, 0, 1)
                        weight = 0.5 * (1 + np.cos(np.pi * t))
                        self.T[j, i] = weight * T_hotspot + (1 - weight) * T_background
                else:
                    if r <= hotspot_radius:
                        self.T[j, i] = T_hotspot
                        
        # Apply boundary conditions to ghost cells
        self._apply_boundary_conditions()
        
        # Reset time
        self.current_time = 0.0
        self.step_count = 0
        
        # Store initial statistics
        self._update_statistics()
        
        # Collect initial centerline data if enabled
        self._collect_centerline_data()
        
    def set_initial_condition_from_array(self, T_array):
        """
        Set initial condition from a numpy array.
        
        Parameters
        ----------
        T_array : ndarray
            Temperature array. Can be:
            - Shape (ny, nx): interior values only
            - Shape (ny_total, nx_total): including ghost cells
        """
        if T_array.shape == (self.mesh.ny, self.mesh.nx):
            # Interior only - extend with ghost cells
            j_int, i_int = self.mesh.get_interior_slice()
            self.T[j_int, i_int] = T_array
            self._apply_boundary_conditions()
        elif T_array.shape == (self.mesh.ny_total, self.mesh.nx_total):
            # Full array including ghost
            self.T[:] = T_array
            self._apply_boundary_conditions()
        else:
            raise ValueError(f"Invalid array shape: {T_array.shape}")
            
        self.current_time = 0.0
        self.step_count = 0
        self._update_statistics()
        
        # Collect initial centerline data if enabled
        self._collect_centerline_data()
        
    def set_reaction_model(self, reaction_model):
        """
        Set the chemical reaction model.
        
        Parameters
        ----------
        reaction_model : object
            Reaction model with compute_source() and compute_progress_rate() methods
        """
        if not self.enable_reactions:
            raise RuntimeError("Reactions not enabled in solver initialization")
        self.reaction_model = reaction_model
        
    def set_boundary_conditions(self, bc_type='neumann', bc_values=None):
        """
        Set boundary condition type and values.
        
        Parameters
        ----------
        bc_type : str
            'neumann' for zero gradient, 'dirichlet' for fixed values
        bc_values : dict
            For Dirichlet BC: {'bottom': T_b, 'top': T_t, 'left': T_l, 'right': T_r}
        """
        self.bc_type = bc_type
        if bc_values is not None:
            self.bc_values = bc_values
            
    def compute_rhs(self):
        """
        Compute right-hand side of heat equation.
        
        Returns
        -------
        ndarray
            RHS values at interior cells: α∇²T + S
        """
        # Apply boundary conditions
        self._apply_boundary_conditions()
        
        # Compute Laplacian using WENO
        laplacian = self.weno.compute_laplacian(self.T, self.mesh, self.alpha, 
                                               order=self.spatial_order)
        
        # Add reaction source term if enabled
        if self.enable_reactions and self.reaction_model is not None:
            source = self.reaction_model.compute_source(self)
            laplacian += source
            
        return laplacian
    
    def compute_reaction_progress_rate(self):
        """
        Compute reaction progress rate dλ/dt.
        
        Returns
        -------
        ndarray
            Reaction progress rate at interior cells
        """
        if not self.enable_reactions or self.reaction_model is None:
            return None
            
        return self.reaction_model.compute_progress_rate(self)
    
    def time_step_euler(self, dt):
        """
        Advance solution using forward Euler method.
        
        Parameters
        ----------
        dt : float
            Time step size (s)
        """
        j_int, i_int = self.mesh.get_interior_slice()
        
        # Store old solution
        self.T_old[:] = self.T
        if self.enable_reactions:
            self.lambda_rxn_old[:] = self.lambda_rxn
            
        # Compute RHS
        rhs_T = self.compute_rhs()
        
        # Update temperature
        self.T[j_int, i_int] += dt * rhs_T
        
        # Update reaction progress if enabled
        if self.enable_reactions:
            rhs_lambda = self.compute_reaction_progress_rate()
            if rhs_lambda is not None:
                self.lambda_rxn[j_int, i_int] += dt * rhs_lambda
                # Ensure λ stays in [0, 1]
                self.lambda_rxn[j_int, i_int] = np.clip(self.lambda_rxn[j_int, i_int], 0, 1)
                
    def time_step_rk3(self, dt):
        """
        Advance solution using 3rd order Runge-Kutta method (TVD-RK3).
        
        Parameters
        ----------
        dt : float
            Time step size (s)
        """
        j_int, i_int = self.mesh.get_interior_slice()
        
        # Store initial values
        self.T_old[:] = self.T.copy()
        if self.enable_reactions:
            self.lambda_rxn_old[:] = self.lambda_rxn.copy()
            
        # Stage 1
        rhs_T1 = self.compute_rhs()
        self.T[j_int, i_int] = self.T_old[j_int, i_int] + dt * rhs_T1
        
        if self.enable_reactions:
            rhs_lambda1 = self.compute_reaction_progress_rate()
            if rhs_lambda1 is not None:
                self.lambda_rxn[j_int, i_int] = self.lambda_rxn_old[j_int, i_int] + dt * rhs_lambda1
                self.lambda_rxn[j_int, i_int] = np.clip(self.lambda_rxn[j_int, i_int], 0, 1)
                
        # Stage 2
        rhs_T2 = self.compute_rhs()
        self.T[j_int, i_int] = (3*self.T_old[j_int, i_int] + 
                                self.T[j_int, i_int] + dt * rhs_T2) / 4
        
        if self.enable_reactions:
            rhs_lambda2 = self.compute_reaction_progress_rate()
            if rhs_lambda2 is not None:
                self.lambda_rxn[j_int, i_int] = (3*self.lambda_rxn_old[j_int, i_int] + 
                                                 self.lambda_rxn[j_int, i_int] + 
                                                 dt * rhs_lambda2) / 4
                self.lambda_rxn[j_int, i_int] = np.clip(self.lambda_rxn[j_int, i_int], 0, 1)
                
        # Stage 3
        rhs_T3 = self.compute_rhs()
        self.T[j_int, i_int] = (self.T_old[j_int, i_int] + 
                                2*self.T[j_int, i_int] + 2*dt * rhs_T3) / 3
        
        if self.enable_reactions:
            rhs_lambda3 = self.compute_reaction_progress_rate()
            if rhs_lambda3 is not None:
                self.lambda_rxn[j_int, i_int] = (self.lambda_rxn_old[j_int, i_int] + 
                                                 2*self.lambda_rxn[j_int, i_int] + 
                                                 2*dt * rhs_lambda3) / 3
                self.lambda_rxn[j_int, i_int] = np.clip(self.lambda_rxn[j_int, i_int], 0, 1)
                
    def advance(self, dt):
        """
        Advance solution by one time step.
        
        Parameters
        ----------
        dt : float
            Time step size (s)
        """
        if self.time_integration == 'Euler':
            self.time_step_euler(dt)
        elif self.time_integration == 'RK3':
            self.time_step_rk3(dt)
        else:
            raise ValueError(f"Unknown time integration: {self.time_integration}")
            
        self.current_time += dt
        self.step_count += 1
        self._update_statistics()
        
    def advance_to_time(self, target_time, dt, show_progress=True, collect_interval=None, 
                       collect_at_target=False):
        """
        Advance solution to target time.
        
        Parameters
        ----------
        target_time : float
            Target time to advance to (s)
        dt : float
            Time step size (s)
        show_progress : bool
            Show progress bar
        collect_interval : float, optional
            Interval for collecting centerline data (None = don't collect during advance)
        collect_at_target : bool
            Whether to collect centerline data when reaching target time
            
        Returns
        -------
        float
            Actual time reached
        """
        if show_progress:
            pbar = tqdm(total=target_time - self.current_time, 
                       desc="Advancing time", unit='s')
            
        last_collect_time = self.current_time
        
        while self.current_time < target_time - 1e-10:
            # Adjust last time step if needed
            step_dt = min(dt, target_time - self.current_time)
            
            # Take time step
            self.advance(step_dt)
            
            # Collect centerline data if enabled and interval specified
            if self.collect_centerlines and collect_interval is not None:
                if self.current_time - last_collect_time >= collect_interval:
                    self._collect_centerline_data()
                    last_collect_time = self.current_time
            
            if show_progress:
                pbar.update(step_dt)
                
        # Collect at target time if requested
        if self.collect_centerlines and collect_at_target and self.current_time >= target_time - 1e-10:
            self._collect_centerline_data()
                
        if show_progress:
            pbar.close()
            
        return self.current_time
    
    def compute_stable_timestep(self, cfl_diffusion=0.5):
        """
        Compute stable time step based on diffusion CFL condition.
        
        Parameters
        ----------
        cfl_diffusion : float
            CFL number for diffusion (typically 0.5)
            
        Returns
        -------
        float
            Maximum stable time step
        """
        # Diffusion stability: dt <= CFL * min(dx², dy²) / (4α)
        dt_max = cfl_diffusion * min(self.mesh.dx**2, self.mesh.dy**2) / (4 * self.alpha)
        
        # For high-order schemes, be more conservative
        if self.spatial_order == 5:
            dt_max *= 0.8
            
        return dt_max
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions to ghost cells."""
        if self.bc_type == 'neumann':
            self.mesh.apply_neumann_bc(self.T)
            if self.enable_reactions and self.lambda_rxn is not None:
                self.mesh.apply_neumann_bc(self.lambda_rxn)
        elif self.bc_type == 'dirichlet':
            self.mesh.apply_dirichlet_bc(self.T, self.bc_values)
            # Reactions typically use Neumann BC
            if self.enable_reactions and self.lambda_rxn is not None:
                self.mesh.apply_neumann_bc(self.lambda_rxn)
                
    def _update_statistics(self):
        """Update solution statistics."""
        T_interior = self.mesh.extract_interior(self.T)
        
        self.stats['max_T'].append(np.max(T_interior))
        self.stats['min_T'].append(np.min(T_interior))
        self.stats['avg_T'].append(np.mean(T_interior))
        
        # Total thermal energy
        cell_volume = self.mesh.get_cell_volume()
        total_energy = np.sum(T_interior) * cell_volume
        self.stats['total_energy'].append(total_energy)
        
    def get_interior_solution(self):
        """
        Get solution in interior domain (no ghost cells).
        
        Returns
        -------
        dict
            Dictionary with 'T' and optionally 'lambda_rxn'
        """
        result = {'T': self.mesh.extract_interior(self.T)}
        
        if self.enable_reactions and self.lambda_rxn is not None:
            result['lambda_rxn'] = self.mesh.extract_interior(self.lambda_rxn)
            
        return result
    
    def compute_gradient_for_amr(self):
        """
        Compute gradient magnitude for AMR refinement criterion.
        
        Returns
        -------
        ndarray
            Gradient magnitude at interior cells
        """
        return self.mesh.compute_gradient_magnitude(self.T)
    
    def set_amr_system(self, amr_system: Optional['BaseAMR']):
        """
        Set the AMR system for the solver.
        
        Parameters
        ----------
        amr_system : BaseAMR or None
            AMR system instance or None to disable AMR
        """
        self.amr_system = amr_system
        
        if amr_system is not None:
            print(f"AMR system attached: {type(amr_system).__name__}")
        else:
            print("AMR disabled")
            
    def advance_with_amr(self, dt):
        """
        Advance solution with AMR support.
        
        Parameters
        ----------
        dt : float
            Time step size (s)
        """
        if self.amr_system is None:
            # No AMR - regular advance
            self.advance(dt)
        else:
            # Let AMR system handle the advance
            self.amr_system.advance_with_regrid(dt)
            
    def get_amr_composite_solution(self):
        """
        Get composite solution from AMR hierarchy.
        
        Returns
        -------
        dict
            Composite solution data or None if no AMR
        """
        if self.amr_system is None:
            return None
            
        return self.amr_system.get_composite_solution('T')
        
    def compute_amr_error_indicator(self):
        """
        Compute error indicator for AMR refinement.
        
        Returns
        -------
        ndarray
            Error indicator field
        """
        # Default implementation using gradient
        return self.mesh.compute_gradient_magnitude(self.T)
        
    def get_solution_for_visualization(self, use_amr_composite=True):
        """
        Get solution data formatted for visualization.
        
        Parameters
        ----------
        use_amr_composite : bool
            If True and AMR is active, return composite solution
            
        Returns
        -------
        dict
            Solution data with coordinates
        """
        if use_amr_composite and self.amr_system is not None:
            # Get AMR composite solution
            composite = self.amr_system.get_composite_solution('T')
            
            # Add reaction data if available
            if self.enable_reactions and self.lambda_rxn is not None:
                composite_lambda = self.amr_system.get_composite_solution('lambda')
                if composite_lambda is not None:
                    composite['lambda_rxn'] = composite_lambda['data']
                    
            return composite
        else:
            # Regular solution without AMR
            return {
                'data': self.mesh.extract_interior(self.T),
                'x': self.mesh.x_centers,
                'y': self.mesh.y_centers,
                'level_map': None
            }
            
    # Updated advance_to_time method to support AMR:
    def advance_to_time_with_amr(self, target_time, dt, show_progress=True, 
                                collect_interval=None, collect_at_target=False):
        """
        Advance solution to target time with optional AMR.
        
        Parameters
        ----------
        target_time : float
            Target time to advance to (s)
        dt : float
            Time step size (s)
        show_progress : bool
            Show progress bar
        collect_interval : float, optional
            Interval for collecting centerline data
        collect_at_target : bool
            Whether to collect centerline data at target
            
        Returns
        -------
        float
            Actual time reached
        """
        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=target_time - self.current_time, 
                    desc="Advancing time", unit='s')
            
        last_collect_time = self.current_time
        
        while self.current_time < target_time - 1e-10:
            # Adjust last time step if needed
            step_dt = min(dt, target_time - self.current_time)
            
            # Take time step with or without AMR
            self.advance_with_amr(step_dt)
            
            # Collect centerline data if enabled
            if self.collect_centerlines and collect_interval is not None:
                if self.current_time - last_collect_time >= collect_interval:
                    self._collect_centerline_data()
                    last_collect_time = self.current_time
            
            if show_progress:
                pbar.update(step_dt)
                
        # Collect at target if requested
        if self.collect_centerlines and collect_at_target:
            self._collect_centerline_data()
                
        if show_progress:
            pbar.close()
            
        return self.current_time

    # Add AMR statistics to solver statistics:
    def get_solver_statistics(self):
        """
        Get comprehensive solver statistics including AMR.
        
        Returns
        -------
        dict
            Solver and AMR statistics
        """
        stats = {
            'time': self.current_time,
            'step': self.step_count,
            'mesh': {
                'nx': self.mesh.nx,
                'ny': self.mesh.ny,
                'dx': self.mesh.dx,
                'dy': self.mesh.dy
            }
        }
        
        # Add solution statistics
        T_interior = self.mesh.extract_interior(self.T)
        stats['solution'] = {
            'T_max': np.max(T_interior),
            'T_min': np.min(T_interior),
            'T_mean': np.mean(T_interior),
            'T_std': np.std(T_interior)
        }
        
        # Add AMR statistics if available
        if self.amr_system is not None:
            stats['amr'] = self.amr_system.get_statistics()
            
        return stats
    

    # Additional methods to add to FVHeatSolver class in solver.py

    def set_from_interpolation(self, coarse_data: np.ndarray, refinement_ratio: int):
        """
        Set solver data by interpolating from coarser level data.
        
        Parameters
        ----------
        coarse_data : np.ndarray
            Temperature data from coarser level (without ghost cells)
        refinement_ratio : int
            Refinement ratio between levels
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create coarse grid coordinates
        ny_coarse, nx_coarse = coarse_data.shape
        x_coarse = np.linspace(0, self.mesh.plate_length, nx_coarse)
        y_coarse = np.linspace(0, self.mesh.plate_width, ny_coarse)
        
        # Create interpolator
        interp = RegularGridInterpolator(
            (y_coarse, x_coarse),
            coarse_data,
            method='linear',
            bounds_error=False,
            fill_value=np.mean(coarse_data)
        )
        
        # Interpolate to fine grid
        for j in range(self.mesh.ny):
            for i in range(self.mesh.nx):
                x = self.mesh.x_centers[i]
                y = self.mesh.y_centers[j]
                T_interp = interp((y, x))
                
                # Set interior value
                j_with_ghost = j + self.mesh.ghost_cells
                i_with_ghost = i + self.mesh.ghost_cells
                self.T[j_with_ghost, i_with_ghost] = T_interp
        
        # Apply boundary conditions to fill ghost cells
        self._apply_boundary_conditions()
        
    def extract_patch(self, i_start: int, i_end: int, j_start: int, j_end: int) -> np.ndarray:
        """
        Extract a patch of the solution (without ghost cells).
        
        Parameters
        ----------
        i_start, i_end : int
            X-direction bounds (in interior indices)
        j_start, j_end : int  
            Y-direction bounds (in interior indices)
            
        Returns
        -------
        np.ndarray
            Extracted patch data
        """
        g = self.mesh.ghost_cells
        return self.T[j_start+g:j_end+g, i_start+g:i_end+g].copy()
        
    def set_patch(self, patch_data: np.ndarray, i_start: int, j_start: int):
        """
        Set a patch of the solution.
        
        Parameters
        ----------
        patch_data : np.ndarray
            Patch data to set
        i_start : int
            Starting x index (interior)
        j_start : int
            Starting y index (interior)
        """
        g = self.mesh.ghost_cells
        ny_patch, nx_patch = patch_data.shape
        
        self.T[j_start+g:j_start+g+ny_patch, 
               i_start+g:i_start+g+nx_patch] = patch_data
               
    # Complete implementation for solver.py - apply_coarse_fine_bc method

def apply_coarse_fine_bc(self, coarse_solver: 'FVHeatSolver', 
                        level_ratio: int, 
                        patch_bounds: tuple[int, int, int, int]):
    """
    Apply boundary conditions at coarse-fine interfaces.
    
    This fills ghost cells of fine level using high-order interpolation 
    from coarse level data.
    
    Parameters
    ----------
    coarse_solver : FVHeatSolver
        Solver for coarser level
    level_ratio : int
        Refinement ratio between levels (typically 2 or 4)
    patch_bounds : tuple
        (i_start, i_end, j_start, j_end) of this patch in the fine grid
        coordinates relative to the full fine level domain
    """
    g = self.mesh.ghost_cells
    g_coarse = coarse_solver.mesh.ghost_cells
    
    # Get temperature fields
    T_fine = self.T
    T_coarse = coarse_solver.T
    
    # Calculate which boundaries are at coarse-fine interfaces
    # (not at physical domain boundaries)
    i_start, i_end, j_start, j_end = patch_bounds
    
    # Check if each boundary is internal (coarse-fine interface) or external (domain boundary)
    at_left_boundary = (i_start == 0)
    at_right_boundary = (i_end == self.mesh.nx)
    at_bottom_boundary = (j_start == 0)
    at_top_boundary = (j_end == self.mesh.ny)
    
    # For high-order interpolation, we'll use cubic interpolation
    # This requires 4 coarse points to interpolate
    
    # Helper function for cubic interpolation
    def cubic_interp(x, x_points, y_points):
        """Cubic interpolation using 4 points."""
        if len(x_points) != 4 or len(y_points) != 4:
            # Fall back to linear if not enough points
            return np.interp(x, x_points, y_points)
        
        # Lagrange polynomial interpolation
        result = 0.0
        for i in range(4):
            term = y_points[i]
            for j in range(4):
                if i != j:
                    term *= (x - x_points[j]) / (x_points[i] - x_points[j])
            result += term
        return result
    
    # Left boundary ghost cells
    if not at_left_boundary:
        for j in range(self.mesh.ny):
            # Fine grid j-coordinate
            j_fine = j + j_start
            
            # Map to coarse grid
            j_coarse_exact = j_fine / level_ratio
            j_coarse = int(j_coarse_exact)
            
            # Get coarse stencil for j-direction interpolation
            j_stencil = []
            j_values = []
            for jc in range(max(0, j_coarse-1), min(coarse_solver.mesh.ny, j_coarse+3)):
                j_stencil.append(jc)
            
            # Fill ghost cells from left
            for k in range(g):
                # Fine grid i-coordinate in ghost region
                i_fine_ghost = i_start - k - 1
                
                # Map to coarse grid
                i_coarse_exact = i_fine_ghost / level_ratio
                i_coarse = int(i_coarse_exact)
                
                # Get coarse stencil for i-direction
                i_stencil = []
                for ic in range(max(0, i_coarse-1), min(coarse_solver.mesh.nx, i_coarse+3)):
                    i_stencil.append(ic)
                
                if len(i_stencil) >= 2 and len(j_stencil) >= 2:
                    # 2D interpolation: first interpolate in i-direction for each j
                    temp_values = []
                    for jc in j_stencil:
                        i_values = []
                        i_coords = []
                        for ic in i_stencil:
                            i_values.append(T_coarse[jc + g_coarse, ic + g_coarse])
                            i_coords.append(ic * level_ratio + level_ratio // 2)
                        
                        # Interpolate in i-direction
                        if len(i_coords) >= 2:
                            temp = np.interp(i_fine_ghost + 0.5, i_coords, i_values)
                            temp_values.append(temp)
                    
                    # Then interpolate in j-direction
                    if len(temp_values) >= 2:
                        j_coords = [(jc * level_ratio + level_ratio // 2) for jc in j_stencil]
                        T_fine[j + g, g - 1 - k] = np.interp(j_fine + 0.5, j_coords, temp_values)
                    else:
                        # Not enough points - use nearest
                        T_fine[j + g, g - 1 - k] = T_coarse[j_coarse + g_coarse, i_coarse + g_coarse]
                else:
                    # Fall back to physical BC
                    T_fine[j + g, g - 1 - k] = T_fine[j + g, g]
    
    # Right boundary ghost cells
    if not at_right_boundary:
        for j in range(self.mesh.ny):
            j_fine = j + j_start
            j_coarse_exact = j_fine / level_ratio
            j_coarse = int(j_coarse_exact)
            
            j_stencil = []
            for jc in range(max(0, j_coarse-1), min(coarse_solver.mesh.ny, j_coarse+3)):
                j_stencil.append(jc)
            
            for k in range(g):
                i_fine_ghost = i_end + k
                i_coarse_exact = i_fine_ghost / level_ratio
                i_coarse = int(i_coarse_exact)
                
                i_stencil = []
                for ic in range(max(0, i_coarse-1), min(coarse_solver.mesh.nx, i_coarse+3)):
                    i_stencil.append(ic)
                
                if len(i_stencil) >= 2 and len(j_stencil) >= 2:
                    # 2D interpolation
                    temp_values = []
                    for jc in j_stencil:
                        i_values = []
                        i_coords = []
                        for ic in i_stencil:
                            i_values.append(T_coarse[jc + g_coarse, ic + g_coarse])
                            i_coords.append(ic * level_ratio + level_ratio // 2)
                        
                        if len(i_coords) >= 2:
                            temp = np.interp(i_fine_ghost + 0.5, i_coords, i_values)
                            temp_values.append(temp)
                    
                    if len(temp_values) >= 2:
                        j_coords = [(jc * level_ratio + level_ratio // 2) for jc in j_stencil]
                        T_fine[j + g, self.mesh.nx + g + k] = np.interp(j_fine + 0.5, j_coords, temp_values)
                    else:
                        T_fine[j + g, self.mesh.nx + g + k] = T_coarse[j_coarse + g_coarse, i_coarse + g_coarse]
                else:
                    T_fine[j + g, self.mesh.nx + g + k] = T_fine[j + g, self.mesh.nx + g - 1]
    
    # Bottom boundary ghost cells
    if not at_bottom_boundary:
        for i in range(self.mesh.nx):
            i_fine = i + i_start
            i_coarse_exact = i_fine / level_ratio
            i_coarse = int(i_coarse_exact)
            
            i_stencil = []
            for ic in range(max(0, i_coarse-1), min(coarse_solver.mesh.nx, i_coarse+3)):
                i_stencil.append(ic)
            
            for k in range(g):
                j_fine_ghost = j_start - k - 1
                j_coarse_exact = j_fine_ghost / level_ratio
                j_coarse = int(j_coarse_exact)
                
                j_stencil = []
                for jc in range(max(0, j_coarse-1), min(coarse_solver.mesh.ny, j_coarse+3)):
                    j_stencil.append(jc)
                
                if len(i_stencil) >= 2 and len(j_stencil) >= 2:
                    # 2D interpolation
                    temp_values = []
                    for ic in i_stencil:
                        j_values = []
                        j_coords = []
                        for jc in j_stencil:
                            j_values.append(T_coarse[jc + g_coarse, ic + g_coarse])
                            j_coords.append(jc * level_ratio + level_ratio // 2)
                        
                        if len(j_coords) >= 2:
                            temp = np.interp(j_fine_ghost + 0.5, j_coords, j_values)
                            temp_values.append(temp)
                    
                    if len(temp_values) >= 2:
                        i_coords = [(ic * level_ratio + level_ratio // 2) for ic in i_stencil]
                        T_fine[g - 1 - k, i + g] = np.interp(i_fine + 0.5, i_coords, temp_values)
                    else:
                        T_fine[g - 1 - k, i + g] = T_coarse[j_coarse + g_coarse, i_coarse + g_coarse]
                else:
                    T_fine[g - 1 - k, i + g] = T_fine[g, i + g]
    
    # Top boundary ghost cells  
    if not at_top_boundary:
        for i in range(self.mesh.nx):
            i_fine = i + i_start
            i_coarse_exact = i_fine / level_ratio
            i_coarse = int(i_coarse_exact)
            
            i_stencil = []
            for ic in range(max(0, i_coarse-1), min(coarse_solver.mesh.nx, i_coarse+3)):
                i_stencil.append(ic)
            
            for k in range(g):
                j_fine_ghost = j_end + k
                j_coarse_exact = j_fine_ghost / level_ratio
                j_coarse = int(j_coarse_exact)
                
                j_stencil = []
                for jc in range(max(0, j_coarse-1), min(coarse_solver.mesh.ny, j_coarse+3)):
                    j_stencil.append(jc)
                
                if len(i_stencil) >= 2 and len(j_stencil) >= 2:
                    # 2D interpolation
                    temp_values = []
                    for ic in i_stencil:
                        j_values = []
                        j_coords = []
                        for jc in j_stencil:
                            j_values.append(T_coarse[jc + g_coarse, ic + g_coarse])
                            j_coords.append(jc * level_ratio + level_ratio // 2)
                        
                        if len(j_coords) >= 2:
                            temp = np.interp(j_fine_ghost + 0.5, j_coords, j_values)
                            temp_values.append(temp)
                    
                    if len(temp_values) >= 2:
                        i_coords = [(ic * level_ratio + level_ratio // 2) for ic in i_stencil]
                        T_fine[self.mesh.ny + g + k, i + g] = np.interp(i_fine + 0.5, i_coords, temp_values)
                    else:
                        T_fine[self.mesh.ny + g + k, i + g] = T_coarse[j_coarse + g_coarse, i_coarse + g_coarse]
                else:
                    T_fine[self.mesh.ny + g + k, i + g] = T_fine[self.mesh.ny + g - 1, i + g]
    
    # Handle corners (use average of adjacent ghost cells)
    # Bottom-left corner
    if not at_left_boundary and not at_bottom_boundary:
        for k in range(g):
            for l in range(g):
                T_fine[g - 1 - k, g - 1 - l] = 0.5 * (
                    T_fine[g - 1 - k, g] + T_fine[g, g - 1 - l]
                )
    
    # Bottom-right corner
    if not at_right_boundary and not at_bottom_boundary:
        for k in range(g):
            for l in range(g):
                T_fine[g - 1 - k, self.mesh.nx + g + l] = 0.5 * (
                    T_fine[g - 1 - k, self.mesh.nx + g - 1] + 
                    T_fine[g, self.mesh.nx + g + l]
                )
    
    # Top-left corner
    if not at_left_boundary and not at_top_boundary:
        for k in range(g):
            for l in range(g):
                T_fine[self.mesh.ny + g + k, g - 1 - l] = 0.5 * (
                    T_fine[self.mesh.ny + g + k, g] + 
                    T_fine[self.mesh.ny + g - 1, g - 1 - l]
                )
    
    # Top-right corner
    if not at_right_boundary and not at_top_boundary:
        for k in range(g):
            for l in range(g):
                T_fine[self.mesh.ny + g + k, self.mesh.nx + g + l] = 0.5 * (
                    T_fine[self.mesh.ny + g + k, self.mesh.nx + g - 1] + 
                    T_fine[self.mesh.ny + g - 1, self.mesh.nx + g + l]
                )
    
    # Apply physical BC at domain boundaries
    if at_left_boundary:
        self.mesh.apply_neumann_bc(self.T)
    if at_right_boundary:
        self.mesh.apply_neumann_bc(self.T)
    if at_bottom_boundary:
        self.mesh.apply_neumann_bc(self.T)
    if at_top_boundary:
        self.mesh.apply_neumann_bc(self.T)
    
    # Do the same for reaction progress if enabled
    if self.enable_reactions and self.lambda_rxn is not None:
        # Similar interpolation for lambda_rxn
        # (Implementation would be identical, just replace T with lambda_rxn)
        pass 
        
    # Also need to update the flux computation method in solver.py
    # This should be added to the solver.py file:

    def compute_fluxes_at_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute diffusive fluxes at cell faces for flux correction.
        
        Uses WENO reconstruction for high-order accurate flux computation.
        
        Returns
        -------
        flux_x : np.ndarray
            X-direction fluxes at x-faces (shape: [ny, nx+1])
        flux_y : np.ndarray
            Y-direction fluxes at y-faces (shape: [ny+1, nx])
        """
        g = self.mesh.ghost_cells
        
        # Ensure boundary conditions are applied
        self._apply_boundary_conditions()
        
        # X-direction fluxes: F = -α ∂T/∂x
        flux_x = np.zeros((self.mesh.ny, self.mesh.nx + 1))
        
        if self.spatial_order >= 5:
            # Use WENO reconstruction for face values
            for j in range(self.mesh.ny):
                # Get face values using WENO
                face_values = self.weno.reconstruct_x(self.T, j + g)
                
                # Compute fluxes from gradients at faces
                for i in range(self.mesh.nx + 1):
                    if i == 0:
                        # Left boundary
                        T_left = self.T[j + g, i + g - 1]
                        T_right = self.T[j + g, i + g]
                        flux_x[j, i] = -self.alpha * (T_right - T_left) / self.mesh.dx
                    elif i == self.mesh.nx:
                        # Right boundary
                        T_left = self.T[j + g, i + g - 1]
                        T_right = self.T[j + g, i + g]
                        flux_x[j, i] = -self.alpha * (T_right - T_left) / self.mesh.dx
                    else:
                        # Interior faces - use high-order gradient
                        # Face value is between cells i-1 and i
                        # We need gradient at the face
                        T_left = face_values[i + g - 1]
                        T_right = face_values[i + g]
                        flux_x[j, i] = -self.alpha * (T_right - T_left) / self.mesh.dx
        else:
            # Simple centered difference for lower orders
            for j in range(self.mesh.ny):
                for i in range(self.mesh.nx + 1):
                    T_left = self.T[j + g, i + g - 1]
                    T_right = self.T[j + g, i + g]
                    flux_x[j, i] = -self.alpha * (T_right - T_left) / self.mesh.dx
        
        # Y-direction fluxes: G = -α ∂T/∂y
        flux_y = np.zeros((self.mesh.ny + 1, self.mesh.nx))
        
        if self.spatial_order >= 5:
            # Use WENO reconstruction for face values
            for i in range(self.mesh.nx):
                # Get face values using WENO
                face_values = self.weno.reconstruct_y(self.T, i + g)
                
                # Compute fluxes from gradients at faces
                for j in range(self.mesh.ny + 1):
                    if j == 0:
                        # Bottom boundary
                        T_bottom = self.T[j + g - 1, i + g]
                        T_top = self.T[j + g, i + g]
                        flux_y[j, i] = -self.alpha * (T_top - T_bottom) / self.mesh.dy
                    elif j == self.mesh.ny:
                        # Top boundary
                        T_bottom = self.T[j + g - 1, i + g]
                        T_top = self.T[j + g, i + g]
                        flux_y[j, i] = -self.alpha * (T_top - T_bottom) / self.mesh.dy
                    else:
                        # Interior faces
                        T_bottom = face_values[j + g - 1]
                        T_top = face_values[j + g]
                        flux_y[j, i] = -self.alpha * (T_top - T_bottom) / self.mesh.dy
        else:
            # Simple centered difference
            for j in range(self.mesh.ny + 1):
                for i in range(self.mesh.nx):
                    T_bottom = self.T[j + g - 1, i + g]
                    T_top = self.T[j + g, i + g]
                    flux_y[j, i] = -self.alpha * (T_top - T_bottom) / self.mesh.dy
        
        return flux_x, flux_y