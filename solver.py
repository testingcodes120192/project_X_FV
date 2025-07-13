# solver.py
import numpy as np
from tqdm import tqdm
from mesh import FVMesh
from weno import WENOReconstructor

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