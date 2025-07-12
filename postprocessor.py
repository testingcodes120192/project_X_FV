# postprocessor.py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

class FVPostProcessor:
    """
    Post-processing utilities for finite volume solutions.
    
    Handles interpolation to plotting grids, statistics computation,
    solution export, and preparation for visualization.
    """
    
    def __init__(self, mesh):
        """
        Initialize post-processor.
        
        Parameters
        ----------
        mesh : FVMesh
            Finite volume mesh object
        """
        self.mesh = mesh
        
    def interpolate_to_grid(self, field, nx_plot, ny_plot, smooth=False, sigma=0.5):
        """
        Interpolate cell-centered data to a uniform plotting grid.
        
        Parameters
        ----------
        field : ndarray
            Field data with ghost cells
        nx_plot : int
            Number of plotting points in x-direction
        ny_plot : int
            Number of plotting points in y-direction
        smooth : bool
            Apply Gaussian smoothing to reduce cell artifacts
        sigma : float
            Smoothing parameter (in grid points)
            
        Returns
        -------
        x_plot : ndarray
            1D array of x-coordinates
        y_plot : ndarray
            1D array of y-coordinates
        field_plot : ndarray
            2D array of interpolated values
        """
        # Extract interior solution
        field_interior = self.mesh.extract_interior(field)
        
        # Create plotting grid
        x_plot = np.linspace(0, self.mesh.plate_length, nx_plot)
        y_plot = np.linspace(0, self.mesh.plate_width, ny_plot)
        
        # Create interpolator (using cell centers)
        interp = RegularGridInterpolator(
            (self.mesh.y_centers, self.mesh.x_centers),
            field_interior,
            method='linear',
            bounds_error=False,
            fill_value=np.mean(field_interior)
        )
        
        # Create mesh grid for evaluation
        X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
        points = np.column_stack([Y_plot.ravel(), X_plot.ravel()])
        
        # Interpolate
        field_plot = interp(points).reshape(ny_plot, nx_plot)
        
        # Optional smoothing
        if smooth:
            field_plot = gaussian_filter(field_plot, sigma=sigma)
            
        return x_plot, y_plot, field_plot
    
    def get_solution_on_grid(self, solver, nx_plot, ny_plot, smooth=False):
        """
        Get complete solution interpolated to plotting grid.
        
        Parameters
        ----------
        solver : FVHeatSolver
            Solver object
        nx_plot : int
            Number of plotting points in x-direction
        ny_plot : int
            Number of plotting points in y-direction
        smooth : bool
            Apply smoothing
            
        Returns
        -------
        dict
            Dictionary with coordinates and field values
        """
        x, y, T = self.interpolate_to_grid(solver.T, nx_plot, ny_plot, smooth)
        
        result = {
            'x': x,
            'y': y,
            'T': T,
            'time': solver.current_time
        }
        
        if solver.enable_reactions and solver.lambda_rxn is not None:
            _, _, lambda_rxn = self.interpolate_to_grid(
                solver.lambda_rxn, nx_plot, ny_plot, smooth
            )
            result['lambda_rxn'] = lambda_rxn
            
        return result
    
    def compute_statistics(self, solver):
        """
        Compute comprehensive solution statistics.
        
        Parameters
        ----------
        solver : FVHeatSolver
            Solver object
            
        Returns
        -------
        dict
            Dictionary of statistics
        """
        # Extract interior solutions
        T_interior = self.mesh.extract_interior(solver.T)
        
        stats = {
            'time': solver.current_time,
            'step': solver.step_count,
            'T_max': np.max(T_interior),
            'T_min': np.min(T_interior),
            'T_mean': np.mean(T_interior),
            'T_std': np.std(T_interior)
        }
        
        # Compute total thermal energy
        cell_volume = self.mesh.get_cell_volume()
        stats['total_energy'] = np.sum(T_interior) * cell_volume
        
        # Compute heat flux at boundaries
        stats['boundary_flux'] = self._compute_boundary_flux(solver.T, solver.alpha)
        
        # Reaction statistics if enabled
        if solver.enable_reactions and solver.lambda_rxn is not None:
            lambda_interior = self.mesh.extract_interior(solver.lambda_rxn)
            stats['lambda_mean'] = np.mean(lambda_interior)
            stats['lambda_max'] = np.max(lambda_interior)
            stats['reaction_complete_percent'] = 100.0 * np.sum(lambda_interior > 0.99) / lambda_interior.size
            
        return stats
    
    def compute_conserved_quantities(self, solver):
        """
        Compute quantities that should be conserved (for verification).
        
        Parameters
        ----------
        solver : FVHeatSolver
            Solver object
            
        Returns
        -------
        dict
            Dictionary of conserved quantities
        """
        T_interior = self.mesh.extract_interior(solver.T)
        cell_volume = self.mesh.get_cell_volume()
        
        conserved = {
            'total_energy': np.sum(T_interior) * cell_volume,
            'total_cells': T_interior.size,
            'domain_volume': self.mesh.plate_length * self.mesh.plate_width
        }
        
        if solver.enable_reactions:
            # Without reactions, total energy should be conserved (with Neumann BC)
            # With reactions, track total chemical energy
            if solver.lambda_rxn is not None:
                lambda_interior = self.mesh.extract_interior(solver.lambda_rxn)
                conserved['total_reaction_progress'] = np.sum(lambda_interior) * cell_volume
                
        return conserved
    
    def extract_centerline_data(self, field, direction='x'):
        """
        Extract centerline data for 1D plots.
        
        Parameters
        ----------
        field : ndarray
            Field data with ghost cells
        direction : str
            'x' for horizontal centerline, 'y' for vertical centerline
            
        Returns
        -------
        coords : ndarray
            Coordinate values along centerline
        values : ndarray
            Field values along centerline
        """
        field_interior = self.mesh.extract_interior(field)
        
        if direction == 'x':
            # Horizontal centerline at y = plate_width/2
            j_center = self.mesh.ny // 2
            values = field_interior[j_center, :]
            coords = self.mesh.x_centers
        elif direction == 'y':
            # Vertical centerline at x = plate_length/2
            i_center = self.mesh.nx // 2
            values = field_interior[:, i_center]
            coords = self.mesh.y_centers
        else:
            raise ValueError(f"Unknown direction: {direction}")
            
        return coords, values
    
    def prepare_animation_frame(self, solver, nx_plot=100, ny_plot=100):
        """
        Prepare data for one animation frame.
        
        Parameters
        ----------
        solver : FVHeatSolver
            Solver object
        nx_plot : int
            Plotting resolution in x
        ny_plot : int
            Plotting resolution in y
            
        Returns
        -------
        dict
            Frame data ready for plotting
        """
        frame_data = self.get_solution_on_grid(solver, nx_plot, ny_plot, smooth=True)
        frame_data['stats'] = self.compute_statistics(solver)
        
        return frame_data
    
    def export_solution(self, solver, filename, format='npz'):
        """
        Export solution to file.
        
        Parameters
        ----------
        solver : FVHeatSolver
            Solver object
        filename : str
            Output filename
        format : str
            Export format ('npz', 'vtk', 'csv')
        """
        if format == 'npz':
            self._export_npz(solver, filename)
        elif format == 'vtk':
            self._export_vtk(solver, filename)
        elif format == 'csv':
            self._export_csv(solver, filename)
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def _export_npz(self, solver, filename):
        """Export to NumPy compressed format."""
        data = {
            'T': self.mesh.extract_interior(solver.T),
            'x_centers': self.mesh.x_centers,
            'y_centers': self.mesh.y_centers,
            'time': solver.current_time,
            'mesh_info': {
                'nx': self.mesh.nx,
                'ny': self.mesh.ny,
                'dx': self.mesh.dx,
                'dy': self.mesh.dy,
                'plate_length': self.mesh.plate_length,
                'plate_width': self.mesh.plate_width
            }
        }
        
        if solver.enable_reactions and solver.lambda_rxn is not None:
            data['lambda_rxn'] = self.mesh.extract_interior(solver.lambda_rxn)
            
        np.savez_compressed(filename, **data)
        
    def _export_vtk(self, solver, filename):
        """Export to VTK format for ParaView."""
        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError:
            print("VTK not available. Install with: pip install vtk")
            return
            
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(self.mesh.nx, self.mesh.ny, 1)
        
        # Create points
        points = vtk.vtkPoints()
        for j in range(self.mesh.ny):
            for i in range(self.mesh.nx):
                x = self.mesh.x_centers[i]
                y = self.mesh.y_centers[j]
                points.InsertNextPoint(x, y, 0)
                
        grid.SetPoints(points)
        
        # Add temperature data
        T_interior = self.mesh.extract_interior(solver.T).flatten(order='F')
        T_array = numpy_support.numpy_to_vtk(T_interior)
        T_array.SetName("Temperature")
        grid.GetPointData().AddArray(T_array)
        
        # Add reaction progress if available
        if solver.enable_reactions and solver.lambda_rxn is not None:
            lambda_interior = self.mesh.extract_interior(solver.lambda_rxn).flatten(order='F')
            lambda_array = numpy_support.numpy_to_vtk(lambda_interior)
            lambda_array.SetName("ReactionProgress")
            grid.GetPointData().AddArray(lambda_array)
            
        # Write file
        writer = vtk.vtkStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()
        
    def _export_csv(self, solver, filename):
        """Export to CSV format."""
        import pandas as pd
        
        # Create coordinate grids
        X, Y = np.meshgrid(self.mesh.x_centers, self.mesh.y_centers)
        
        # Flatten arrays
        data = {
            'x': X.flatten(),
            'y': Y.flatten(),
            'T': self.mesh.extract_interior(solver.T).flatten()
        }
        
        if solver.enable_reactions and solver.lambda_rxn is not None:
            data['lambda'] = self.mesh.extract_interior(solver.lambda_rxn).flatten()
            
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
    def _compute_boundary_flux(self, T, alpha):
        """
        Compute heat flux through boundaries.
        
        Parameters
        ----------
        T : ndarray
            Temperature field with ghost cells
        alpha : float
            Thermal diffusivity
            
        Returns
        -------
        dict
            Heat flux through each boundary
        """
        g = self.mesh.ghost_cells
        
        flux = {}
        
        # Bottom boundary (j = 0)
        dTdy = (T[g, g:-g] - T[g-1, g:-g]) / self.mesh.dy
        flux['bottom'] = -alpha * np.sum(dTdy) * self.mesh.dx
        
        # Top boundary (j = ny-1)
        dTdy = (T[-g, g:-g] - T[-g-1, g:-g]) / self.mesh.dy
        flux['top'] = -alpha * np.sum(dTdy) * self.mesh.dx
        
        # Left boundary (i = 0)
        dTdx = (T[g:-g, g] - T[g:-g, g-1]) / self.mesh.dx
        flux['left'] = -alpha * np.sum(dTdx) * self.mesh.dy
        
        # Right boundary (i = nx-1)
        dTdx = (T[g:-g, -g] - T[g:-g, -g-1]) / self.mesh.dx
        flux['right'] = -alpha * np.sum(dTdx) * self.mesh.dy
        
        # Net flux (should be ~0 for Neumann BC)
        flux['net'] = flux['bottom'] + flux['top'] + flux['left'] + flux['right']
        
        return flux
    
    def plot_mesh_with_solution(self, solver, show_values=False, show_hotspot=False,
                                hotspot_params=None, figsize=(10, 8)):
        """
        Plot the finite volume mesh with optional solution values and hotspot boundary.
        
        Parameters
        ----------
        solver : FVHeatSolver
            Solver object
        show_values : bool
            Show temperature values in cells
        show_hotspot : bool
            Overlay hotspot boundary
        hotspot_params : dict
            Parameters for hotspot circle: center_x, center_y, radius
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Extract interior solution
        T_interior = self.mesh.extract_interior(solver.T)
        
        # Create colormap normalization
        norm = Normalize(vmin=np.min(T_interior), vmax=np.max(T_interior))
        sm = ScalarMappable(norm=norm, cmap='hot')
        
        # Plot cells
        for j in range(self.mesh.ny):
            for i in range(self.mesh.nx):
                # Cell boundaries
                x_left = self.mesh.x_faces[i]
                x_right = self.mesh.x_faces[i+1]
                y_bottom = self.mesh.y_faces[j]
                y_top = self.mesh.y_faces[j+1]
                
                # Cell temperature
                T_cell = T_interior[j, i]
                color = sm.to_rgba(T_cell)
                
                # Draw cell
                rect = patches.Rectangle((x_left, y_bottom), 
                                       x_right - x_left, 
                                       y_top - y_bottom,
                                       linewidth=0.5, 
                                       edgecolor='black',
                                       facecolor=color,
                                       alpha=0.8)
                ax.add_patch(rect)
                
                # Show temperature value if requested
                if show_values and self.mesh.nx <= 20 and self.mesh.ny <= 20:
                    x_center = self.mesh.x_centers[i]
                    y_center = self.mesh.y_centers[j]
                    ax.text(x_center, y_center, f'{T_cell:.0f}', 
                           ha='center', va='center', fontsize=8,
                           color='white' if T_cell > (norm.vmax + norm.vmin)/2 else 'black')
        
        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Temperature (K)', fontsize=12)
        
        # Overlay hotspot boundary if requested
        if show_hotspot and hotspot_params is not None:
            center_x = hotspot_params.get('center_x', self.mesh.plate_length/2)
            center_y = hotspot_params.get('center_y', self.mesh.plate_width/2)
            radius = hotspot_params.get('radius', 0.05)
            
            # Draw hotspot circle
            circle = patches.Circle((center_x, center_y), radius,
                                  linewidth=2, edgecolor='lime',
                                  facecolor='none', linestyle='--',
                                  label='Initial hotspot boundary')
            ax.add_patch(circle)
            ax.legend()
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Finite Volume Mesh ({self.mesh.nx}×{self.mesh.ny} cells) at t={solver.current_time:.3f}s',
                    fontsize=14)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.mesh.plate_length)
        ax.set_ylim(0, self.mesh.plate_width)
        
        plt.tight_layout()
        return fig, ax
    
    
    def create_convergence_data(self, solver_list, exact_solution=None):
        """
        Create convergence study data from multiple solver runs.
        
        Parameters
        ----------
        solver_list : list
            List of (mesh_size, solver) tuples
        exact_solution : callable, optional
            Function exact_solution(x, y, t) for error computation
            
        Returns
        -------
        dict
            Convergence data
        """
        convergence = {
            'mesh_sizes': [],
            'errors_L2': [],
            'errors_Linf': [],
            'cell_sizes': []
        }
        
        for mesh_size, solver in solver_list:
            convergence['mesh_sizes'].append(mesh_size)
            convergence['cell_sizes'].append(solver.mesh.dx)
            
            if exact_solution is not None:
                # Compute error norms
                T_interior = solver.mesh.extract_interior(solver.T)
                T_exact = np.zeros_like(T_interior)
                
                for j in range(solver.mesh.ny):
                    for i in range(solver.mesh.nx):
                        x = solver.mesh.x_centers[i]
                        y = solver.mesh.y_centers[j]
                        T_exact[j, i] = exact_solution(x, y, solver.current_time)
                        
                error = T_interior - T_exact
                L2_error = np.sqrt(np.mean(error**2))
                Linf_error = np.max(np.abs(error))
                
                convergence['errors_L2'].append(L2_error)
                convergence['errors_Linf'].append(Linf_error)
                
        return convergence

    def plot_centerlines_from_history(self, centerline_history, hotspot_params=None, 
                                    show_mesh_lines=False):
        """
        Plot centerlines from collected history data.
        
        Parameters
        ----------
        centerline_history : dict
            History data from solver.get_centerline_history()
        hotspot_params : dict, optional
            Hotspot parameters for overlay
        show_mesh_lines : bool
            Show mesh grid lines
            
        Returns
        -------
        fig, axes : matplotlib objects
        """
        import matplotlib.pyplot as plt
        
        if centerline_history is None:
            raise ValueError("No centerline history data available")
            
        # Extract data
        times = centerline_history['times']
        x_data = centerline_history['x_centerline']
        y_data = centerline_history['y_centerline']
        x_coords = centerline_history['x_coords']
        y_coords = centerline_history['y_coords']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Create colormap for time evolution
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
        
        # Plot X-centerline evolution
        for idx, (t, color) in enumerate(zip(times, colors)):
            # Format time label
            if t < 1e-6:
                label = f't = {t*1e9:.1f} ns'
            elif t < 1e-3:
                label = f't = {t*1e6:.1f} μs'
            elif t < 1:
                label = f't = {t*1e3:.1f} ms'
            else:
                label = f't = {t:.3f} s'
                
            ax1.plot(x_coords, x_data[idx], color=color, linewidth=2, label=label)
        
        # Add mesh lines if requested
        if show_mesh_lines:
            for i in range(self.mesh.nx + 1):
                ax1.axvline(x=self.mesh.x_faces[i], color='gray', 
                           linewidth=0.5, alpha=0.3)
        
        # Add hotspot markers if provided
        if hotspot_params is not None:
            center_x = hotspot_params.get('center_x', self.mesh.plate_length/2)
            center_y = hotspot_params.get('center_y', self.mesh.plate_width/2)
            radius = hotspot_params.get('radius', 0.05)
            
            # For x-centerline at y = plate_width/2
            y_center = self.mesh.plate_width / 2
            if abs(y_center - center_y) <= radius:
                dy = y_center - center_y
                dx = np.sqrt(radius**2 - dy**2)
                x_left = center_x - dx
                x_right = center_x + dx
                
                ax1.axvspan(x_left, x_right, alpha=0.1, color='red')
                ax1.axvline(x=x_left, color='green', linewidth=1.5, 
                           linestyle='--', alpha=0.5)
                ax1.axvline(x=x_right, color='green', linewidth=1.5, 
                           linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('X Position (m)', fontsize=12)
        ax1.set_ylabel('Temperature (K)', fontsize=12)
        ax1.set_title(f'Temperature Evolution along X-Centerline (y = {self.mesh.plate_width/2:.3f} m)', 
                     fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot Y-centerline evolution
        for idx, (t, color) in enumerate(zip(times, colors)):
            # Format time label
            if t < 1e-6:
                label = f't = {t*1e9:.1f} ns'
            elif t < 1e-3:
                label = f't = {t*1e6:.1f} μs'
            elif t < 1:
                label = f't = {t*1e3:.1f} ms'
            else:
                label = f't = {t:.3f} s'
                
            ax2.plot(y_coords, y_data[idx], color=color, linewidth=2, label=label)
        
        # Add mesh lines if requested
        if show_mesh_lines:
            for j in range(self.mesh.ny + 1):
                ax2.axvline(x=self.mesh.y_faces[j], color='gray', 
                           linewidth=0.5, alpha=0.3)
        
        # Add hotspot markers if provided
        if hotspot_params is not None:
            # For y-centerline at x = plate_length/2
            x_center = self.mesh.plate_length / 2
            if abs(x_center - center_x) <= radius:
                dx = x_center - center_x
                dy = np.sqrt(radius**2 - dx**2)
                y_bottom = center_y - dy
                y_top = center_y + dy
                
                ax2.axvspan(y_bottom, y_top, alpha=0.1, color='red')
                ax2.axvline(x=y_bottom, color='green', linewidth=1.5, 
                           linestyle='--', alpha=0.5)
                ax2.axvline(x=y_top, color='green', linewidth=1.5, 
                           linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Y Position (m)', fontsize=12)
        ax2.set_ylabel('Temperature (K)', fontsize=12)
        ax2.set_title(f'Temperature Evolution along Y-Centerline (x = {self.mesh.plate_length/2:.3f} m)', 
                     fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_centerlines(self, solver, direction='both', time_label=None, ax=None, 
                        show_mesh_lines=False, show_hotspot_marker=False, hotspot_params=None):
        """
        Plot temperature along centerlines.
        
        Parameters
        ----------
        solver : FVHeatSolver
            Solver object
        direction : str
            'x', 'y', or 'both'
        time_label : str, optional
            Time label for the plot
        ax : matplotlib axes, optional
            Axes to plot on (creates new if None)
        show_mesh_lines : bool
            Show vertical lines at cell boundaries
        show_hotspot_marker : bool
            Mark hotspot boundaries on centerline
        hotspot_params : dict
            Hotspot parameters (center_x, center_y, radius)
            
        Returns
        -------
        fig, ax : matplotlib objects
            Figure and axes (only fig if ax was provided)
        """
        import matplotlib.pyplot as plt
        
        # Create axes if needed
        if ax is None:
            if direction == 'both':
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax1 = ax if direction == 'x' else None
                ax2 = ax if direction == 'y' else None
        else:
            fig = ax.figure
            if direction == 'both':
                raise ValueError("Cannot plot both directions on single provided axes")
            ax1 = ax if direction == 'x' else None
            ax2 = ax if direction == 'y' else None
        
        # Time label
        if time_label is None:
            time_label = f't = {solver.current_time:.3f} s'
        
        # Plot x-centerline
        if direction in ['x', 'both']:
            x_coords, T_x = self.extract_centerline_data(solver.T, 'x')
            ax1.plot(x_coords, T_x, 'b-', linewidth=2, label=time_label)
            
            # Add mesh lines
            if show_mesh_lines:
                for i in range(self.mesh.nx + 1):
                    ax1.axvline(x=self.mesh.x_faces[i], color='gray', 
                               linewidth=0.5, alpha=0.3)
            
            # Add hotspot markers
            if show_hotspot_marker and hotspot_params is not None:
                center_x = hotspot_params.get('center_x', self.mesh.plate_length/2)
                center_y = hotspot_params.get('center_y', self.mesh.plate_width/2)
                radius = hotspot_params.get('radius', 0.05)
                
                # For x-centerline, we're at y = plate_width/2
                y_center_line = self.mesh.plate_width / 2
                
                # Check if centerline passes through hotspot
                if abs(y_center_line - center_y) <= radius:
                    # Calculate x-intersections
                    dy = y_center_line - center_y
                    dx = np.sqrt(radius**2 - dy**2)
                    x_left = center_x - dx
                    x_right = center_x + dx
                    
                    # Mark boundaries
                    ax1.axvline(x=x_left, color='green', linewidth=2, 
                               linestyle='--', alpha=0.7, label='Hotspot boundary')
                    ax1.axvline(x=x_right, color='green', linewidth=2, 
                               linestyle='--', alpha=0.7)
                    
                    # Shade hotspot region
                    ax1.axvspan(x_left, x_right, alpha=0.1, color='red', 
                               label='Initial hotspot region')
            
            ax1.set_xlabel('X Position (m)', fontsize=12)
            ax1.set_ylabel('Temperature (K)', fontsize=12)
            ax1.set_title(f'Temperature along X-Centerline (y = {self.mesh.plate_width/2:.3f} m)', 
                         fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Plot y-centerline
        if direction in ['y', 'both']:
            y_coords, T_y = self.extract_centerline_data(solver.T, 'y')
            ax2.plot(y_coords, T_y, 'r-', linewidth=2, label=time_label)
            
            # Add mesh lines
            if show_mesh_lines:
                for j in range(self.mesh.ny + 1):
                    ax2.axvline(x=self.mesh.y_faces[j], color='gray', 
                               linewidth=0.5, alpha=0.3)
            
            # Add hotspot markers
            if show_hotspot_marker and hotspot_params is not None:
                center_x = hotspot_params.get('center_x', self.mesh.plate_length/2)
                center_y = hotspot_params.get('center_y', self.mesh.plate_width/2)
                radius = hotspot_params.get('radius', 0.05)
                
                # For y-centerline, we're at x = plate_length/2
                x_center_line = self.mesh.plate_length / 2
                
                # Check if centerline passes through hotspot
                if abs(x_center_line - center_x) <= radius:
                    # Calculate y-intersections
                    dx = x_center_line - center_x
                    dy = np.sqrt(radius**2 - dx**2)
                    y_bottom = center_y - dy
                    y_top = center_y + dy
                    
                    # Mark boundaries
                    ax2.axvline(x=y_bottom, color='green', linewidth=2, 
                               linestyle='--', alpha=0.7, label='Hotspot boundary')
                    ax2.axvline(x=y_top, color='green', linewidth=2, 
                               linestyle='--', alpha=0.7)
                    
                    # Shade hotspot region
                    ax2.axvspan(y_bottom, y_top, alpha=0.1, color='red', 
                               label='Initial hotspot region')
            
            ax2.set_xlabel('Y Position (m)', fontsize=12)
            ax2.set_ylabel('Temperature (K)', fontsize=12)
            ax2.set_title(f'Temperature along Y-Centerline (x = {self.mesh.plate_length/2:.3f} m)', 
                         fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        if ax is None:
            return fig, (ax1, ax2) if direction == 'both' else (ax1 if direction == 'x' else ax2)
        else:
            return fig
    
    def plot_centerlines_evolution(self, solver_states, times, direction='both',
                                 show_mesh_lines=False, show_hotspot_marker=False,
                                 hotspot_params=None):
        """
        Plot centerline temperature evolution at multiple times.
        
        Parameters
        ----------
        solver_states : list
            List of solver states (each containing T field)
        times : list
            Corresponding time values
        direction : str
            'x', 'y', or 'both'
        show_mesh_lines : bool
            Show cell boundaries
        show_hotspot_marker : bool
            Mark initial hotspot region
        hotspot_params : dict
            Hotspot parameters
            
        Returns
        -------
        fig, axes : matplotlib objects
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        if direction == 'both':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax1 = ax if direction == 'x' else None
            ax2 = ax if direction == 'y' else None
        
        # Color map for different times
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
        
        # Plot each time
        for idx, (state, t, color) in enumerate(zip(solver_states, times, colors)):
            # Time label
            if t < 1e-6:
                time_label = f't = {t*1e9:.1f} ns'
            elif t < 1e-3:
                time_label = f't = {t*1e6:.1f} μs'
            elif t < 1:
                time_label = f't = {t*1e3:.1f} ms'
            else:
                time_label = f't = {t:.3f} s'
            
            # X-centerline
            if direction in ['x', 'both']:
                x_coords, T_x = self.extract_centerline_data(state['T'], 'x')
                ax1.plot(x_coords, T_x, color=color, linewidth=2, label=time_label)
            
            # Y-centerline
            if direction in ['y', 'both']:
                y_coords, T_y = self.extract_centerline_data(state['T'], 'y')
                ax2.plot(y_coords, T_y, color=color, linewidth=2, label=time_label)
        
        # Add mesh lines and hotspot markers (only once)
        if show_mesh_lines:
            if direction in ['x', 'both']:
                for i in range(self.mesh.nx + 1):
                    ax1.axvline(x=self.mesh.x_faces[i], color='gray', 
                               linewidth=0.5, alpha=0.3)
            if direction in ['y', 'both']:
                for j in range(self.mesh.ny + 1):
                    ax2.axvline(x=self.mesh.y_faces[j], color='gray', 
                               linewidth=0.5, alpha=0.3)
        
        if show_hotspot_marker and hotspot_params is not None:
            center_x = hotspot_params.get('center_x', self.mesh.plate_length/2)
            center_y = hotspot_params.get('center_y', self.mesh.plate_width/2)
            radius = hotspot_params.get('radius', 0.05)
            
            # X-centerline hotspot
            if direction in ['x', 'both']:
                y_center_line = self.mesh.plate_width / 2
                if abs(y_center_line - center_y) <= radius:
                    dy = y_center_line - center_y
                    dx = np.sqrt(radius**2 - dy**2)
                    x_left = center_x - dx
                    x_right = center_x + dx
                    ax1.axvspan(x_left, x_right, alpha=0.1, color='red')
                    ax1.axvline(x=x_left, color='green', linewidth=1.5, 
                               linestyle='--', alpha=0.5)
                    ax1.axvline(x=x_right, color='green', linewidth=1.5, 
                               linestyle='--', alpha=0.5)
            
            # Y-centerline hotspot
            if direction in ['y', 'both']:
                x_center_line = self.mesh.plate_length / 2
                if abs(x_center_line - center_x) <= radius:
                    dx = x_center_line - center_x
                    dy = np.sqrt(radius**2 - dx**2)
                    y_bottom = center_y - dy
                    y_top = center_y + dy
                    ax2.axvspan(y_bottom, y_top, alpha=0.1, color='red')
                    ax2.axvline(x=y_bottom, color='green', linewidth=1.5, 
                               linestyle='--', alpha=0.5)
                    ax2.axvline(x=y_top, color='green', linewidth=1.5, 
                               linestyle='--', alpha=0.5)
        
        # Format axes
        if direction in ['x', 'both']:
            ax1.set_xlabel('X Position (m)', fontsize=12)
            ax1.set_ylabel('Temperature (K)', fontsize=12)
            ax1.set_title(f'Temperature Evolution along X-Centerline (y = {self.mesh.plate_width/2:.3f} m)', 
                         fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')
        
        if direction in ['y', 'both']:
            ax2.set_xlabel('Y Position (m)', fontsize=12)
            ax2.set_ylabel('Temperature (K)', fontsize=12)
            ax2.set_title(f'Temperature Evolution along Y-Centerline (x = {self.mesh.plate_length/2:.3f} m)', 
                         fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')
        
        plt.tight_layout()
        return fig, (ax1, ax2) if direction == 'both' else (ax1 if direction == 'x' else ax2)
    
    