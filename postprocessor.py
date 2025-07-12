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