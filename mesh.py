# mesh.py
import numpy as np

class FVMesh:
    """
    Finite Volume mesh with cell-centered storage and ghost cells.
    
    This class manages the mesh geometry, coordinate systems, and provides
    utilities for index management between physical and computational domains.
    """
    
    def __init__(self, nx_cells, ny_cells, plate_length, plate_width, ghost_cells=3):
        """
        Initialize a 2D finite volume mesh.
        
        Parameters
        ----------
        nx_cells : int
            Number of cells in x-direction (physical domain)
        ny_cells : int
            Number of cells in y-direction (physical domain)
        plate_length : float
            Physical domain length in x-direction (m)
        plate_width : float
            Physical domain width in y-direction (m)
        ghost_cells : int
            Number of ghost cells on each boundary (default: 3 for WENO5)
        """
        self.nx = nx_cells
        self.ny = ny_cells
        self.plate_length = plate_length
        self.plate_width = plate_width
        self.ghost_cells = ghost_cells
        
        # Cell dimensions
        self.dx = plate_length / nx_cells
        self.dy = plate_width / ny_cells
        
        # Total array dimensions including ghost cells
        self.nx_total = nx_cells + 2 * ghost_cells
        self.ny_total = ny_cells + 2 * ghost_cells
        
        # Setup coordinate systems
        self._setup_coordinates()
        
    def _setup_coordinates(self):
        """Setup cell centers, faces, and coordinate arrays."""
        
        # Cell centers in physical domain (excluding ghost cells)
        self.x_centers = np.linspace(self.dx/2, self.plate_length - self.dx/2, self.nx)
        self.y_centers = np.linspace(self.dy/2, self.plate_width - self.dy/2, self.ny)
        
        # Cell faces for flux computation
        self.x_faces = np.linspace(0, self.plate_length, self.nx + 1)
        self.y_faces = np.linspace(0, self.plate_width, self.ny + 1)
        
        # Extended coordinates including ghost cells
        x_start = -self.ghost_cells * self.dx + self.dx/2
        x_end = self.plate_length + (self.ghost_cells - 0.5) * self.dx
        self.x_all = np.linspace(x_start, x_end, self.nx_total)
        
        y_start = -self.ghost_cells * self.dy + self.dy/2
        y_end = self.plate_width + (self.ghost_cells - 0.5) * self.dy
        self.y_all = np.linspace(y_start, y_end, self.ny_total)
        
        # 2D mesh grids for plotting and initialization
        self.X_centers, self.Y_centers = np.meshgrid(self.x_centers, self.y_centers)
        self.X_all, self.Y_all = np.meshgrid(self.x_all, self.y_all)
        
        # Store physical domain bounds for easy access
        self.x_min = 0.0
        self.x_max = self.plate_length
        self.y_min = 0.0
        self.y_max = self.plate_width
        
    def get_interior_slice(self):
        """
        Return slice objects for accessing interior cells (physical domain).
        
        Returns
        -------
        tuple
            (j_slice, i_slice) for indexing arrays as arr[j_slice, i_slice]
        """
        g = self.ghost_cells
        return slice(g, -g), slice(g, -g)
    
    def get_interior_indices(self):
        """
        Return the start and end indices for interior cells.
        
        Returns
        -------
        dict
            Dictionary with 'j_start', 'j_end', 'i_start', 'i_end'
        """
        g = self.ghost_cells
        return {
            'j_start': g,
            'j_end': self.ny_total - g,
            'i_start': g,
            'i_end': self.nx_total - g
        }
    
    def physical_to_index(self, x, y):
        """
        Convert physical coordinates to cell indices.
        
        Parameters
        ----------
        x : float or array
            X-coordinate(s) in physical space
        y : float or array
            Y-coordinate(s) in physical space
            
        Returns
        -------
        i, j : int or array
            Cell indices including ghost cell offset
        """
        # Find cell in physical domain
        i_phys = np.clip(np.floor(x / self.dx).astype(int), 0, self.nx - 1)
        j_phys = np.clip(np.floor(y / self.dy).astype(int), 0, self.ny - 1)
        
        # Add ghost cell offset
        i = i_phys + self.ghost_cells
        j = j_phys + self.ghost_cells
        
        return i, j
    
    def index_to_physical(self, i, j):
        """
        Convert cell indices to physical coordinates (cell centers).
        
        Parameters
        ----------
        i : int or array
            Cell index in x-direction (including ghost offset)
        j : int or array
            Cell index in y-direction (including ghost offset)
            
        Returns
        -------
        x, y : float or array
            Physical coordinates of cell centers
        """
        return self.x_all[i], self.y_all[j]
    
    def get_cell_volume(self):
        """
        Get cell volume (area in 2D).
        
        Returns
        -------
        float
            Cell volume (dx * dy)
        """
        return self.dx * self.dy
    
    def get_face_area_x(self):
        """Get face area for x-direction faces (dy in 2D)."""
        return self.dy
    
    def get_face_area_y(self):
        """Get face area for y-direction faces (dx in 2D)."""
        return self.dx
    
    def create_field(self, init_value=0.0):
        """
        Create a field array with proper dimensions including ghost cells.
        
        Parameters
        ----------
        init_value : float
            Initial value for all cells
            
        Returns
        -------
        ndarray
            2D array of shape (ny_total, nx_total)
        """
        return np.full((self.ny_total, self.nx_total), init_value, dtype=np.float64)
    
    def extract_interior(self, field):
        """
        Extract interior (physical) domain from a field with ghost cells.
        
        Parameters
        ----------
        field : ndarray
            2D array including ghost cells
            
        Returns
        -------
        ndarray
            2D array of shape (ny, nx) containing only physical domain
        """
        j_slice, i_slice = self.get_interior_slice()
        return field[j_slice, i_slice].copy()
    
    def extend_with_ghost(self, interior_field):
        """
        Create a full field array from interior data, filling ghost cells with zeros.
        
        Parameters
        ----------
        interior_field : ndarray
            2D array of shape (ny, nx) containing physical domain data
            
        Returns
        -------
        ndarray
            2D array of shape (ny_total, nx_total) with ghost cells
        """
        field = self.create_field(0.0)
        j_slice, i_slice = self.get_interior_slice()
        field[j_slice, i_slice] = interior_field
        return field
    
    def apply_neumann_bc(self, field):
        """
        Apply Neumann (zero gradient) boundary conditions.
        
        Parameters
        ----------
        field : ndarray
            2D array including ghost cells (modified in-place)
        """
        g = self.ghost_cells
        
        # Copy boundary values to ghost cells
        # Bottom boundary
        for k in range(g):
            field[g-1-k, g:-g] = field[g, g:-g]
            
        # Top boundary
        for k in range(g):
            field[-g+k, g:-g] = field[-g-1, g:-g]
            
        # Left boundary (including corners)
        for k in range(g):
            field[:, g-1-k] = field[:, g]
            
        # Right boundary (including corners)
        for k in range(g):
            field[:, -g+k] = field[:, -g-1]
    
    def apply_dirichlet_bc(self, field, boundary_values):
        """
        Apply Dirichlet boundary conditions.
        
        Parameters
        ----------
        field : ndarray
            2D array including ghost cells (modified in-place)
        boundary_values : dict
            Dictionary with keys 'bottom', 'top', 'left', 'right' containing
            boundary values or arrays
        """
        g = self.ghost_cells
        
        # Apply boundary values
        if 'bottom' in boundary_values:
            field[:g, g:-g] = boundary_values['bottom']
        if 'top' in boundary_values:
            field[-g:, g:-g] = boundary_values['top']
        if 'left' in boundary_values:
            field[:, :g] = boundary_values['left']
        if 'right' in boundary_values:
            field[:, -g:] = boundary_values['right']
    
    def compute_gradient_magnitude(self, field):
        """
        Compute gradient magnitude at cell centers for AMR criteria.
        
        Parameters
        ----------
        field : ndarray
            2D array including ghost cells
            
        Returns
        -------
        ndarray
            Gradient magnitude at interior cells
        """
        j_slice, i_slice = self.get_interior_slice()
        
        # Central differences for interior cells
        grad_x = (field[j_slice, i_slice.start+1:i_slice.stop+1] - 
                  field[j_slice, i_slice.start-1:i_slice.stop-1]) / (2 * self.dx)
        
        grad_y = (field[j_slice.start+1:j_slice.stop+1, i_slice] - 
                  field[j_slice.start-1:j_slice.stop-1, i_slice]) / (2 * self.dy)
        
        # Adjust slicing to ensure matching dimensions
        grad_x = grad_x[:, 1:-1]
        grad_y = grad_y[1:-1, :]
        
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def __str__(self):
        """String representation of mesh information."""
        return (f"FVMesh: {self.nx}×{self.ny} cells, "
                f"domain: [{self.plate_length}×{self.plate_width}] m, "
                f"dx={self.dx:.3e} m, dy={self.dy:.3e} m, "
                f"ghost cells: {self.ghost_cells}")