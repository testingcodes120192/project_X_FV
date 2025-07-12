# initial_conditions.py
import numpy as np
from abc import ABC, abstractmethod

class FVInitialCondition(ABC):
    """Base class for finite volume initial conditions."""
    
    @abstractmethod
    def set(self, solver):
        """Set the initial condition on the solver."""
        pass

class FVCircularHotspot(FVInitialCondition):
    """
    Circular hotspot initial condition for FV solver.
    
    Creates a circular region of high temperature in a cooler background.
    """
    
    def __init__(self, params):
        """
        Initialize circular hotspot parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            - background_temp : float, background temperature (K)
            - hotspot_temp : float, hotspot temperature (K)
            - center_x : float, x-coordinate of center (m)
            - center_y : float, y-coordinate of center (m)
            - hotspot_radius : float, radius of hotspot (m)
            - smooth_transition : bool, use smooth transition
            - transition_width : float, width of transition zone (m)
        """
        self.params = params
        
    def set(self, solver):
        """Set circular hotspot on the FV solver."""
        # Extract parameters with defaults
        T_background = self.params.get('background_temp', 300.0)
        T_hotspot = self.params.get('hotspot_temp', 6000.0)
        center_x = self.params.get('center_x', solver.mesh.plate_length / 2)
        center_y = self.params.get('center_y', solver.mesh.plate_width / 2)
        hotspot_radius = self.params.get('hotspot_radius', 0.05)
        smooth_transition = self.params.get('smooth_transition', True)
        transition_width = self.params.get('transition_width', 0.005)
        
        # Use solver's built-in method
        solver.set_initial_condition_circular(
            T_background=T_background,
            T_hotspot=T_hotspot,
            center_x=center_x,
            center_y=center_y,
            hotspot_radius=hotspot_radius,
            smooth_transition=smooth_transition,
            transition_width=transition_width
        )
        
        print(f"Set circular hotspot IC: T_bg={T_background}K, T_hot={T_hotspot}K, "
              f"center=({center_x:.3f}, {center_y:.3f})m, radius={hotspot_radius}m")


class FVImageBased(FVInitialCondition):
    """
    Image-based initial condition for FV solver.
    
    Maps temperature from an image to the FV grid using cell averages.
    """
    
    def __init__(self, params):
        """
        Initialize image-based IC parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            - image_path : str, path to temperature image
            - use_constant_temp : bool, use constant T for marked regions
            - constant_temp : float, temperature for marked regions (K)
            - smooth_interface : bool, smooth the interface
            - smooth_width : int, smoothing width in pixels
            - domain_size_microns : float, physical domain size
            - interactive : bool, use interactive selection
        """
        self.params = params
        
    def set(self, solver):
        """Set image-based initial condition on the FV solver."""
        # Import here to avoid circular dependency
        from image_tools import TemperatureRegionToFRGrid
        
        # Extract parameters
        image_path = self.params.get('image_path')
        if not image_path:
            raise ValueError("No image path provided")
            
        T_background = self.params.get('background_temp', 300.0)
        use_constant_temp = self.params.get('use_constant_temp', True)
        constant_temp = self.params.get('constant_temp', 1000.0)
        smooth_interface = self.params.get('smooth_interface', True)
        smooth_width = self.params.get('smooth_width', 2)
        domain_size_microns = self.params.get('domain_size_microns', None)
        interactive = self.params.get('interactive', True)
        
        # Create temperature mapper
        # Note: We'll adapt the existing FR mapper for FV use
        mapper = TemperatureRegionToFRGrid(
            image_path=image_path,
            nx_elem=solver.mesh.nx,  # Use number of cells
            ny_elem=solver.mesh.ny,
            p=0,  # No polynomial order for FV
            plate_length=None,  # Will be determined from image
            plate_width=None,
            interactive=interactive,
            domain_size_microns=domain_size_microns
        )
        
        # Update solver's mesh dimensions if needed
        if mapper.plate_length != solver.mesh.plate_length:
            print(f"Warning: Adjusting domain size from {solver.mesh.plate_length}m "
                  f"to {mapper.plate_length}m based on image")
            # Would need to recreate mesh here in full implementation
            
        # Map temperature to FV cells
        T_cells = self._map_to_cells(
            mapper, solver.mesh, T_background, 
            use_constant_temp, constant_temp,
            smooth_interface, smooth_width
        )
        
        # Set on solver
        solver.set_initial_condition_from_array(T_cells)
        
        # Print analysis if available
        if hasattr(mapper, 'print_region_geometry'):
            mapper.print_region_geometry()
            
    def _map_to_cells(self, mapper, mesh, T_background, use_constant_temp,
                      constant_temp, smooth_interface, smooth_width):
        """
        Map image temperatures to FV cells.
        
        For finite volume, we need cell-averaged values rather than point values.
        """
        from scipy.ndimage import gaussian_filter
        from scipy.interpolate import RegularGridInterpolator
        
        # Get temperature map from image
        if use_constant_temp:
            temp_map = np.where(mapper.temp_mask, constant_temp, T_background)
        else:
            if hasattr(mapper, 'temperature_mapping') and mapper.temperature_mapping:
                temp_map = mapper.create_temperature_map(T_background, smooth_interface, smooth_width)
            else:
                temp_map = np.where(mapper.temp_mask, 1000.0, T_background)
                
        # Apply smoothing if requested
        if smooth_interface:
            temp_map = gaussian_filter(temp_map.astype(float), sigma=smooth_width)
            
        # Create interpolator for the image
        y_img = np.arange(mapper.domain_img.shape[0])
        x_img = np.arange(mapper.domain_img.shape[1])
        
        interp_func = RegularGridInterpolator(
            (y_img, x_img),
            temp_map,
            method='linear',
            bounds_error=False,
            fill_value=T_background
        )
        
        # Initialize cell array
        T_cells = np.zeros((mesh.ny, mesh.nx))
        
        # For each cell, compute average temperature
        # Simple approach: sample at cell center
        # More accurate: sample at multiple points and average
        for j in range(mesh.ny):
            for i in range(mesh.nx):
                x_phys = mesh.x_centers[i]
                y_phys = mesh.y_centers[j]
                
                # Convert to image coordinates
                x_img_coord = (x_phys * 1e6) / mapper.X_ratio
                y_img_coord = (y_phys * 1e6) / mapper.Y_ratio
                
                # Get temperature (simple center sampling)
                T_cells[j, i] = interp_func((y_img_coord, x_img_coord))
                
        # Ensure physical bounds
        T_cells = np.maximum(T_cells, T_background * 0.99)
        T_cells = np.minimum(T_cells, max(constant_temp, T_background) * 1.1)
        
        return T_cells


class FVGaussianPulse(FVInitialCondition):
    """
    Gaussian pulse initial condition (useful for testing diffusion).
    """
    
    def __init__(self, params):
        """
        Initialize Gaussian pulse parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            - background_temp : float, background temperature (K)
            - pulse_amplitude : float, pulse amplitude above background (K)
            - center_x : float, x-coordinate of center (m)
            - center_y : float, y-coordinate of center (m)
            - sigma_x : float, standard deviation in x (m)
            - sigma_y : float, standard deviation in y (m)
        """
        self.params = params
        
    def set(self, solver):
        """Set Gaussian pulse on the FV solver."""
        T_background = self.params.get('background_temp', 300.0)
        amplitude = self.params.get('pulse_amplitude', 1000.0)
        center_x = self.params.get('center_x', solver.mesh.plate_length / 2)
        center_y = self.params.get('center_y', solver.mesh.plate_width / 2)
        sigma_x = self.params.get('sigma_x', solver.mesh.plate_length / 10)
        sigma_y = self.params.get('sigma_y', solver.mesh.plate_width / 10)
        
        # Initialize array
        T_cells = np.zeros((solver.mesh.ny, solver.mesh.nx))
        
        # Set Gaussian pulse
        for j in range(solver.mesh.ny):
            for i in range(solver.mesh.nx):
                x = solver.mesh.x_centers[i]
                y = solver.mesh.y_centers[j]
                
                # Gaussian function
                exponent = -0.5 * ((x - center_x)**2 / sigma_x**2 + 
                                  (y - center_y)**2 / sigma_y**2)
                T_cells[j, i] = T_background + amplitude * np.exp(exponent)
                
        solver.set_initial_condition_from_array(T_cells)
        
        print(f"Set Gaussian pulse IC: T_bg={T_background}K, amplitude={amplitude}K, "
              f"center=({center_x:.3f}, {center_y:.3f})m, "
              f"sigma=({sigma_x:.3f}, {sigma_y:.3f})m")


class FVMultipleHotspots(FVInitialCondition):
    """
    Multiple hotspots initial condition for testing interactions.
    """
    
    def __init__(self, params):
        """
        Initialize multiple hotspots parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing:
            - background_temp : float, background temperature (K)
            - hotspots : list of dicts, each with:
                - temp : float, hotspot temperature (K)
                - center_x : float, x-coordinate (m)
                - center_y : float, y-coordinate (m)
                - radius : float, hotspot radius (m)
        """
        self.params = params
        
    def set(self, solver):
        """Set multiple hotspots on the FV solver."""
        T_background = self.params.get('background_temp', 300.0)
        hotspots = self.params.get('hotspots', [])
        
        # Initialize to background
        T_cells = np.full((solver.mesh.ny, solver.mesh.nx), T_background)
        
        # Add each hotspot
        for hotspot in hotspots:
            T_hot = hotspot.get('temp', 1000.0)
            cx = hotspot.get('center_x', 0.0)
            cy = hotspot.get('center_y', 0.0)
            radius = hotspot.get('radius', 0.05)
            
            for j in range(solver.mesh.ny):
                for i in range(solver.mesh.nx):
                    x = solver.mesh.x_centers[i]
                    y = solver.mesh.y_centers[j]
                    r = np.sqrt((x - cx)**2 + (y - cy)**2)
                    
                    if r <= radius:
                        # Take maximum temperature if overlapping
                        T_cells[j, i] = max(T_cells[j, i], T_hot)
                        
        solver.set_initial_condition_from_array(T_cells)
        
        print(f"Set {len(hotspots)} hotspots with T_bg={T_background}K")


def get_fv_initial_condition(ic_type, params):
    """
    Factory function to get FV initial condition plugin.
    
    Parameters
    ----------
    ic_type : str
        Type of initial condition ('circular', 'image', 'gaussian', 'multiple')
    params : dict
        Parameters for the initial condition
        
    Returns
    -------
    FVInitialCondition
        Initial condition object
    """
    ic_map = {
        'circular': FVCircularHotspot,
        'image': FVImageBased,
        'gaussian': FVGaussianPulse,
        'multiple': FVMultipleHotspots
    }
    
    if ic_type not in ic_map:
        raise ValueError(f"Unknown initial condition type: {ic_type}")
        
    return ic_map[ic_type](params)


class FVICAdapter:
    """
    Adapter to use existing FR initial condition plugins with FV solver.
    """
    
    def __init__(self, fr_ic_plugin):
        """
        Initialize adapter with FR plugin.
        
        Parameters
        ----------
        fr_ic_plugin : object
            FR initial condition plugin
        """
        self.fr_plugin = fr_ic_plugin
        
    def set(self, fv_solver):
        """
        Adapt FR initial condition to FV solver.
        
        This creates a temporary FR-like structure, applies the IC,
        then extracts cell averages for FV.
        """
        # Create mock FR solver structure
        class MockFRSolver:
            def __init__(self, mesh):
                self.nx_elem = mesh.nx
                self.ny_elem = mesh.ny
                self.plate_length = mesh.plate_length
                self.plate_width = mesh.plate_width
                # Create 4D array with just 1 point per element (cell center)
                self.T = np.zeros((mesh.ny, mesh.nx, 1, 1))
                self.x_nodes = mesh.X_centers.reshape(mesh.ny, mesh.nx, 1, 1)
                self.y_nodes = mesh.Y_centers.reshape(mesh.ny, mesh.nx, 1, 1)
                
        mock_solver = MockFRSolver(fv_solver.mesh)
        
        # Apply FR initial condition
        self.fr_plugin.set(mock_solver)
        
        # Extract cell values
        T_cells = mock_solver.T[:, :, 0, 0]
        
        # Set on FV solver
        fv_solver.set_initial_condition_from_array(T_cells)