# image_tools.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.patches import Rectangle
import cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
import fr_heat_sim.fr_utils as fr_utils

def min_distance_to_mask(i, j, mask):
    if mask[i, j]:
        return 0
    max_dist = 10
    for d in range(1, max_dist + 1):
        for di in range(-d, d + 1):
            for dj in [-d, d]:
                ni, nj = i + di, j + dj
                if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj]:
                    return d
        for dj in range(-d + 1, d):
            for di in [-d, d]:
                ni, nj = i + di, j + dj
                if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj]:
                    return d
    return max_dist

class DomainSelector:
    """Interactive domain selection by clicking 4 corners."""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.corners = []
        self.domain_bounds = None
        
    def select_domain(self):
        """Interactive selection of domain corners."""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(self.img_rgb)
        ax.set_title('Click 4 corners of the domain (excluding axes and colorbar)\n'
                    'Click: Top-left, Top-right, Bottom-right, Bottom-left', 
                    fontsize=14, color='red')
        
        self.corners = []
        self.markers = []
        
        def on_click(event):
            if event.inaxes and len(self.corners) < 4:
                x, y = int(event.xdata), int(event.ydata)
                self.corners.append((x, y))
                
                marker = ax.plot(x, y, 'ro', markersize=10)[0]
                self.markers.append(marker)
                
                corner_names = ['Top-left', 'Top-right', 'Bottom-right', 'Bottom-left']
                ax.text(x+10, y+10, corner_names[len(self.corners)-1], 
                       color='red', fontsize=10, weight='bold')
                
                if len(self.corners) == 4:
                    x_coords = [c[0] for c in self.corners]
                    y_coords = [c[1] for c in self.corners]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    
                    rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='lime', linewidth=3)
                    ax.add_patch(rect)
                    
                    ax.set_title('Domain selected! Close window to continue.', 
                               fontsize=14, color='green')
                    
                    self.domain_bounds = {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': x2 - x1, 'height': y2 - y1
                    }
                
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        
        return self.domain_bounds

class TemperatureCalibrator:
    """Temperature scale calibration functionality."""
    
    def __init__(self, image_path, domain_bounds=None):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.domain_bounds = domain_bounds
        
        self.colorbar_box = None
        self.temp_min_value = None
        self.temp_max_value = None
        self.num_levels = None
        self.calibration_complete = False
        self.temperature_mapping = None
        
    def calibrate(self):
        """Run the calibration workflow."""
        self.step1_draw_colorbar_box()
        
        if self.colorbar_box:
            self.step2_mark_temperatures()
            
            if self.calibration_complete:
                self.temperature_mapping = self.create_temperature_mapping()
                return self.temperature_mapping
        
        return None
    
    def step1_draw_colorbar_box(self):
        """Step 1: Draw box around colorbar scale."""
        while True:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(self.img_rgb)
            
            if self.domain_bounds:
                domain_rect = Rectangle((self.domain_bounds['x1'], self.domain_bounds['y1']), 
                                      self.domain_bounds['width'], self.domain_bounds['height'],
                                      fill=False, edgecolor='blue', linewidth=2, linestyle='--')
                ax.add_patch(domain_rect)
            
            ax.set_title('Draw a box around the COLOR SCALE ONLY (no text/empty space)\n'
                        'The colorbar should be OUTSIDE the blue domain box', 
                        fontsize=14, color='red')
            
            self.rect = None
            self.box_drawn = False
            self.box_start = None
            
            def on_press(event):
                if event.inaxes:
                    self.box_start = (int(event.xdata), int(event.ydata))
            
            def on_drag(event):
                if event.inaxes and self.box_start:
                    x1, y1 = self.box_start
                    x2, y2 = int(event.xdata), int(event.ydata)
                    
                    if self.rect:
                        self.rect.remove()
                    
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    x = min(x1, x2)
                    y = min(y1, y2)
                    
                    self.rect = Rectangle((x, y), width, height, 
                                        fill=False, edgecolor='lime', linewidth=3)
                    ax.add_patch(self.rect)
                    fig.canvas.draw_idle()
            
            def on_release(event):
                if event.inaxes and self.box_start:
                    x1, y1 = self.box_start
                    x2, y2 = int(event.xdata), int(event.ydata)
                    
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    
                    if width < 5 or height < 5:
                        return
                    
                    self.colorbar_box = {
                        'x1': min(x1, x2),
                        'y1': min(y1, y2),
                        'x2': max(x1, x2),
                        'y2': max(y1, y2)
                    }
                    
                    self.box_drawn = True
                    ax.set_title('Colorbar selected! Close window to proceed', 
                               fontsize=14, color='green')
                    fig.canvas.draw_idle()
            
            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.mpl_connect('motion_notify_event', on_drag)
            fig.canvas.mpl_connect('button_release_event', on_release)
            
            plt.show()
            
            if self.box_drawn and self.colorbar_box:
                self.colorbar_region = self.img_rgb[
                    self.colorbar_box['y1']:self.colorbar_box['y2'],
                    self.colorbar_box['x1']:self.colorbar_box['x2']
                ]
                
                if self.confirm_colorbar_selection():
                    break
                else:
                    print("Redrawing colorbar box...")
                    self.colorbar_box = None
                    self.colorbar_region = None
    
    def confirm_colorbar_selection(self):
        """Show the selected colorbar and ask for confirmation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(self.img_rgb)
        rect = Rectangle((self.colorbar_box['x1'], self.colorbar_box['y1']), 
                        self.colorbar_box['x2'] - self.colorbar_box['x1'],
                        self.colorbar_box['y2'] - self.colorbar_box['y1'],
                        fill=False, edgecolor='lime', linewidth=3)
        ax1.add_patch(rect)
        ax1.set_title('Full Image with Selected Region')
        
        ax2.imshow(self.colorbar_region)
        ax2.set_title('Extracted Colorbar Region')
        ax2.axis('off')
        
        fig.suptitle('Is this the correct colorbar region?', fontsize=16, color='blue')
        
        self.confirmed = None
        
        ax_yes = plt.axes([0.35, 0.02, 0.1, 0.04])
        ax_no = plt.axes([0.55, 0.02, 0.1, 0.04])
        
        btn_yes = Button(ax_yes, 'Yes', color='lightgreen')
        btn_no = Button(ax_no, 'No', color='lightcoral')
        
        def on_yes(event):
            self.confirmed = True
            plt.close(fig)
        
        def on_no(event):
            self.confirmed = False
            plt.close(fig)
        
        btn_yes.on_clicked(on_yes)
        btn_no.on_clicked(on_no)
        
        plt.show()
        
        return self.confirmed
    
    def step2_mark_temperatures(self):
        """Step 2: Mark temperature endpoints and input values."""
        if self.colorbar_region is None:
            print("No colorbar region selected!")
            return
        
        fig = plt.figure(figsize=(14, 10))
        
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(self.img_rgb)
        rect = Rectangle((self.colorbar_box['x1'], self.colorbar_box['y1']), 
                        self.colorbar_box['x2'] - self.colorbar_box['x1'],
                        self.colorbar_box['y2'] - self.colorbar_box['y1'],
                        fill=False, edgecolor='lime', linewidth=3)
        ax1.add_patch(rect)
        ax1.set_title('Full Image with Selected Colorbar')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(self.colorbar_region)
        ax2.set_title('Step 2: Click on MIN temperature position (bottom/left of scale)', 
                     fontsize=12, color='blue')
        
        self.points = []
        self.point_markers = []
        
        def on_click(event):
            if event.inaxes == ax2:
                x, y = int(event.xdata), int(event.ydata)
                
                if len(self.points) == 0:
                    self.points.append(('min', x, y))
                    marker = ax2.plot(x, y, 'bo', markersize=10, label='MIN')[0]
                    self.point_markers.append(marker)
                    ax2.set_title('Now click on MAX temperature position (top/right of scale)', 
                                fontsize=12, color='red')
                    ax2.legend()
                    
                elif len(self.points) == 1:
                    self.points.append(('max', x, y))
                    marker = ax2.plot(x, y, 'ro', markersize=10, label='MAX')[0]
                    self.point_markers.append(marker)
                    ax2.set_title('Temperature endpoints marked. Close window to continue.', 
                                fontsize=12, color='green')
                    ax2.legend()
                    
                    min_x, min_y = self.points[0][1], self.points[0][2]
                    max_x, max_y = self.points[1][1], self.points[1][2]
                    ax2.plot([min_x, max_x], [min_y, max_y], 'g--', linewidth=2)
                
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        
        if len(self.points) == 2:
            self.get_temperature_values()
    
    def get_temperature_values(self):
        """Get temperature values and number of levels from user."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        ax.text(0.5, 0.9, 'Enter Temperature Scale Information', 
               transform=ax.transAxes, ha='center', fontsize=16, weight='bold')
        
        axbox_min = plt.axes([0.3, 0.7, 0.4, 0.05])
        text_min = TextBox(axbox_min, 'Min Temp:', initial='0')
        
        axbox_max = plt.axes([0.3, 0.6, 0.4, 0.05])
        text_max = TextBox(axbox_max, 'Max Temp:', initial='6000')
        
        axbox_levels = plt.axes([0.3, 0.5, 0.4, 0.05])
        text_levels = TextBox(axbox_levels, 'Contour Levels:', initial='10')
        
        self.temp_values = {'min': 0, 'max': 6000, 'levels': 10}
        
        def update_min(text):
            try:
                self.temp_values['min'] = float(text)
            except:
                pass
        
        def update_max(text):
            try:
                self.temp_values['max'] = float(text)
            except:
                pass
        
        def update_levels(text):
            try:
                self.temp_values['levels'] = int(text)
            except:
                pass
        
        text_min.on_submit(update_min)
        text_max.on_submit(update_max)
        text_levels.on_submit(update_levels)
        
        axbutton = plt.axes([0.4, 0.3, 0.2, 0.075])
        btn = Button(axbutton, 'OK')
        
        def on_ok(event):
            self.temp_min_value = self.temp_values['min']
            self.temp_max_value = self.temp_values['max']
            self.num_levels = self.temp_values['levels']
            self.calibration_complete = True
            plt.close(fig)
        
        btn.on_clicked(on_ok)
        
        plt.show()
    
    def create_temperature_mapping(self):
        """Create the final temperature mapping based on calibration."""
        if not self.calibration_complete:
            return None
        
        min_x, min_y = self.points[0][1], self.points[0][2]
        max_x, max_y = self.points[1][1], self.points[1][2]
        
        dx = max_x - min_x
        dy = max_y - min_y
        
        if abs(dy) > abs(dx):
            orientation = 'vertical'
            scale_length = abs(dy)
        else:
            orientation = 'horizontal'
            scale_length = abs(dx)
        
        num_samples = 100
        color_samples = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            
            sample_x = int(min_x + t * dx)
            sample_y = int(min_y + t * dy)
            
            color = self.colorbar_region[sample_y, sample_x]
            temp = self.temp_min_value + t * (self.temp_max_value - self.temp_min_value)
            
            color_samples.append({
                'position': t,
                'color': color,
                'temperature': temp,
                'pixel_coords': (sample_x, sample_y)
            })
        
        temperature_mapping = {
            'colorbar_box': self.colorbar_box,
            'colorbar_region': self.colorbar_region,
            'orientation': orientation,
            'scale_length': scale_length,
            'temp_range': (self.temp_min_value, self.temp_max_value),
            'num_levels': self.num_levels,
            'color_samples': color_samples,
            'min_point': (min_x, min_y),
            'max_point': (max_x, max_y),
            'domain_bounds': self.domain_bounds
        }
        
        return temperature_mapping


class TemperatureRegionToFRGrid:
    """Map temperature region from image to FR grid."""
    
    def __init__(self, image_path, nx_elem, ny_elem, p, plate_length=None, plate_width=None,
                 interactive=True, domain_size_microns=None):
        """
        Initialize the mapper.
        
        Args:
            image_path: Path to the temperature image
            nx_elem: Number of FR elements in x-direction
            ny_elem: Number of FR elements in y-direction
            p: Polynomial order
            plate_length: Physical domain length (m) - if None, will be set from image
            plate_width: Physical domain width (m) - if None, will be set from image
            interactive: If True, use interactive domain selection and calibration
            domain_size_microns: If provided, use this X dimension in microns
        """
        self.image_path = image_path
        self.nx_elem = nx_elem
        self.ny_elem = ny_elem
        self.p = p
        self.p1 = p + 1
        
        print("=== Temperature Region to FR Grid Mapping ===\n")
        
        if interactive:
            # Step 1: Interactive domain selection
            print("Step 1: Select domain boundaries (4 corners)")
            domain_selector = DomainSelector(image_path)
            self.domain_bounds = domain_selector.select_domain()
            
            if not self.domain_bounds:
                raise ValueError("Domain selection failed or was cancelled.")
            
            # Calculate domain dimensions in pixels
            self.X_image_pixels = self.domain_bounds['width']
            self.Y_image_pixels = self.domain_bounds['height']
            
            print(f"\nDomain selected: ({self.domain_bounds['x1']}, {self.domain_bounds['y1']}) to "
                  f"({self.domain_bounds['x2']}, {self.domain_bounds['y2']})")
            print(f"Domain size: {self.X_image_pixels} × {self.Y_image_pixels} pixels")
            
            # Step 2: Temperature calibration (optional)
            print("\nStep 2: Calibrate temperature scale (optional)")
            response = input("Do you want to calibrate the temperature scale? (y/n): ")
            
            if response.lower() == 'y':
                calibrator = TemperatureCalibrator(image_path, self.domain_bounds)
                self.temperature_mapping = calibrator.calibrate()
                
                if self.temperature_mapping:
                    print("Temperature calibration completed successfully!")
                else:
                    print("Temperature calibration failed or was cancelled.")
                    self.temperature_mapping = None
            else:
                self.temperature_mapping = None
                print("Skipping temperature calibration.")
        else:
            # Non-interactive mode: use full image as domain
            img = cv2.imread(image_path)
            self.domain_bounds = {
                'x1': 0, 'y1': 0,
                'x2': img.shape[1], 'y2': img.shape[0],
                'width': img.shape[1], 'height': img.shape[0]
            }
            self.X_image_pixels = self.domain_bounds['width']
            self.Y_image_pixels = self.domain_bounds['height']
            self.temperature_mapping = None
        
        # Step 3: Set up grid dimensions
        print("\nStep 3: Set up FR grid dimensions")
        
        if domain_size_microns is not None:
            # Use provided domain size
            self.X_grid = domain_size_microns
        else:
            # Get user input for grid dimensions
            X_grid_input = input(f"Enter X grid dimension in microns (default 800): ")
            self.X_grid = float(X_grid_input) if X_grid_input else 800.0
        
        # Calculate Y grid to maintain aspect ratio
        self.Y_grid = self.X_grid * (self.Y_image_pixels / self.X_image_pixels)
        
        # Set plate dimensions in meters based on grid dimensions
        self.plate_length = self.X_grid * 1e-6  # Convert microns to meters
        self.plate_width = self.Y_grid * 1e-6   # Convert microns to meters
        
        # Conversion factors
        self.X_ratio = self.X_grid / self.X_image_pixels  # microns/pixel
        self.Y_ratio = self.Y_grid / self.Y_image_pixels  # microns/pixel
        
        print(f"\nGrid dimensions: {self.X_grid:.1f} × {self.Y_grid:.1f} microns")
        print(f"Domain dimensions: {self.plate_length*1e3:.3f} × {self.plate_width*1e3:.3f} mm")
        print(f"Conversion ratios: X = {self.X_ratio:.3f} μm/pixel, Y = {self.Y_ratio:.3f} μm/pixel")
        
        # Load image and extract domain
        self.load_and_extract_domain()
        
        # Setup FR grid
        self.setup_fr_grid()
    
    def load_and_extract_domain(self):
        """Load image and extract the domain region."""
        # Read image
        img = cv2.imread(self.image_path)
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract domain region
        self.domain_img = self.img_rgb[
            self.domain_bounds['y1']:self.domain_bounds['y2'],
            self.domain_bounds['x1']:self.domain_bounds['x2']
        ]
        
        # Convert to grayscale for mask creation
        gray = cv2.cvtColor(self.domain_img, cv2.COLOR_RGB2GRAY)
        
        # Create mask for temperature region (non-white areas)
        self.temp_mask = gray < 245
        
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        self.temp_mask = cv2.morphologyEx(self.temp_mask.astype(np.uint8) * 255, 
                                         cv2.MORPH_CLOSE, kernel)
        self.temp_mask = cv2.morphologyEx(self.temp_mask, cv2.MORPH_OPEN, kernel)
        self.temp_mask = self.temp_mask > 127
        
        # Create a separate mask for actual temperature data (excluding white regions)
        self.data_mask = np.zeros_like(self.temp_mask)
        for i in range(self.domain_img.shape[0]):
            for j in range(self.domain_img.shape[1]):
                if self.temp_mask[i, j]:
                    pixel = self.domain_img[i, j]
                    # Only include non-white pixels as actual data
                    if not np.all(pixel > 240):
                        self.data_mask[i, j] = True
        
    def setup_fr_grid(self):
        """Setup the FR grid with Gauss-Legendre nodes."""
        # Element sizes in meters (not microns)
        self.dx = self.plate_length / self.nx_elem
        self.dy = self.plate_width / self.ny_elem
        
        # Get Gauss-Legendre nodes
        self.r_nodes, self.weights = fr_utils.get_gauss_legendre_nodes(self.p)
        
        # Element boundaries in meters
        self.x_faces = np.linspace(0, self.plate_length, self.nx_elem + 1)
        self.y_faces = np.linspace(0, self.plate_width, self.ny_elem + 1)
        
        # Physical coordinates of solution points in meters
        self.x_nodes = np.zeros((self.ny_elem, self.nx_elem, self.p1, self.p1))
        self.y_nodes = np.zeros((self.ny_elem, self.nx_elem, self.p1, self.p1))
        
        for ey in range(self.ny_elem):
            for ex in range(self.nx_elem):
                xL, xR = self.x_faces[ex], self.x_faces[ex + 1]
                yB, yT = self.y_faces[ey], self.y_faces[ey + 1]
                
                for j in range(self.p1):
                    for i in range(self.p1):
                        self.x_nodes[ey, ex, j, i] = (
                            0.5 * (xL + xR) + 0.5 * (xR - xL) * self.r_nodes[i]
                        )
                        self.y_nodes[ey, ex, j, i] = (
                            0.5 * (yB + yT) + 0.5 * (yT - yB) * self.r_nodes[j]
                        )
    
    def color_to_temperature(self, pixel_color, temperature_mapping):
        """Map a color to temperature using calibrated mapping."""
        min_distance = float('inf')
        best_temp = temperature_mapping['temp_range'][0]
        
        for sample in temperature_mapping['color_samples']:
            distance = np.linalg.norm(pixel_color - sample['color'])
            
            if distance < min_distance:
                min_distance = distance
                best_temp = sample['temperature']
        
        return best_temp
    
    def map_temperature_to_grid(self, T_background=300.0, use_constant_temp=False, constant_temp=1000.0,
                        smooth_interface=True, smooth_width=2):
        """
        Map the temperature region onto the FR grid.
        
        Args:
            T_background: Temperature outside the region (K)
            use_constant_temp: If True, use constant temperature for the entire region
            constant_temp: Constant temperature value to use if use_constant_temp is True
            smooth_interface: If True, smooth the temperature interface to reduce Gibbs oscillations
            smooth_width: Width of smoothing in pixels (default 2)
        
        Returns:
            T: Temperature field on FR grid nodes
        """
        # Initialize temperature array with the specified background temperature
        T = np.ones((self.ny_elem, self.nx_elem, self.p1, self.p1)) * T_background
        
        # Create temperature map based on user choice
        if use_constant_temp:
            # Use constant temperature for the entire region
            temp_map = np.where(self.temp_mask, constant_temp, T_background)
            
            # Smooth the interface to reduce Gibbs oscillations
            if smooth_interface:
                from scipy.ndimage import gaussian_filter
                # Convert to float for proper smoothing
                temp_map = temp_map.astype(float)
                
                # Apply smoothing multiple times for better transition
                for _ in range(2):
                    temp_map = gaussian_filter(temp_map, sigma=smooth_width)
                
                print(f"Applied interface smoothing (sigma={smooth_width} pixels, 2 passes)")
            
            print(f"Using constant temperature of {constant_temp} K for the region")
            print(f"Using background temperature of {T_background} K outside the region")
        else:
            # Use actual temperature distribution from calibration
            if self.temperature_mapping:
                # Create temperature map from calibration with proper background
                temp_map = self.create_temperature_map(T_background, smooth_interface=smooth_interface, 
                                                    smooth_width=smooth_width)
                print("Using calibrated temperature distribution from image")
                print(f"Using background temperature of {T_background} K outside the region")
                if smooth_interface:
                    print(f"Applied interface smoothing (sigma={smooth_width} pixels)")
            else:
                # Use default hot temperature if no calibration
                temp_map = np.where(self.temp_mask, 1000.0, T_background)
                if smooth_interface:
                    from scipy.ndimage import gaussian_filter
                    temp_map = temp_map.astype(float)
                    # Apply smoothing multiple times
                    for _ in range(2):
                        temp_map = gaussian_filter(temp_map, sigma=smooth_width)
                    print(f"Applied interface smoothing (sigma={smooth_width} pixels, 2 passes)")
                print("Using default temperature of 1000 K (no calibration available)")
                print(f"Using background temperature of {T_background} K outside the region")
        
        # Apply clipping to ensure no temperatures below background after smoothing
        temp_map = np.maximum(temp_map, T_background)
        
        # Create interpolator for the temperature field using RegularGridInterpolator
        # Image pixel coordinates (note: y is first dimension in image arrays)
        y_img = np.arange(self.domain_img.shape[0])
        x_img = np.arange(self.domain_img.shape[1])
        
        # Create interpolator with proper axis order
        # Use linear interpolation instead of cubic to avoid overshoots
        interp_func = RegularGridInterpolator(
            (y_img, x_img), temp_map, 
            method='linear',  # Linear to avoid overshoots
            bounds_error=False, 
            fill_value=T_background  # Use specified background temperature
        )
        
        # Map to each FR grid node
        for ey in range(self.ny_elem):
            for ex in range(self.nx_elem):
                for j in range(self.p1):
                    for i in range(self.p1):
                        # Get physical coordinates in meters
                        x_phys = self.x_nodes[ey, ex, j, i]
                        y_phys = self.y_nodes[ey, ex, j, i]
                        
                        # Convert to microns then to image pixel coordinates
                        x_img_coord = (x_phys * 1e6) / self.X_ratio
                        y_img_coord = (y_phys * 1e6) / self.Y_ratio
                        
                        # Interpolate temperature
                        if (0 <= x_img_coord < self.domain_img.shape[1] and 
                            0 <= y_img_coord < self.domain_img.shape[0]):
                            # Note: RegularGridInterpolator expects (y, x) order
                            T[ey, ex, j, i] = interp_func([y_img_coord, x_img_coord])[0]
                        else:
                            # Outside the image domain, use background temperature
                            T[ey, ex, j, i] = T_background
        
        # Apply multiple levels of safety to prevent incorrect temperatures
        
        # 1. Hard floor at 0 K (absolute zero)
        T = np.maximum(T, 0.0)
        
        # 2. Ensure background temperature is maintained
        # Any temperature significantly below background in non-hotspot regions is suspicious
        T = np.maximum(T, T_background * 0.99)  # Allow only 1% below background
        
        # 3. Check for outliers and clip them
        T_median = np.median(T)
        T_std = np.std(T)
        
        # Conservative clipping for the lower bound
        T_lower = T_background * 0.99
        T_upper = T_median + 10*T_std
        
        if use_constant_temp:
            # For constant temperature case, we know the expected range
            T_upper = min(T_upper, constant_temp * 1.1)  # Allow 10% overshoot
        else:
            # For calibrated case, be more permissive but still reasonable
            if self.temperature_mapping:
                T_max_calib = self.temperature_mapping['temp_range'][1]
                T_upper = min(T_upper, T_max_calib * 1.1)
            else:
                T_upper = min(T_upper, 10000.0)
        
        T = np.clip(T, T_lower, T_upper)
        
        # 4. Final check: ensure no temperature below background
        T = np.maximum(T, T_background)
        
        # Print statistics for debugging
        print(f"\nTemperature field statistics:")
        print(f"  Min: {np.min(T):.1f} K")
        print(f"  Max: {np.max(T):.1f} K")
        print(f"  Mean: {np.mean(T):.1f} K")
        print(f"  Std: {np.std(T):.1f} K")
        
        if np.min(T) < T_background * 0.99:
            print(f"  WARNING: Minimum temperature ({np.min(T):.1f} K) is below 99% of background!")
        
        return T
    
    def create_temperature_map(self, T_background=300.0, smooth_interface=True, smooth_width=2):
        """Create temperature map from image using calibration.
        
        Args:
            T_background: Temperature to use outside the region (K)
            smooth_interface: If True, smooth the temperature interface
            smooth_width: Width of smoothing in pixels
        """
        # Initialize with the specified background temperature (not calibrated minimum)
        temp_map = np.ones(self.domain_img.shape[:2]) * T_background
        
        print(f"\nDEBUG: create_temperature_map called with T_background={T_background}")
        print(f"DEBUG: temp_mask shape: {self.temp_mask.shape}, True pixels: {np.sum(self.temp_mask)}")
        
        if self.temperature_mapping:
            print(f"DEBUG: Using calibration with range: {self.temperature_mapping['temp_range']}")
            
            # Map colors to temperatures only inside the temperature region
            pixels_mapped = 0
            for i in range(self.domain_img.shape[0]):
                for j in range(self.domain_img.shape[1]):
                    if self.temp_mask[i, j]:
                        pixel = self.domain_img[i, j]
                        
                        # Check if pixel is white/near-white (empty region)
                        if np.all(pixel > 240):
                            # This pixel is white - should it be background or calibrated min?
                            # For regions INSIDE the mask but white, use background
                            temp_map[i, j] = T_background
                        else:
                            # This is actual temperature data
                            temp = self.color_to_temperature(pixel, self.temperature_mapping)
                            temp_map[i, j] = temp
                            pixels_mapped += 1
                    # else: pixel is outside mask, already set to T_background
            
            print(f"DEBUG: Mapped {pixels_mapped} pixels with calibrated temperatures")
            print(f"DEBUG: Before smoothing - Min: {np.min(temp_map):.1f}, Max: {np.max(temp_map):.1f}")
        
        # Apply smoothing if requested
        if smooth_interface:
            from scipy.ndimage import gaussian_filter
            temp_map = gaussian_filter(temp_map.astype(float), sigma=smooth_width)
            
            # After smoothing, we need to ensure regions far from the hotspot stay at background
            # This is crucial! Smoothing can spread high temperatures into background regions
            
            # Create a distance map from the mask
            from scipy.ndimage import distance_transform_edt
            distance_from_region = distance_transform_edt(~self.temp_mask)
            
            # Restore background temperature for pixels far from the region
            # Pixels more than 3*smooth_width away should be at background
            far_from_region = distance_from_region > (3 * smooth_width)
            temp_map[far_from_region] = T_background
            
            print(f"DEBUG: After smoothing - Min: {np.min(temp_map):.1f}, Max: {np.max(temp_map):.1f}")
            print(f"DEBUG: Pixels reset to background: {np.sum(far_from_region)}")
            
        return temp_map
    
    def analyze_region_geometry(self):
        """Analyze the geometry of the temperature region and calculate effective radii."""
        from scipy import ndimage
        from skimage import measure
        
        # Get the temperature region mask
        mask = self.temp_mask.astype(np.uint8)
        
        # Calculate area (number of pixels)
        area_pixels = np.sum(mask)
        
        # Calculate area in physical units (square microns)
        pixel_area = self.X_ratio * self.Y_ratio  # μm²/pixel
        area_microns2 = area_pixels * pixel_area
        
        # Calculate perimeter using contour detection
        contours = measure.find_contours(mask, 0.5)
        perimeter_pixels = 0
        for contour in contours:
            # Calculate perimeter of each contour by summing distances between consecutive points
            diffs = np.diff(contour, axis=0)
            perimeter_pixels += np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
        
        # Convert perimeter to microns
        # Average pixel size for perimeter calculation
        avg_pixel_size = (self.X_ratio + self.Y_ratio) / 2
        perimeter_microns = perimeter_pixels * avg_pixel_size
        
        # Calculate effective radii
        # From area: A = πr² → r = √(A/π)
        radius_from_area = np.sqrt(area_microns2 / np.pi)
        
        # From perimeter: P = 2πr → r = P/(2π)
        radius_from_perimeter = perimeter_microns / (2 * np.pi)
        
        # Calculate compactness (circularity) metric
        # For a perfect circle, this ratio should be 1.0
        if perimeter_microns > 0:
            compactness = (4 * np.pi * area_microns2) / (perimeter_microns ** 2)
        else:
            compactness = 0
        
        # Store results
        self.region_geometry = {
            'area_pixels': area_pixels,
            'area_microns2': area_microns2,
            'area_mm2': area_microns2 * 1e-6,
            'perimeter_pixels': perimeter_pixels,
            'perimeter_microns': perimeter_microns,
            'perimeter_mm': perimeter_microns * 1e-3,
            'radius_from_area_microns': radius_from_area,
            'radius_from_area_mm': radius_from_area * 1e-3,
            'radius_from_perimeter_microns': radius_from_perimeter,
            'radius_from_perimeter_mm': radius_from_perimeter * 1e-3,
            'compactness': compactness,
            'contours': contours
        }
        
        return self.region_geometry
    
    def print_region_geometry(self):
        """Print the geometric analysis of the temperature region."""
        if not hasattr(self, 'region_geometry'):
            self.analyze_region_geometry()
        
        g = self.region_geometry
        
        print("\n=== Temperature Region Geometry Analysis ===")
        print(f"Area:")
        print(f"  Pixels: {g['area_pixels']:,}")
        print(f"  Physical: {g['area_microns2']:.1f} μm² ({g['area_mm2']:.4f} mm²)")
        
        print(f"\nPerimeter:")
        print(f"  Pixels: {g['perimeter_pixels']:.1f}")
        print(f"  Physical: {g['perimeter_microns']:.1f} μm ({g['perimeter_mm']:.3f} mm)")
        
        print(f"\nEffective Radii:")
        print(f"  From area (A = πr²):")
        print(f"    r = {g['radius_from_area_microns']:.1f} μm ({g['radius_from_area_mm']:.3f} mm)")
        print(f"  From perimeter (P = 2πr):")
        print(f"    r = {g['radius_from_perimeter_microns']:.1f} μm ({g['radius_from_perimeter_mm']:.3f} mm)")
        
        print(f"\nShape Analysis:")
        print(f"  Compactness (circularity): {g['compactness']:.3f}")
        print(f"  (1.0 = perfect circle, <1.0 = irregular shape)")
        
        # Compare the two radii
        radius_ratio = g['radius_from_area_microns'] / g['radius_from_perimeter_microns']
        print(f"\n  Radius ratio (area/perimeter): {radius_ratio:.3f}")
        if radius_ratio < 0.9:
            print("  → Region is more irregular/elongated than a circle")
        elif radius_ratio > 1.1:
            print("  → Region has protrusions or complex boundaries")
        else:
            print("  → Region is approximately circular")
        
        # Suggest equivalent circular hotspot
        avg_radius = (g['radius_from_area_microns'] + g['radius_from_perimeter_microns']) / 2
        print(f"\nEquivalent circular hotspot radius: {avg_radius:.1f} μm ({avg_radius*1e-3:.3f} mm)")

