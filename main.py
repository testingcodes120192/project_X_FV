# main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import configparser
import os
import sys

# Import our FV modules
from mesh import FVMesh
from solver import FVHeatSolver
from postprocessor import FVPostProcessor
from initial_conditions import get_fv_initial_condition
from animation import create_fv_animation, plot_fv_solution_snapshots
from simple_amr import SimpleAMR

class FVHeatSimulatorGUI:
    """
    GUI for Finite Volume Heat Diffusion Simulator.
    
    Simplified from FR version, focusing on FV-specific parameters.
    """
    
    def __init__(self, master):
        self.master = master
        master.title("FV Heat Diffusion Simulator")
        master.geometry("800x900")
        
        # Variables
        self.setup_variables()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_simulation_tab()
        self.create_grid_tab()
        self.create_initial_condition_tab()
        self.create_numerical_tab()
        self.create_reaction_tab()
        self.create_amr_tab()
        self.create_visualization_tab()
        
        # Run button
        self.run_button = ttk.Button(master, text="Run Simulation", 
                                    command=self.run_simulation,
                                    style='Accent.TButton')
        self.run_button.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(master, mode='indeterminate')
        self.progress.pack(fill='x', padx=10, pady=5)
        
        # Status label
        self.status_label = ttk.Label(master, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(fill='x', padx=10, pady=5)
        
        # Load default configuration
        self.load_config('config_fv.ini')
        
    def setup_variables(self):
        """Initialize all GUI variables."""
        # Simulation
        self.var_total_time = tk.DoubleVar(value=10.0)
        self.var_time_points = tk.StringVar(value="0.0,1.0,5.0,10.0")
        self.var_dt = tk.StringVar(value="")
        self.var_alpha = tk.DoubleVar(value=9.7e-5)
        self.var_cfl = tk.DoubleVar(value=0.5)
        
        # Grid
        self.var_nx_cells = tk.IntVar(value=100)
        self.var_ny_cells = tk.IntVar(value=100)
        self.var_plate_length = tk.DoubleVar(value=0.5)
        self.var_plate_width = tk.DoubleVar(value=0.5)
        
        # Numerical
        self.var_spatial_order = tk.IntVar(value=5)
        self.var_time_integration = tk.StringVar(value='RK3')
        
        # Initial condition
        self.var_ic_type = tk.StringVar(value='circular')
        self.var_background_temp = tk.DoubleVar(value=300.0)
        self.var_hotspot_temp = tk.DoubleVar(value=6000.0)
        self.var_center_x = tk.DoubleVar(value=0.25)
        self.var_center_y = tk.DoubleVar(value=0.25)
        self.var_hotspot_radius = tk.DoubleVar(value=0.05)
        self.var_smooth_transition = tk.BooleanVar(value=True)
        self.var_transition_width = tk.DoubleVar(value=0.005)
        self.var_image_path = tk.StringVar(value="")
        self.var_use_constant_temp = tk.BooleanVar(value=True)
        self.var_constant_temp = tk.DoubleVar(value=1000.0)
        
        # Reactions
        self.var_enable_reactions = tk.BooleanVar(value=False)
        self.var_reaction_model = tk.StringVar(value='arrhenius')
        self.var_reaction_A = tk.DoubleVar(value=1.5e15)
        self.var_reaction_Ea = tk.DoubleVar(value=180000)
        self.var_reaction_n = tk.DoubleVar(value=1.0)
        self.var_reaction_Q = tk.DoubleVar(value=2.5e6)
        
        # AMR
        self.var_enable_amr = tk.BooleanVar(value=False)
        self.var_max_levels = tk.IntVar(value=3)
        self.var_refinement_ratio = tk.IntVar(value=2)
        self.var_refine_threshold = tk.DoubleVar(value=100.0)
        self.var_coarsen_threshold = tk.DoubleVar(value=10.0)
        
        # Visualization
        self.var_nx_plot = tk.IntVar(value=101)
        self.var_ny_plot = tk.IntVar(value=101)
        self.var_create_animation = tk.BooleanVar(value=False)
        self.var_frame_skip = tk.IntVar(value=10)
        self.var_save_animation = tk.BooleanVar(value=True)
        self.var_T_min_fixed = tk.DoubleVar(value=300.0)
        self.var_T_max_fixed = tk.DoubleVar(value=6000.0)
        
    def create_simulation_tab(self):
        """Create simulation settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Simulation")
        
        # Time settings
        ttk.Label(tab, text="Simulation Time Settings", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Total Time (s):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_total_time, width=15).grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text="Output Times:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_time_points, width=30).grid(row=2, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text="Time Step (s):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_dt, width=15).grid(row=3, column=1, sticky='w', padx=5)
        ttk.Label(tab, text="(leave empty for auto)").grid(row=3, column=2, sticky='w')
        
        # Physical parameters
        ttk.Label(tab, text="Physical Parameters", font=('TkDefaultFont', 10, 'bold')).grid(
            row=4, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Thermal Diffusivity (m²/s):").grid(row=5, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_alpha, width=15).grid(row=5, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text="CFL Number:").grid(row=6, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_cfl, width=15).grid(row=6, column=1, sticky='w', padx=5)
        
    def create_grid_tab(self):
        """Create grid settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Grid")
        
        ttk.Label(tab, text="Grid Resolution", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Number of Cells (X):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_nx_cells, width=15).grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text="Number of Cells (Y):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_ny_cells, width=15).grid(row=2, column=1, sticky='w', padx=5)
        
        # Resolution info
        self.resolution_label = ttk.Label(tab, text="", font=('TkDefaultFont', 9, 'italic'))
        self.resolution_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Update resolution info when values change
        self.var_nx_cells.trace('w', self.update_resolution_info)
        self.var_ny_cells.trace('w', self.update_resolution_info)
        
        ttk.Label(tab, text="Domain Size", font=('TkDefaultFont', 10, 'bold')).grid(
            row=4, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Plate Length (m):").grid(row=5, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_plate_length, width=15).grid(row=5, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text="Plate Width (m):").grid(row=6, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_plate_width, width=15).grid(row=6, column=1, sticky='w', padx=5)
        
    def create_numerical_tab(self):
        """Create numerical methods tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Numerical")
        
        ttk.Label(tab, text="Spatial Discretization", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Spatial Order:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        order_frame = ttk.Frame(tab)
        order_frame.grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Radiobutton(order_frame, text="1st", variable=self.var_spatial_order, 
                       value=1).pack(side='left')
        ttk.Radiobutton(order_frame, text="2nd", variable=self.var_spatial_order, 
                       value=2).pack(side='left')
        ttk.Radiobutton(order_frame, text="5th (WENO)", variable=self.var_spatial_order, 
                       value=5).pack(side='left')
        
        ttk.Label(tab, text="Time Integration", font=('TkDefaultFont', 10, 'bold')).grid(
            row=2, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Method:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        time_frame = ttk.Frame(tab)
        time_frame.grid(row=3, column=1, sticky='w', padx=5)
        
        ttk.Radiobutton(time_frame, text="Euler", variable=self.var_time_integration, 
                       value='Euler').pack(side='left')
        ttk.Radiobutton(time_frame, text="RK3", variable=self.var_time_integration, 
                       value='RK3').pack(side='left')
        
        # Info about methods
        info_text = """
Spatial Orders:
- 1st: Simple, robust, diffusive
- 2nd: Good balance of accuracy/cost
- 5th (WENO): High accuracy for smooth solutions

Time Integration:
- Euler: Simple, requires small time steps
- RK3: More stable, allows larger time steps
        """
        info_label = ttk.Label(tab, text=info_text, justify='left', 
                              font=('TkDefaultFont', 9))
        info_label.grid(row=4, column=0, columnspan=2, pady=20, padx=10)
        
    def create_initial_condition_tab(self):
        """Create initial condition tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Initial Condition")
        
        ttk.Label(tab, text="Initial Condition Type", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Type:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        ic_combo = ttk.Combobox(tab, textvariable=self.var_ic_type, 
                               values=['circular', 'gaussian', 'multiple', 'image'],
                               state='readonly', width=20)
        ic_combo.grid(row=1, column=1, sticky='w', padx=5)
        ic_combo.bind('<<ComboboxSelected>>', self.update_ic_options)
        
        # Common parameters
        ttk.Label(tab, text="Common Parameters", font=('TkDefaultFont', 10, 'bold')).grid(
            row=2, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Background Temp (K):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_background_temp, width=15).grid(row=3, column=1, sticky='w', padx=5)
        
        # Circular hotspot frame
        self.circular_frame = ttk.LabelFrame(tab, text="Circular Hotspot Parameters")
        self.circular_frame.grid(row=4, column=0, columnspan=3, pady=10, padx=10, sticky='ew')
        
        ttk.Label(self.circular_frame, text="Hotspot Temp (K):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.circular_frame, textvariable=self.var_hotspot_temp, width=15).grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(self.circular_frame, text="Center X (m):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.circular_frame, textvariable=self.var_center_x, width=15).grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Label(self.circular_frame, text="Center Y (m):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.circular_frame, textvariable=self.var_center_y, width=15).grid(row=2, column=1, sticky='w', padx=5)
        
        ttk.Label(self.circular_frame, text="Radius (m):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.circular_frame, textvariable=self.var_hotspot_radius, width=15).grid(row=3, column=1, sticky='w', padx=5)
        
        ttk.Checkbutton(self.circular_frame, text="Smooth Transition", 
                       variable=self.var_smooth_transition).grid(row=4, column=0, columnspan=2, pady=5)
        
        ttk.Label(self.circular_frame, text="Transition Width (m):").grid(row=5, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.circular_frame, textvariable=self.var_transition_width, width=15).grid(row=5, column=1, sticky='w', padx=5)
        
        # Image-based frame (initially hidden)
        self.image_frame = ttk.LabelFrame(tab, text="Image-based Parameters")
        
        ttk.Label(self.image_frame, text="Image Path:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.image_frame, textvariable=self.var_image_path, width=30).grid(row=0, column=1, sticky='w', padx=5)
        ttk.Button(self.image_frame, text="Browse", command=self.browse_image).grid(row=0, column=2, padx=5)
        
        ttk.Checkbutton(self.image_frame, text="Use Constant Temp", 
                       variable=self.var_use_constant_temp).grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Label(self.image_frame, text="Constant Temp (K):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.image_frame, textvariable=self.var_constant_temp, width=15).grid(row=2, column=1, sticky='w', padx=5)
        
    def create_reaction_tab(self):
        """Create reaction settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Reactions")
        
        # Enable reactions checkbox
        self.enable_reactions_check = ttk.Checkbutton(tab, text="Enable Chemical Reactions", 
                                                     variable=self.var_enable_reactions,
                                                     command=self.toggle_reactions)
        self.enable_reactions_check.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Reaction parameters frame
        self.reaction_frame = ttk.LabelFrame(tab, text="Reaction Parameters")
        self.reaction_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky='ew')
        
        # Arrhenius parameters
        params = [
            ("Pre-exponential A (1/s):", self.var_reaction_A),
            ("Activation Energy Ea (J/mol):", self.var_reaction_Ea),
            ("Reaction Order n:", self.var_reaction_n),
            ("Heat of Reaction Q (J/kg):", self.var_reaction_Q)
        ]
        
        for i, (label, var) in enumerate(params):
            ttk.Label(self.reaction_frame, text=label).grid(row=i, column=0, sticky='e', padx=5, pady=5)
            entry = ttk.Entry(self.reaction_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky='w', padx=5)
            
        # Initially disable if reactions not enabled
        self.toggle_reactions()
        
    def create_amr_tab(self):
        """Create AMR settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="AMR")
        
        # Enable AMR checkbox
        self.enable_amr_check = ttk.Checkbutton(tab, text="Enable Adaptive Mesh Refinement", 
                                               variable=self.var_enable_amr,
                                               command=self.toggle_amr)
        self.enable_amr_check.grid(row=0, column=0, columnspan=2, pady=10)
        
        # AMR parameters frame
        self.amr_frame = ttk.LabelFrame(tab, text="AMR Parameters")
        self.amr_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky='ew')
        
        ttk.Label(self.amr_frame, text="Max Levels:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        ttk.Spinbox(self.amr_frame, from_=1, to=5, textvariable=self.var_max_levels, 
                   width=10).grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(self.amr_frame, text="Refinement Ratio:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        ratio_frame = ttk.Frame(self.amr_frame)
        ratio_frame.grid(row=1, column=1, sticky='w', padx=5)
        ttk.Radiobutton(ratio_frame, text="2", variable=self.var_refinement_ratio, 
                       value=2).pack(side='left')
        ttk.Radiobutton(ratio_frame, text="4", variable=self.var_refinement_ratio, 
                       value=4).pack(side='left')
        
        ttk.Label(self.amr_frame, text="Refine Threshold:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.amr_frame, textvariable=self.var_refine_threshold, width=15).grid(row=2, column=1, sticky='w', padx=5)
        
        ttk.Label(self.amr_frame, text="Coarsen Threshold:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(self.amr_frame, textvariable=self.var_coarsen_threshold, width=15).grid(row=3, column=1, sticky='w', padx=5)
        
        # Info about AMR
        info_text = """
AMR adaptively refines the mesh in regions with high gradients.
This can significantly reduce computational cost while maintaining accuracy.

Note: AMR is experimental in this implementation.
        """
        info_label = ttk.Label(self.amr_frame, text=info_text, justify='left', 
                              font=('TkDefaultFont', 9))
        info_label.grid(row=4, column=0, columnspan=2, pady=10, padx=10)
        
        self.toggle_amr()
        
    def create_visualization_tab(self):
        """Create visualization settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Visualization")
        
        ttk.Label(tab, text="Plotting Resolution", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="Plot Points (X):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_nx_plot, width=15).grid(row=1, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text="Plot Points (Y):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_ny_plot, width=15).grid(row=2, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text="Animation Settings", font=('TkDefaultFont', 10, 'bold')).grid(
            row=3, column=0, columnspan=2, pady=10)
        
        ttk.Checkbutton(tab, text="Create Animation", 
                       variable=self.var_create_animation).grid(row=4, column=0, columnspan=2, pady=5)
        
        ttk.Label(tab, text="Frame Skip:").grid(row=5, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_frame_skip, width=15).grid(row=5, column=1, sticky='w', padx=5)
        
        ttk.Checkbutton(tab, text="Save Animation", 
                       variable=self.var_save_animation).grid(row=6, column=0, columnspan=2, pady=5)
        
        ttk.Label(tab, text="Temperature Range", font=('TkDefaultFont', 10, 'bold')).grid(
            row=7, column=0, columnspan=2, pady=10)
        
        ttk.Label(tab, text="T Min (K):").grid(row=8, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_T_min_fixed, width=15).grid(row=8, column=1, sticky='w', padx=5)
        
        ttk.Label(tab, text="T Max (K):").grid(row=9, column=0, sticky='e', padx=5, pady=5)
        ttk.Entry(tab, textvariable=self.var_T_max_fixed, width=15).grid(row=9, column=1, sticky='w', padx=5)
        
    def update_resolution_info(self, *args):
        """Update resolution information display."""
        try:
            nx = self.var_nx_cells.get()
            ny = self.var_ny_cells.get()
            total_cells = nx * ny
            
            self.resolution_label.config(
                text=f"Total cells: {total_cells:,} ({nx}×{ny})"
            )
            
            # Estimate memory usage (rough)
            # Each cell needs ~8 bytes (double) + ghost cells
            memory_mb = total_cells * 8 * 2 / 1024 / 1024  # Factor of 2 for T and T_old
            if memory_mb < 1:
                self.resolution_label.config(
                    text=f"Total cells: {total_cells:,} ({nx}×{ny}) - Memory: <1 MB"
                )
            else:
                self.resolution_label.config(
                    text=f"Total cells: {total_cells:,} ({nx}×{ny}) - Memory: ~{memory_mb:.1f} MB"
                )
        except:
            pass
            
    def update_ic_options(self, event=None):
        """Show/hide IC options based on type."""
        ic_type = self.var_ic_type.get()
        
        # Hide all frames first
        self.circular_frame.grid_forget()
        self.image_frame.grid_forget()
        
        # Show relevant frame
        if ic_type == 'circular':
            self.circular_frame.grid(row=4, column=0, columnspan=3, pady=10, padx=10, sticky='ew')
        elif ic_type == 'image':
            self.image_frame.grid(row=4, column=0, columnspan=3, pady=10, padx=10, sticky='ew')
        # Add other IC types as needed
        
    def toggle_reactions(self):
        """Enable/disable reaction parameters."""
        state = 'normal' if self.var_enable_reactions.get() else 'disabled'
        for child in self.reaction_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.configure(state=state)
                
    def toggle_amr(self):
        """Enable/disable AMR parameters."""
        state = 'normal' if self.var_enable_amr.get() else 'disabled'
        for child in self.amr_frame.winfo_children():
            if isinstance(child, (ttk.Entry, ttk.Spinbox)):
                child.configure(state=state)
                
    def browse_image(self):
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select Temperature Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), 
                      ("All files", "*.*")]
        )
        if filename:
            self.var_image_path.set(filename)
            
    def get_config_dict(self):
        """Get configuration as dictionary."""
        config = {
            # Simulation
            'total_time': self.var_total_time.get(),
            'time_points': [float(t.strip()) for t in self.var_time_points.get().split(',')],
            'dt': self.var_dt.get(),
            'alpha': self.var_alpha.get(),
            'cfl': self.var_cfl.get(),
            
            # Grid
            'nx_cells': self.var_nx_cells.get(),
            'ny_cells': self.var_ny_cells.get(),
            'plate_length': self.var_plate_length.get(),
            'plate_width': self.var_plate_width.get(),
            
            # Numerical
            'spatial_order': self.var_spatial_order.get(),
            'time_integration': self.var_time_integration.get(),
            
            # Initial condition
            'ic_type': self.var_ic_type.get(),
            'background_temp': self.var_background_temp.get(),
            'hotspot_temp': self.var_hotspot_temp.get(),
            'center_x': self.var_center_x.get(),
            'center_y': self.var_center_y.get(),
            'hotspot_radius': self.var_hotspot_radius.get(),
            'smooth_transition': self.var_smooth_transition.get(),
            'transition_width': self.var_transition_width.get(),
            'image_path': self.var_image_path.get(),
            'use_constant_temp': self.var_use_constant_temp.get(),
            'constant_temp': self.var_constant_temp.get(),
            
            # Reactions
            'enable_reactions': self.var_enable_reactions.get(),
            'reaction_model': self.var_reaction_model.get(),
            'reaction_A': self.var_reaction_A.get(),
            'reaction_Ea': self.var_reaction_Ea.get(),
            'reaction_n': self.var_reaction_n.get(),
            'reaction_Q': self.var_reaction_Q.get(),
            
            # AMR
            'enable_amr': self.var_enable_amr.get(),
            'max_levels': self.var_max_levels.get(),
            'refinement_ratio': self.var_refinement_ratio.get(),
            'refine_threshold': self.var_refine_threshold.get(),
            'coarsen_threshold': self.var_coarsen_threshold.get(),
            
            # Visualization
            'nx_plot': self.var_nx_plot.get(),
            'ny_plot': self.var_ny_plot.get(),
            'create_animation': self.var_create_animation.get(),
            'frame_skip': self.var_frame_skip.get(),
            'save_animation': self.var_save_animation.get(),
            'T_min_fixed': self.var_T_min_fixed.get(),
            'T_max_fixed': self.var_T_max_fixed.get(),
        }
        
        return config
        
    def load_config(self, filename):
        """Load configuration from file."""
        if not os.path.exists(filename):
            return
            
        config = configparser.ConfigParser()
        config.read(filename)
        
        # Load values into GUI variables
        try:
            # Simulation
            if 'Simulation' in config:
                self.var_total_time.set(config.getfloat('Simulation', 'total_time', fallback=10.0))
                self.var_alpha.set(config.getfloat('Simulation', 'alpha', fallback=9.7e-5))
                self.var_cfl.set(config.getfloat('Simulation', 'cfl_diffusion', fallback=0.5))
                
            # Grid
            if 'Grid' in config:
                self.var_nx_cells.set(config.getint('Grid', 'nx_cells', fallback=100))
                self.var_ny_cells.set(config.getint('Grid', 'ny_cells', fallback=100))
                
            # Add other sections...
        except Exception as e:
            print(f"Error loading config: {e}")
            
    def run_simulation(self):
        """Run the FV simulation."""
        self.status_label.config(text="Running simulation...")
        self.progress.start()
        self.run_button.config(state='disabled')
        
        try:
            # Get configuration
            config = self.get_config_dict()
            
            # Print configuration summary
            print("\n" + "="*50)
            print("FV Heat Diffusion Simulation")
            print("="*50)
            print(f"Grid: {config['nx_cells']}×{config['ny_cells']} cells")
            print(f"Domain: {config['plate_length']}×{config['plate_width']} m")
            print(f"Spatial order: {config['spatial_order']}")
            print(f"Time integration: {config['time_integration']}")
            print(f"Reactions: {'Enabled' if config['enable_reactions'] else 'Disabled'}")
            print(f"AMR: {'Enabled' if config['enable_amr'] else 'Disabled'}")
            print(f"Simulation time: {config['total_time']} s")
            print("="*50 + "\n")
            
            # Create mesh
            mesh = FVMesh(
                nx_cells=config['nx_cells'],
                ny_cells=config['ny_cells'],
                plate_length=config['plate_length'],
                plate_width=config['plate_width']
            )
            
            # Create solver
            solver = FVHeatSolver(
                mesh=mesh,
                alpha=config['alpha'],
                spatial_order=config['spatial_order'],
                time_integration=config['time_integration'],
                enable_reactions=config['enable_reactions']
            )
            
            # Set initial condition
            ic_params = {
                'background_temp': config['background_temp'],
                'hotspot_temp': config['hotspot_temp'],
                'center_x': config['center_x'],
                'center_y': config['center_y'],
                'hotspot_radius': config['hotspot_radius'],
                'smooth_transition': config['smooth_transition'],
                'transition_width': config['transition_width'],
                'image_path': config['image_path'],
                'use_constant_temp': config['use_constant_temp'],
                'constant_temp': config['constant_temp'],
            }
            
            ic = get_fv_initial_condition(config['ic_type'], ic_params)
            ic.set(solver)
            
            # Setup reactions if enabled
            if config['enable_reactions']:
                # Import reaction model (implement this based on your needs)
                messagebox.showinfo("Info", "Reaction models not yet implemented")
                
            # Setup AMR if enabled
            amr_system = None
            if config['enable_amr']:
                amr_system = SimpleAMR(
                    base_solver=solver,
                    max_levels=config['max_levels'],
                    refinement_ratio=config['refinement_ratio'],
                    refine_threshold=config['refine_threshold'],
                    coarsen_threshold=config['coarsen_threshold']
                )
                
            # Create post-processor
            postprocessor = FVPostProcessor(mesh)
            
            # Determine time step
            if config['dt']:
                dt = float(config['dt'])
            else:
                dt = solver.compute_stable_timestep(config['cfl'])
                print(f"Auto-computed time step: {dt:.3e} s")
                
            # Plot solution at specified times
            time_points = config['time_points']
            
            fig = plot_fv_solution_snapshots(
                solver, postprocessor, time_points,
                nx_plot=config['nx_plot'],
                ny_plot=config['ny_plot'],
                T_min_fixed=config['T_min_fixed'],
                T_max_fixed=config['T_max_fixed']
            )
            
            plt.show()
            
            # Create animation if requested
            if config['create_animation']:
                # Reset solver for animation
                ic.set(solver)
                
                anim = create_fv_animation(
                    solver, postprocessor,
                    total_time=config['total_time'],
                    dt=dt,
                    nx_plot=config['nx_plot'],
                    ny_plot=config['ny_plot'],
                    frame_skip=config['frame_skip'],
                    T_min_fixed=config['T_min_fixed'],
                    T_max_fixed=config['T_max_fixed'],
                    save_animation=config['save_animation']
                )
                
                plt.show()
                
            self.status_label.config(text="Simulation completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed:\n{str(e)}")
            self.status_label.config(text="Simulation failed!")
            raise
            
        finally:
            self.progress.stop()
            self.run_button.config(state='normal')


def main():
    """Main entry point."""
    root = tk.Tk()
    app = FVHeatSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()