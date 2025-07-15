# Add this as the FIRST lines in main.py
import sys

print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[:3]}...")  # First few paths

try:
    import amrex.space2d as amr

    print("Direct import in main.py successful!")
except ImportError as e:
    print(f"Direct import in main.py failed: {e}")

# Continue with rest of imports...


# main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import configparser
import os
import sys
from amr.simple_amr import SimpleAMRRefactored as SimpleAMRRefactored

# Import our FV modules
from mesh import FVMesh
from solver import FVHeatSolver
from postprocessor import FVPostProcessor
from initial_conditions import get_fv_initial_condition
from animation import (
    create_fv_animation_from_history,
    plot_fv_solution_snapshots,
    plot_convergence_study,
)
from typing import Optional


class FVHeatSimulatorGUI:
    """
    GUI for Finite Volume Heat Diffusion Simulator.

    Simplified from FR version, focusing on FV-specific parameters.
    """

    def __init__(self, master):
        self.master = master
        master.title("FV Heat Diffusion Simulator")
        master.geometry("1000x700")  # Wider to accommodate side panel

        # Variables
        self.setup_variables()

        # Create main container frame
        main_container = ttk.Frame(master)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Create left frame for notebook
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side="left", fill="both", expand=True)

        # Create right frame for controls
        right_frame = ttk.Frame(main_container, width=200)
        right_frame.pack(side="right", fill="y", padx=(10, 0))
        right_frame.pack_propagate(False)  # Maintain fixed width

        # Create notebook for tabs in left frame
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill="both", expand=True)

        # Create tabs
        self.create_simulation_tab()
        self.create_grid_tab()
        self.create_initial_condition_tab()
        self.create_numerical_tab()
        self.create_reaction_tab()
        self.create_amr_tab()
        self.create_visualization_tab()

        # Right panel contents
        # Title
        title_label = ttk.Label(
            right_frame, text="Simulation Control", font=("TkDefaultFont", 12, "bold")
        )
        title_label.pack(pady=(10, 20))

        # Run button
        self.run_button = ttk.Button(
            right_frame,
            text="Run Simulation",
            command=self.run_simulation,
            style="Accent.TButton",
        )
        self.run_button.pack(pady=10, padx=20, fill="x")

        # Separator
        ttk.Separator(right_frame, orient="horizontal").pack(fill="x", pady=10)

        # Quick actions frame
        quick_frame = ttk.LabelFrame(right_frame, text="Quick Actions")
        quick_frame.pack(pady=10, padx=10, fill="x")

        ttk.Button(quick_frame, text="Save Config", command=self.save_config).pack(
            pady=5, padx=10, fill="x"
        )
        ttk.Button(
            quick_frame, text="Load Config", command=self.load_config_dialog
        ).pack(pady=5, padx=10, fill="x")
        ttk.Button(
            quick_frame, text="Reset to Defaults", command=self.reset_to_defaults
        ).pack(pady=5, padx=10, fill="x")

        # Separator
        ttk.Separator(right_frame, orient="horizontal").pack(fill="x", pady=10)

        # Status frame
        status_frame = ttk.LabelFrame(right_frame, text="Status")
        status_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=10)

        # Status text with scrollbar
        status_text_frame = ttk.Frame(status_frame)
        status_text_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.status_text = tk.Text(
            status_text_frame,
            height=10,
            width=20,
            wrap="word",
            font=("TkDefaultFont", 9),
        )
        self.status_text.pack(side="left", fill="both", expand=True)

        status_scrollbar = ttk.Scrollbar(
            status_text_frame, orient="vertical", command=self.status_text.yview
        )
        status_scrollbar.pack(side="right", fill="y")
        self.status_text.config(yscrollcommand=status_scrollbar.set)

        # Status label (single line)
        self.status_label = ttk.Label(right_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(fill="x", padx=10, pady=5)

        # Load default configuration
        self.load_config("config_fv.ini")

        # Track simulation state
        self.simulation_running = False

        # Initialize status
        self.update_status("System ready")

        # Create a custom style for primary buttons
        style = ttk.Style()
        style.configure("Primary.TButton", font=("TkDefaultFont", 10, "bold"))

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
        self.var_time_integration = tk.StringVar(value="RK3")

        # Initial condition
        self.var_ic_type = tk.StringVar(value="circular")
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
        self.var_reaction_model = tk.StringVar(value="arrhenius")
        self.var_reaction_A = tk.DoubleVar(value=1.5e15)
        self.var_reaction_Ea = tk.DoubleVar(value=180000)
        self.var_reaction_n = tk.DoubleVar(value=1.0)
        self.var_reaction_Q = tk.DoubleVar(value=2.5e6)

        # AMR
        self.var_enable_amr = tk.BooleanVar(value=False)
        self.var_amr_backend = tk.StringVar(value="amrex")
        self.var_max_levels = tk.IntVar(value=3)
        self.var_refinement_ratio = tk.IntVar(value=2)
        self.var_regrid_interval = tk.IntVar(value=10)
        self.var_refine_threshold = tk.DoubleVar(value=100.0)
        self.var_coarsen_threshold = tk.DoubleVar(value=10.0)

        # AMReX specific
        self.var_amrex_max_grid_size = tk.IntVar(value=32)
        self.var_amrex_blocking_factor = tk.IntVar(value=8)
        self.var_amrex_grid_eff = tk.DoubleVar(value=0.7)
        self.var_amrex_n_error_buf = tk.IntVar(value=2)

        # AMR visualization
        self.var_show_amr_grid = tk.BooleanVar(value=False)
        self.var_color_by_level = tk.BooleanVar(value=True)
        self.var_show_refinement_criteria = tk.BooleanVar(value=False)

        # AMR workflow control
        self.var_amr_initial_levels = tk.IntVar(value=1)
        self.var_amr_adapt_after_ic = tk.BooleanVar(value=True)
        self.var_amr_show_before_adapt = tk.BooleanVar(value=False)
        self.var_temp_threshold = tk.DoubleVar(value=500.0)
        self.var_show_error_indicator = tk.BooleanVar(value=False)

        # Visualization
        self.var_nx_plot = tk.IntVar(value=101)
        self.var_ny_plot = tk.IntVar(value=101)
        self.var_create_animation = tk.BooleanVar(value=False)
        self.var_frame_skip = tk.IntVar(value=10)
        self.var_save_animation = tk.BooleanVar(value=True)
        self.var_T_min_fixed = tk.DoubleVar(value=300.0)
        self.var_T_max_fixed = tk.DoubleVar(value=6000.0)
        self.var_show_mesh = tk.BooleanVar(value=False)
        self.var_show_hotspot = tk.BooleanVar(value=False)
        self.var_show_centerlines = tk.BooleanVar(value=False)

    def create_simulation_tab(self):
        """Create simulation settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Simulation")

        # Time settings
        ttk.Label(
            tab, text="Simulation Time Settings", font=("TkDefaultFont", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(tab, text="Total Time (s):").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_total_time, width=15).grid(
            row=1, column=1, sticky="w", padx=5
        )

        ttk.Label(tab, text="Output Times:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_time_points, width=30).grid(
            row=2, column=1, sticky="w", padx=5
        )

        ttk.Label(tab, text="Time Step (s):").grid(
            row=3, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_dt, width=15).grid(
            row=3, column=1, sticky="w", padx=5
        )
        ttk.Label(tab, text="(leave empty for auto)").grid(row=3, column=2, sticky="w")

        # Physical parameters
        ttk.Label(
            tab, text="Physical Parameters", font=("TkDefaultFont", 10, "bold")
        ).grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Label(tab, text="Thermal Diffusivity (m²/s):").grid(
            row=5, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_alpha, width=15).grid(
            row=5, column=1, sticky="w", padx=5
        )

        ttk.Label(tab, text="CFL Number:").grid(
            row=6, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_cfl, width=15).grid(
            row=6, column=1, sticky="w", padx=5
        )

    def create_grid_tab(self):
        """Create grid settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Grid")

        ttk.Label(tab, text="Grid Resolution", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=10
        )

        ttk.Label(tab, text="Number of Cells (X):").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_nx_cells, width=15).grid(
            row=1, column=1, sticky="w", padx=5
        )

        ttk.Label(tab, text="Number of Cells (Y):").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_ny_cells, width=15).grid(
            row=2, column=1, sticky="w", padx=5
        )

        # Resolution info
        self.resolution_label = ttk.Label(
            tab, text="", font=("TkDefaultFont", 9, "italic")
        )
        self.resolution_label.grid(row=3, column=0, columnspan=2, pady=5)

        # Update resolution info when values change
        self.var_nx_cells.trace("w", self.update_resolution_info)
        self.var_ny_cells.trace("w", self.update_resolution_info)

        ttk.Label(tab, text="Domain Size", font=("TkDefaultFont", 10, "bold")).grid(
            row=4, column=0, columnspan=2, pady=10
        )

        ttk.Label(tab, text="Plate Length (m):").grid(
            row=5, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_plate_length, width=15).grid(
            row=5, column=1, sticky="w", padx=5
        )

        ttk.Label(tab, text="Plate Width (m):").grid(
            row=6, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_plate_width, width=15).grid(
            row=6, column=1, sticky="w", padx=5
        )

        # Mesh Preview section
        ttk.Label(tab, text="Mesh Preview", font=("TkDefaultFont", 10, "bold")).grid(
            row=7, column=0, columnspan=2, pady=10
        )

        # Preview options
        preview_frame = ttk.Frame(tab)
        preview_frame.grid(row=8, column=0, columnspan=2, pady=5)

        ttk.Button(
            preview_frame, text="Show Mesh", command=self.show_mesh_preview
        ).pack(side="left", padx=5)

        ttk.Button(
            preview_frame,
            text="Show Mesh + Hotspot",
            command=self.show_mesh_with_hotspot,
        ).pack(side="left", padx=5)

        # Info about cell sizes
        self.cell_size_label = ttk.Label(
            tab, text="", font=("TkDefaultFont", 9, "italic")
        )
        self.cell_size_label.grid(row=9, column=0, columnspan=2, pady=5)

        # Update cell size info when values change
        self.var_nx_cells.trace("w", self.update_cell_size_info)
        self.var_ny_cells.trace("w", self.update_cell_size_info)
        self.var_plate_length.trace("w", self.update_cell_size_info)
        self.var_plate_width.trace("w", self.update_cell_size_info)

        # Initial update
        self.update_cell_size_info()

    def create_numerical_tab(self):
        """Create numerical methods tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Numerical")

        ttk.Label(
            tab, text="Spatial Discretization", font=("TkDefaultFont", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(tab, text="Spatial Order:").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        order_frame = ttk.Frame(tab)
        order_frame.grid(row=1, column=1, sticky="w", padx=5)

        ttk.Radiobutton(
            order_frame, text="1st", variable=self.var_spatial_order, value=1
        ).pack(side="left")
        ttk.Radiobutton(
            order_frame, text="2nd", variable=self.var_spatial_order, value=2
        ).pack(side="left")
        ttk.Radiobutton(
            order_frame, text="5th (WENO)", variable=self.var_spatial_order, value=5
        ).pack(side="left")

        ttk.Label(
            tab, text="Time Integration", font=("TkDefaultFont", 10, "bold")
        ).grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Label(tab, text="Method:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        time_frame = ttk.Frame(tab)
        time_frame.grid(row=3, column=1, sticky="w", padx=5)

        ttk.Radiobutton(
            time_frame, text="Euler", variable=self.var_time_integration, value="Euler"
        ).pack(side="left")
        ttk.Radiobutton(
            time_frame, text="RK3", variable=self.var_time_integration, value="RK3"
        ).pack(side="left")

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
        info_label = ttk.Label(
            tab, text=info_text, justify="left", font=("TkDefaultFont", 9)
        )
        info_label.grid(row=4, column=0, columnspan=2, pady=20, padx=10)

    def create_initial_condition_tab(self):
        """Create initial condition tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Initial Condition")

        ttk.Label(
            tab, text="Initial Condition Type", font=("TkDefaultFont", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(tab, text="Type:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ic_combo = ttk.Combobox(
            tab,
            textvariable=self.var_ic_type,
            values=["circular", "gaussian", "multiple", "image"],
            state="readonly",
            width=20,
        )
        ic_combo.grid(row=1, column=1, sticky="w", padx=5)
        ic_combo.bind("<<ComboboxSelected>>", self.update_ic_options)

        # Common parameters
        ttk.Label(
            tab, text="Common Parameters", font=("TkDefaultFont", 10, "bold")
        ).grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Label(tab, text="Background Temp (K):").grid(
            row=3, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_background_temp, width=15).grid(
            row=3, column=1, sticky="w", padx=5
        )

        # Circular hotspot frame
        self.circular_frame = ttk.LabelFrame(tab, text="Circular Hotspot Parameters")
        self.circular_frame.grid(
            row=4, column=0, columnspan=3, pady=10, padx=10, sticky="ew"
        )

        ttk.Label(self.circular_frame, text="Hotspot Temp (K):").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.circular_frame, textvariable=self.var_hotspot_temp, width=15
        ).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(self.circular_frame, text="Center X (m):").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(self.circular_frame, textvariable=self.var_center_x, width=15).grid(
            row=1, column=1, sticky="w", padx=5
        )

        ttk.Label(self.circular_frame, text="Center Y (m):").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(self.circular_frame, textvariable=self.var_center_y, width=15).grid(
            row=2, column=1, sticky="w", padx=5
        )

        ttk.Label(self.circular_frame, text="Radius (m):").grid(
            row=3, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.circular_frame, textvariable=self.var_hotspot_radius, width=15
        ).grid(row=3, column=1, sticky="w", padx=5)

        ttk.Checkbutton(
            self.circular_frame,
            text="Smooth Transition",
            variable=self.var_smooth_transition,
        ).grid(row=4, column=0, columnspan=2, pady=5)

        ttk.Label(self.circular_frame, text="Transition Width (m):").grid(
            row=5, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.circular_frame, textvariable=self.var_transition_width, width=15
        ).grid(row=5, column=1, sticky="w", padx=5)

        # Image-based frame (initially hidden)
        self.image_frame = ttk.LabelFrame(tab, text="Image-based Parameters")

        ttk.Label(self.image_frame, text="Image Path:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(self.image_frame, textvariable=self.var_image_path, width=30).grid(
            row=0, column=1, sticky="w", padx=5
        )
        ttk.Button(self.image_frame, text="Browse", command=self.browse_image).grid(
            row=0, column=2, padx=5
        )

        ttk.Checkbutton(
            self.image_frame,
            text="Use Constant Temp",
            variable=self.var_use_constant_temp,
        ).grid(row=1, column=0, columnspan=2, pady=5)

        ttk.Label(self.image_frame, text="Constant Temp (K):").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(self.image_frame, textvariable=self.var_constant_temp, width=15).grid(
            row=2, column=1, sticky="w", padx=5
        )

    def create_reaction_tab(self):
        """Create reaction settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Reactions")

        # Enable reactions checkbox
        self.enable_reactions_check = ttk.Checkbutton(
            tab,
            text="Enable Chemical Reactions",
            variable=self.var_enable_reactions,
            command=self.toggle_reactions,
        )
        self.enable_reactions_check.grid(row=0, column=0, columnspan=2, pady=10)

        # Reaction parameters frame
        self.reaction_frame = ttk.LabelFrame(tab, text="Reaction Parameters")
        self.reaction_frame.grid(
            row=1, column=0, columnspan=2, pady=10, padx=10, sticky="ew"
        )

        # Arrhenius parameters
        params = [
            ("Pre-exponential A (1/s):", self.var_reaction_A),
            ("Activation Energy Ea (J/mol):", self.var_reaction_Ea),
            ("Reaction Order n:", self.var_reaction_n),
            ("Heat of Reaction Q (J/kg):", self.var_reaction_Q),
        ]

        for i, (label, var) in enumerate(params):
            ttk.Label(self.reaction_frame, text=label).grid(
                row=i, column=0, sticky="e", padx=5, pady=5
            )
            entry = ttk.Entry(self.reaction_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky="w", padx=5)

        # Initially disable if reactions not enabled
        self.toggle_reactions()

    # Fix for create_amr_tab method - Complete replacement

    def create_amr_tab(self):
        """Create AMR settings tab with backend selection and workflow options."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="AMR")

        # Create scrollable frame for AMR content
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create inner frame for grid layout
        inner_frame = ttk.Frame(scrollable_frame)
        inner_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Now add all AMR content to inner_frame using grid
        current_row = 0

        # Enable AMR checkbox
        self.enable_amr_check = ttk.Checkbutton(
            inner_frame,
            text="Enable Adaptive Mesh Refinement",
            variable=self.var_enable_amr,
            command=self.toggle_amr,
        )
        self.enable_amr_check.grid(row=current_row, column=0, columnspan=2, pady=10)
        current_row += 1

        # AMR Backend selection
        backend_frame = ttk.LabelFrame(inner_frame, text="AMR Backend")
        backend_frame.grid(
            row=current_row, column=0, columnspan=2, pady=5, padx=10, sticky="ew"
        )
        current_row += 1

        ttk.Label(backend_frame, text="Backend:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )

        # Get available backends dynamically
        from amr.amr_factory import AMRFactory

        available_backends = [
            b for b in AMRFactory.get_available_backends() if b != "none"
        ]

        self.amr_backend_combo = ttk.Combobox(
            backend_frame,
            textvariable=self.var_amr_backend,
            values=available_backends,
            state="readonly",
            width=15,
        )
        self.amr_backend_combo.grid(row=0, column=1, sticky="w", padx=5)
        self.amr_backend_combo.bind("<<ComboboxSelected>>", self.on_amr_backend_changed)

        # Backend info button
        ttk.Button(backend_frame, text="Info", command=self.show_backend_info).grid(
            row=0, column=2, padx=5
        )

        # Backend status label
        self.amr_backend_status = ttk.Label(
            backend_frame, text="", font=("TkDefaultFont", 9, "italic")
        )
        self.amr_backend_status.grid(row=1, column=0, columnspan=3, pady=5)

        # Common AMR parameters frame
        self.amr_common_frame = ttk.LabelFrame(
            inner_frame, text="Common AMR Parameters"
        )
        self.amr_common_frame.grid(
            row=current_row, column=0, columnspan=2, pady=5, padx=10, sticky="ew"
        )
        current_row += 1

        # Common parameters
        ttk.Label(self.amr_common_frame, text="Max Levels:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Spinbox(
            self.amr_common_frame,
            from_=1,
            to=5,
            textvariable=self.var_max_levels,
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(self.amr_common_frame, text="Refinement Ratio:").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        ratio_frame = ttk.Frame(self.amr_common_frame)
        ratio_frame.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Radiobutton(
            ratio_frame, text="2", variable=self.var_refinement_ratio, value=2
        ).pack(side="left")
        ttk.Radiobutton(
            ratio_frame, text="4", variable=self.var_refinement_ratio, value=4
        ).pack(side="left")

        ttk.Label(self.amr_common_frame, text="Regrid Interval:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.amr_common_frame, textvariable=self.var_regrid_interval, width=10
        ).grid(row=2, column=1, sticky="w", padx=5)

        # AMR Workflow frame
        workflow_frame = ttk.LabelFrame(inner_frame, text="AMR Workflow")
        workflow_frame.grid(
            row=current_row, column=0, columnspan=2, pady=5, padx=10, sticky="ew"
        )
        current_row += 1

        ttk.Label(workflow_frame, text="Initial Levels:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Spinbox(
            workflow_frame,
            from_=1,
            to=1,
            textvariable=self.var_amr_initial_levels,
            width=10,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(
            workflow_frame,
            text="(Start with base level only)",
            font=("TkDefaultFont", 8, "italic"),
        ).grid(row=0, column=2, sticky="w")

        ttk.Checkbutton(
            workflow_frame,
            text="Adapt grid after initial condition",
            variable=self.var_amr_adapt_after_ic,
        ).grid(row=1, column=0, columnspan=3, pady=5)

        ttk.Checkbutton(
            workflow_frame,
            text="Show uniform grid before adaptation",
            variable=self.var_amr_show_before_adapt,
        ).grid(row=2, column=0, columnspan=3, pady=5)

        ttk.Checkbutton(
            workflow_frame,
            text="Show error indicator field",
            variable=self.var_show_error_indicator,
        ).grid(row=3, column=0, columnspan=3, pady=5)

        # Adaptation Criteria frame
        adapt_frame = ttk.LabelFrame(inner_frame, text="Adaptation Criteria")
        adapt_frame.grid(
            row=current_row, column=0, columnspan=2, pady=5, padx=10, sticky="ew"
        )
        current_row += 1

        ttk.Label(adapt_frame, text="Gradient Threshold:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(adapt_frame, textvariable=self.var_refine_threshold, width=15).grid(
            row=0, column=1, sticky="w", padx=5
        )
        ttk.Label(
            adapt_frame, text="(for |∇T|)", font=("TkDefaultFont", 8, "italic")
        ).grid(row=0, column=2, sticky="w")

        ttk.Label(adapt_frame, text="Temperature Threshold:").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(adapt_frame, textvariable=self.var_temp_threshold, width=15).grid(
            row=1, column=1, sticky="w", padx=5
        )
        ttk.Label(
            adapt_frame,
            text="(minimum T for refinement)",
            font=("TkDefaultFont", 8, "italic"),
        ).grid(row=1, column=2, sticky="w")

        ttk.Label(adapt_frame, text="Coarsen Threshold:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(adapt_frame, textvariable=self.var_coarsen_threshold, width=15).grid(
            row=2, column=1, sticky="w", padx=5
        )

        # Simple AMR parameters frame
        self.amr_simple_frame = ttk.LabelFrame(
            inner_frame, text="Simple AMR Parameters"
        )
        self.amr_simple_frame.grid(
            row=current_row, column=0, columnspan=2, pady=5, padx=10, sticky="ew"
        )
        current_row += 1

        # (Simple AMR parameters remain the same)
        ttk.Label(self.amr_simple_frame, text="Refine Threshold:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.amr_simple_frame, textvariable=self.var_refine_threshold, width=15
        ).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(self.amr_simple_frame, text="Coarsen Threshold:").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.amr_simple_frame, textvariable=self.var_coarsen_threshold, width=15
        ).grid(row=1, column=1, sticky="w", padx=5)

        # AMReX parameters frame
        self.amr_amrex_frame = ttk.LabelFrame(inner_frame, text="AMReX Parameters")
        self.amr_amrex_frame.grid(
            row=current_row, column=0, columnspan=2, pady=5, padx=10, sticky="ew"
        )
        current_row += 1

        # AMReX specific parameters
        ttk.Label(self.amr_amrex_frame, text="Max Grid Size:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.amr_amrex_frame, textvariable=self.var_amrex_max_grid_size, width=10
        ).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(self.amr_amrex_frame, text="Blocking Factor:").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.amr_amrex_frame, textvariable=self.var_amrex_blocking_factor, width=10
        ).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(self.amr_amrex_frame, text="Grid Efficiency:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.amr_amrex_frame, textvariable=self.var_amrex_grid_eff, width=10
        ).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(self.amr_amrex_frame, text="Error Buffer Cells:").grid(
            row=3, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(
            self.amr_amrex_frame, textvariable=self.var_amrex_n_error_buf, width=10
        ).grid(row=3, column=1, sticky="w", padx=5)

        # AMR visualization options
        viz_frame = ttk.LabelFrame(inner_frame, text="AMR Visualization")
        viz_frame.grid(
            row=current_row, column=0, columnspan=2, pady=5, padx=10, sticky="ew"
        )
        current_row += 1

        ttk.Checkbutton(
            viz_frame, text="Show AMR grid structure", variable=self.var_show_amr_grid
        ).pack(anchor="w", padx=5, pady=2)
        ttk.Checkbutton(
            viz_frame,
            text="Color by refinement level",
            variable=self.var_color_by_level,
        ).pack(anchor="w", padx=5, pady=2)
        ttk.Checkbutton(
            viz_frame,
            text="Show refinement criteria",
            variable=self.var_show_refinement_criteria,
        ).pack(anchor="w", padx=5, pady=2)

        # Updated button frame with Initialize & Adapt button
        button_frame = ttk.Frame(viz_frame)
        button_frame.pack(pady=5)

        # Initialize & Adapt button - primary action
        self.init_adapt_btn = ttk.Button(
            button_frame,
            text="Initialize & Adapt",
            command=self.initialize_and_adapt_amr,
            style="Primary.TButton",
        )
        self.init_adapt_btn.pack(side="left", padx=5)

        ttk.Button(
            button_frame, text="Preview AMR Grid", command=self.preview_amr_grid
        ).pack(side="left", padx=5)
        ttk.Button(
            button_frame,
            text="Verify AMR Adaptation",
            command=self.verify_amr_adaptation,
        ).pack(side="left", padx=5)

        # Info text
        info_frame = ttk.Frame(inner_frame)
        info_frame.grid(
            row=current_row, column=0, columnspan=2, pady=10, padx=10, sticky="ew"
        )

        self.amr_info_text = tk.Text(
            info_frame, height=6, width=50, wrap="word", font=("TkDefaultFont", 9)
        )
        self.amr_info_text.pack(side="left", fill="both", expand=True)

        scrollbar_info = ttk.Scrollbar(
            info_frame, orient="vertical", command=self.amr_info_text.yview
        )
        scrollbar_info.pack(side="right", fill="y")
        self.amr_info_text.config(yscrollcommand=scrollbar_info.set)

        # Configure column weights for inner_frame
        inner_frame.columnconfigure(0, weight=1)
        inner_frame.columnconfigure(1, weight=1)

        # Set default info text
        self.update_amr_info()

        # Initialize UI state
        self.toggle_amr()
        self.on_amr_backend_changed()

    def create_visualization_tab(self):
        """Create visualization settings tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Visualization")

        ttk.Label(
            tab, text="Plotting Resolution", font=("TkDefaultFont", 10, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(tab, text="Plot Points (X):").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_nx_plot, width=15).grid(
            row=1, column=1, sticky="w", padx=5
        )

        ttk.Label(tab, text="Plot Points (Y):").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_ny_plot, width=15).grid(
            row=2, column=1, sticky="w", padx=5
        )

        ttk.Label(
            tab, text="Visualization Options", font=("TkDefaultFont", 10, "bold")
        ).grid(row=3, column=0, columnspan=2, pady=10)

        self.show_mesh_check = ttk.Checkbutton(
            tab, text="Show Mesh Lines", variable=self.var_show_mesh
        )
        self.show_mesh_check.grid(row=4, column=0, columnspan=2, pady=5)

        self.show_hotspot_check = ttk.Checkbutton(
            tab, text="Show Initial Hotspot Boundary", variable=self.var_show_hotspot
        )
        self.show_hotspot_check.grid(row=5, column=0, columnspan=2, pady=5)

        self.show_centerlines_check = ttk.Checkbutton(
            tab, text="Plot Centerline Temperatures", variable=self.var_show_centerlines
        )
        self.show_centerlines_check.grid(row=6, column=0, columnspan=2, pady=5)

        ttk.Label(
            tab,
            text="Note: Centerline data collected during simulation",
            font=("TkDefaultFont", 8, "italic"),
        ).grid(row=7, column=0, columnspan=2)

        ttk.Label(
            tab, text="Animation Settings", font=("TkDefaultFont", 10, "bold")
        ).grid(row=8, column=0, columnspan=2, pady=10)

        self.create_animation_check = ttk.Checkbutton(
            tab, text="Create Animation", variable=self.var_create_animation
        )
        self.create_animation_check.grid(row=9, column=0, columnspan=2, pady=5)

        ttk.Label(tab, text="Frame Skip:").grid(
            row=10, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_frame_skip, width=15).grid(
            row=10, column=1, sticky="w", padx=5
        )

        ttk.Checkbutton(
            tab, text="Save Animation", variable=self.var_save_animation
        ).grid(row=11, column=0, columnspan=2, pady=5)

        ttk.Label(
            tab, text="Temperature Range", font=("TkDefaultFont", 10, "bold")
        ).grid(row=12, column=0, columnspan=2, pady=10)

        ttk.Label(tab, text="T Min (K):").grid(
            row=13, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_T_min_fixed, width=15).grid(
            row=13, column=1, sticky="w", padx=5
        )

        ttk.Label(tab, text="T Max (K):").grid(
            row=14, column=0, sticky="e", padx=5, pady=5
        )
        ttk.Entry(tab, textvariable=self.var_T_max_fixed, width=15).grid(
            row=14, column=1, sticky="w", padx=5
        )

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

    def update_cell_size_info(self, *args):
        """Update cell size information."""
        try:
            nx = self.var_nx_cells.get()
            ny = self.var_ny_cells.get()
            length = self.var_plate_length.get()
            width = self.var_plate_width.get()

            dx = length / nx
            dy = width / ny

            # Convert to mm for display
            dx_mm = dx * 1000
            dy_mm = dy * 1000

            self.cell_size_label.config(
                text=f"Cell size: Δx = {dx_mm:.2f} mm, Δy = {dy_mm:.2f} mm"
            )
        except:
            pass

    def show_mesh_preview(self):
        """Show mesh preview in a new window."""
        try:
            # Get current parameters
            nx = self.var_nx_cells.get()
            ny = self.var_ny_cells.get()
            length = self.var_plate_length.get()
            width = self.var_plate_width.get()

            # Create temporary mesh
            from mesh import FVMesh

            mesh = FVMesh(nx, ny, length, width)

            # Create figure
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, ax = plt.subplots(1, 1, figsize=(8, 8 * width / length))

            # Draw mesh
            # Draw all cells
            for j in range(ny):
                for i in range(nx):
                    x_left = mesh.x_faces[i]
                    x_right = mesh.x_faces[i + 1]
                    y_bottom = mesh.y_faces[j]
                    y_top = mesh.y_faces[j + 1]

                    rect = patches.Rectangle(
                        (x_left, y_bottom),
                        x_right - x_left,
                        y_top - y_bottom,
                        linewidth=0.5,
                        edgecolor="black",
                        facecolor="lightblue",
                        alpha=0.3,
                    )
                    ax.add_patch(rect)

            # Highlight a few cells if mesh is coarse
            if nx <= 30 and ny <= 30:
                # Mark center cell
                i_center = nx // 2
                j_center = ny // 2
                x_left = mesh.x_faces[i_center]
                x_right = mesh.x_faces[i_center + 1]
                y_bottom = mesh.y_faces[j_center]
                y_top = mesh.y_faces[j_center + 1]

                rect = patches.Rectangle(
                    (x_left, y_bottom),
                    x_right - x_left,
                    y_top - y_bottom,
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="pink",
                    alpha=0.5,
                    label="Center cell",
                )
                ax.add_patch(rect)

            # Add grid lines for clarity
            for i in range(nx + 1):
                ax.axvline(x=mesh.x_faces[i], color="gray", linewidth=0.5, alpha=0.5)
            for j in range(ny + 1):
                ax.axhline(y=mesh.y_faces[j], color="gray", linewidth=0.5, alpha=0.5)

            ax.set_xlim(0, length)
            ax.set_ylim(0, width)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)", fontsize=12)
            ax.set_ylabel("Y (m)", fontsize=12)
            ax.set_title(
                f"Finite Volume Mesh: {nx}×{ny} cells\n"
                f"Domain: {length}×{width} m, "
                f"Cell size: {mesh.dx*1000:.1f}×{mesh.dy*1000:.1f} mm",
                fontsize=14,
            )

            if nx <= 30 and ny <= 30:
                ax.legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create mesh preview:\n{str(e)}")

    def show_mesh_with_hotspot(self):
        """Show mesh preview with hotspot overlay."""
        try:
            # Get mesh parameters
            nx = self.var_nx_cells.get()
            ny = self.var_ny_cells.get()
            length = self.var_plate_length.get()
            width = self.var_plate_width.get()

            # Get hotspot parameters
            ic_type = self.var_ic_type.get()
            if ic_type != "circular":
                messagebox.showinfo(
                    "Info",
                    "Hotspot preview only available for circular initial condition",
                )
                return

            center_x = self.var_center_x.get()
            center_y = self.var_center_y.get()
            radius = self.var_hotspot_radius.get()

            # Create temporary mesh
            from mesh import FVMesh

            mesh = FVMesh(nx, ny, length, width)

            # Create figure
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np

            fig, ax = plt.subplots(1, 1, figsize=(8, 8 * width / length))

            # Determine which cells are inside hotspot
            cells_in_hotspot = []
            cells_partial = []

            for j in range(ny):
                for i in range(nx):
                    x_center = mesh.x_centers[i]
                    y_center = mesh.y_centers[j]

                    # Check if cell center is in hotspot
                    r = np.sqrt((x_center - center_x) ** 2 + (y_center - center_y) ** 2)

                    if r <= radius:
                        cells_in_hotspot.append((i, j))
                    elif r <= radius + np.sqrt(mesh.dx**2 + mesh.dy**2) / 2:
                        # Cell might be partially in hotspot
                        cells_partial.append((i, j))

            # Draw all cells
            for j in range(ny):
                for i in range(nx):
                    x_left = mesh.x_faces[i]
                    x_right = mesh.x_faces[i + 1]
                    y_bottom = mesh.y_faces[j]
                    y_top = mesh.y_faces[j + 1]

                    # Determine cell color
                    if (i, j) in cells_in_hotspot:
                        facecolor = "red"
                        alpha = 0.5
                        edgecolor = "darkred"
                    elif (i, j) in cells_partial:
                        facecolor = "orange"
                        alpha = 0.3
                        edgecolor = "darkorange"
                    else:
                        facecolor = "lightblue"
                        alpha = 0.2
                        edgecolor = "black"

                    rect = patches.Rectangle(
                        (x_left, y_bottom),
                        x_right - x_left,
                        y_top - y_bottom,
                        linewidth=0.5,
                        edgecolor=edgecolor,
                        facecolor=facecolor,
                        alpha=alpha,
                    )
                    ax.add_patch(rect)

            # Draw hotspot circle
            circle = patches.Circle(
                (center_x, center_y),
                radius,
                linewidth=2,
                edgecolor="darkgreen",
                facecolor="none",
                linestyle="--",
                label=f"Hotspot (r={radius*1000:.0f}mm)",
            )
            ax.add_patch(circle)

            # Mark hotspot center
            ax.plot(
                center_x,
                center_y,
                "g+",
                markersize=10,
                markeredgewidth=2,
                label=f"Center ({center_x:.3f}, {center_y:.3f})",
            )

            # Add thin grid lines
            for i in range(nx + 1):
                ax.axvline(x=mesh.x_faces[i], color="gray", linewidth=0.3, alpha=0.3)
            for j in range(ny + 1):
                ax.axhline(y=mesh.y_faces[j], color="gray", linewidth=0.3, alpha=0.3)

            ax.set_xlim(0, length)
            ax.set_ylim(0, width)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)", fontsize=12)
            ax.set_ylabel("Y (m)", fontsize=12)
            ax.set_title(
                f"Mesh with Hotspot Preview\n"
                f"{nx}×{ny} cells, "
                f"{len(cells_in_hotspot)} cells fully inside hotspot, "
                f"{len(cells_partial)} cells partially inside",
                fontsize=14,
            )

            # Add legend
            ax.legend(loc="upper right")

            # Add color explanation
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="red", alpha=0.5, label="Fully inside hotspot"),
                Patch(facecolor="orange", alpha=0.3, label="Partially inside"),
                Patch(facecolor="lightblue", alpha=0.2, label="Outside hotspot"),
            ]
            ax2 = ax.twinx()
            ax2.set_yticks([])
            ax2.legend(handles=legend_elements, loc="upper left")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create mesh preview:\n{str(e)}")

    def update_ic_options(self, event=None):
        """Show/hide IC options based on type."""
        ic_type = self.var_ic_type.get()

        # Hide all frames first
        self.circular_frame.grid_forget()
        self.image_frame.grid_forget()

        # Show relevant frame
        if ic_type == "circular":
            self.circular_frame.grid(
                row=4, column=0, columnspan=3, pady=10, padx=10, sticky="ew"
            )
        elif ic_type == "image":
            self.image_frame.grid(
                row=4, column=0, columnspan=3, pady=10, padx=10, sticky="ew"
            )
        # Add other IC types as needed

    def toggle_reactions(self):
        """Enable/disable reaction parameters."""
        state = "normal" if self.var_enable_reactions.get() else "disabled"
        for child in self.reaction_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.configure(state=state)

    def browse_image(self):
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select Temperature Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.var_image_path.set(filename)

    def get_config_dict(self):
        """Get configuration as dictionary."""
        config = {
            # Simulation
            "total_time": self.var_total_time.get(),
            "time_points": [
                float(t.strip()) for t in self.var_time_points.get().split(",")
            ],
            "dt": self.var_dt.get(),
            "alpha": self.var_alpha.get(),
            "cfl": self.var_cfl.get(),
            # Grid
            "nx_cells": self.var_nx_cells.get(),
            "ny_cells": self.var_ny_cells.get(),
            "plate_length": self.var_plate_length.get(),
            "plate_width": self.var_plate_width.get(),
            # Numerical
            "spatial_order": self.var_spatial_order.get(),
            "time_integration": self.var_time_integration.get(),
            # Initial condition
            "ic_type": self.var_ic_type.get(),
            "background_temp": self.var_background_temp.get(),
            "hotspot_temp": self.var_hotspot_temp.get(),
            "center_x": self.var_center_x.get(),
            "center_y": self.var_center_y.get(),
            "hotspot_radius": self.var_hotspot_radius.get(),
            "smooth_transition": self.var_smooth_transition.get(),
            "transition_width": self.var_transition_width.get(),
            "image_path": self.var_image_path.get(),
            "use_constant_temp": self.var_use_constant_temp.get(),
            "constant_temp": self.var_constant_temp.get(),
            # Reactions
            "enable_reactions": self.var_enable_reactions.get(),
            "reaction_model": self.var_reaction_model.get(),
            "reaction_A": self.var_reaction_A.get(),
            "reaction_Ea": self.var_reaction_Ea.get(),
            "reaction_n": self.var_reaction_n.get(),
            "reaction_Q": self.var_reaction_Q.get(),
            # AMR
            "enable_amr": self.var_enable_amr.get(),
            "amr_backend": (
                self.var_amr_backend.get()
                if hasattr(self, "var_amr_backend")
                else "simple"
            ),
            "max_levels": self.var_max_levels.get(),
            "refinement_ratio": self.var_refinement_ratio.get(),
            "refine_threshold": self.var_refine_threshold.get(),
            "coarsen_threshold": self.var_coarsen_threshold.get(),
            # Visualization
            "nx_plot": self.var_nx_plot.get(),
            "ny_plot": self.var_ny_plot.get(),
            "create_animation": self.var_create_animation.get(),
            "frame_skip": self.var_frame_skip.get(),
            "save_animation": self.var_save_animation.get(),
            "T_min_fixed": self.var_T_min_fixed.get(),
            "T_max_fixed": self.var_T_max_fixed.get(),
            "show_mesh": self.var_show_mesh.get(),
            "show_hotspot": self.var_show_hotspot.get(),
            "show_centerlines": self.var_show_centerlines.get(),
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
            if "Simulation" in config:
                self.var_total_time.set(
                    config.getfloat("Simulation", "total_time", fallback=10.0)
                )
                self.var_alpha.set(
                    config.getfloat("Simulation", "alpha", fallback=9.7e-5)
                )
                self.var_cfl.set(
                    config.getfloat("Simulation", "cfl_diffusion", fallback=0.5)
                )

            # Grid
            if "Grid" in config:
                self.var_nx_cells.set(config.getint("Grid", "nx_cells", fallback=100))
                self.var_ny_cells.set(config.getint("Grid", "ny_cells", fallback=100))

            # Add other sections...
        except Exception as e:
            print(f"Error loading config: {e}")

    def disable_controls_during_simulation(self):
        """Disable all controls during simulation."""
        self.simulation_running = True
        self.run_button.config(state="disabled")

        # Disable all tabs
        for i in range(self.notebook.index("end")):
            self.notebook.tab(i, state="disabled")

        # Disable visualization checkboxes that affect simulation
        self.show_centerlines_check.config(state="disabled")
        self.create_animation_check.config(state="disabled")

    def enable_controls_after_simulation(self):
        """Re-enable controls after simulation."""
        self.simulation_running = False
        self.run_button.config(state="normal")

        # Enable all tabs
        for i in range(self.notebook.index("end")):
            self.notebook.tab(i, state="normal")

        # Re-enable visualization checkboxes
        self.show_centerlines_check.config(state="normal")
        self.create_animation_check.config(state="normal")

    def toggle_amr(self):
        """Enable/disable AMR parameters based on checkbox."""
        state = "normal" if self.var_enable_amr.get() else "disabled"

        # Enable/disable backend selection
        self.amr_backend_combo.configure(
            state="readonly" if self.var_enable_amr.get() else "disabled"
        )

        # Enable/disable all parameter frames
        for frame in [
            self.amr_common_frame,
            self.amr_simple_frame,
            self.amr_amrex_frame,
        ]:
            for child in frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Spinbox)):
                    child.configure(state=state)
                elif isinstance(child, ttk.Frame):
                    for subchild in child.winfo_children():
                        if isinstance(subchild, ttk.Radiobutton):
                            subchild.configure(state=state)

        # Update which parameter frames are shown
        if self.var_enable_amr.get():
            self.on_amr_backend_changed()
        else:
            self.amr_simple_frame.grid_remove()
            self.amr_amrex_frame.grid_remove()

    def on_amr_backend_changed(self, event=None):
        """Handle AMR backend selection change."""
        if not self.var_enable_amr.get():
            return

        backend = self.var_amr_backend.get()

        # Hide all backend-specific frames
        self.amr_simple_frame.grid_remove()
        self.amr_amrex_frame.grid_remove()

        # Show appropriate frame
        if backend == "simple":
            self.amr_simple_frame.grid()
        elif backend == "amrex":
            self.amr_amrex_frame.grid()

        # Update status and info
        self.update_backend_status()
        self.update_amr_info()

    def update_backend_status(self):
        """Update AMR backend status label."""
        from amr.amr_factory import AMRFactory

        backend = self.var_amr_backend.get()
        info = AMRFactory.get_backend_info(backend)

        if info["available"]:
            self.amr_backend_status.config(
                text=f"✓ {backend} backend available", foreground="green"
            )
        else:
            self.amr_backend_status.config(
                text=f"✗ {backend} backend not available - check requirements",
                foreground="red",
            )

    def show_backend_info(self):
        """Show detailed information about selected AMR backend."""
        from amr.amr_factory import AMRFactory

        backend = self.var_amr_backend.get()
        info = AMRFactory.get_backend_info(backend)

        # Create info window
        info_window = tk.Toplevel(self.master)
        info_window.title(f"{info['name']} Information")
        info_window.geometry("500x400")

        # Create text widget with info
        text = tk.Text(info_window, wrap="word", padx=10, pady=10)
        text.pack(fill="both", expand=True)

        # Format and insert information
        text.insert("end", f"{info['name']}\n", "title")
        text.insert("end", f"{'-' * 40}\n\n")

        text.insert("end", f"Description:\n", "heading")
        text.insert("end", f"{info['description']}\n\n")

        text.insert("end", f"Status: ", "heading")
        if info["available"]:
            text.insert("end", "Available\n\n", "available")
        else:
            text.insert("end", "Not Available\n\n", "unavailable")

        if info.get("features"):
            text.insert("end", "Features:\n", "heading")
            for feature in info["features"]:
                text.insert("end", f"  • {feature}\n")
            text.insert("end", "\n")

        if info.get("requirements"):
            text.insert("end", "Requirements:\n", "heading")
            for req in info["requirements"]:
                text.insert("end", f"  • {req}\n")
            text.insert("end", "\n")

        text.insert("end", "Capabilities:\n", "heading")
        text.insert("end", f"  • Parallel: {'Yes' if info.get('parallel') else 'No'}\n")
        text.insert(
            "end", f"  • GPU Support: {'Yes' if info.get('gpu_support') else 'No'}\n"
        )

        # Configure tags
        text.tag_config("title", font=("TkDefaultFont", 14, "bold"))
        text.tag_config("heading", font=("TkDefaultFont", 10, "bold"))
        text.tag_config(
            "available", foreground="green", font=("TkDefaultFont", 10, "bold")
        )
        text.tag_config(
            "unavailable", foreground="red", font=("TkDefaultFont", 10, "bold")
        )

        text.config(state="disabled")

        # Add close button
        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)

    def update_amr_info(self):
        """Update AMR info text based on selected backend."""
        self.amr_info_text.config(state="normal")
        self.amr_info_text.delete("1.0", "end")

        if not self.var_enable_amr.get():
            self.amr_info_text.insert(
                "end", "AMR is disabled. Enable to use adaptive mesh refinement."
            )
        else:
            backend = self.var_amr_backend.get()

            if backend == "simple":
                info = """Simple AMR uses a quadtree-based refinement strategy with the following features:
    • Basic block-structured refinement
    • Gradient-based error indicators
    • Time subcycling for efficiency
    • Python-based implementation

    Good for: Learning, prototyping, moderate scale problems
    Limitations: No MPI parallelism, simplified flux correction"""

            elif backend == "amrex":
                info = """AMReX provides state-of-the-art AMR capabilities:
    • Full Berger-Oliger AMR algorithm
    • MPI+OpenMP parallel execution
    • GPU acceleration support
    • Automatic load balancing
    • Conservative flux correction
    • Built-in I/O and visualization

    Good for: Production runs, large-scale problems, HPC systems
    Note: Requires pyAMReX installation"""

            self.amr_info_text.insert("end", info)

        self.amr_info_text.config(state="disabled")

    def get_amr_config(self):
        """Get AMR configuration based on GUI settings."""
        if not self.var_enable_amr.get():
            return None

        backend = self.var_amr_backend.get()

        # Common parameters
        config = {
            "backend": backend,
            "max_levels": self.var_max_levels.get(),
            "refinement_ratio": self.var_refinement_ratio.get(),
            "regrid_interval": self.var_regrid_interval.get(),
            # New workflow parameters
            "initial_levels": self.var_amr_initial_levels.get(),
            "adapt_after_ic": self.var_amr_adapt_after_ic.get(),
            "show_before_adapt": self.var_amr_show_before_adapt.get(),
            "temp_threshold": self.var_temp_threshold.get(),
            "show_error_indicator": self.var_show_error_indicator.get(),
        }

        # Backend-specific parameters
        if backend == "simple":
            config.update(
                {
                    "refine_threshold": self.var_refine_threshold.get(),
                    "coarsen_threshold": self.var_coarsen_threshold.get(),
                }
            )
        elif backend == "amrex":
            config.update(
                {
                    "max_grid_size": self.var_amrex_max_grid_size.get(),
                    "blocking_factor": self.var_amrex_blocking_factor.get(),
                    "grid_eff": self.var_amrex_grid_eff.get(),
                    "n_error_buf": self.var_amrex_n_error_buf.get(),
                    "refine_threshold": self.var_refine_threshold.get(),
                    "coarsen_threshold": self.var_coarsen_threshold.get(),
                }
            )

        return config

    def run_simulation(self):
        """Run the FV simulation with optional AMR support."""
        if self.simulation_running:
            messagebox.showwarning("Warning", "Simulation already running!")
            return

        self.status_label.config(text="Running simulation...")
        self.progress.start()
        self.disable_controls_during_simulation()

        try:
            # Get configuration
            config = self.get_config_dict()

            # Print configuration summary
            print("\n" + "=" * 50)
            print("FV Heat Diffusion Simulation")
            print("=" * 50)
            print(f"Grid: {config['nx_cells']}×{config['ny_cells']} cells")
            print(f"Domain: {config['plate_length']}×{config['plate_width']} m")
            print(f"Spatial order: {config['spatial_order']}")
            print(f"Time integration: {config['time_integration']}")
            print(
                f"Reactions: {'Enabled' if config['enable_reactions'] else 'Disabled'}"
            )

            # AMR information
            if config["enable_amr"]:
                print(f"AMR: Enabled - Backend: {config.get('amr_backend', 'none')}")
                print(f"     Max levels: {config['max_levels']}")
                print(f"     Refinement ratio: {config['refinement_ratio']}")
            else:
                print("AMR: Disabled")

            print(f"Simulation time: {config['total_time']} s")
            print(
                f"Centerline collection: {'Enabled' if config['show_centerlines'] else 'Disabled'}"
            )
            print(
                f"Animation creation: {'Enabled' if config['create_animation'] else 'Disabled'}"
            )
            print("=" * 50 + "\n")

            # Import required modules
            from mesh import FVMesh
            from solver import FVHeatSolver
            from postprocessor import FVPostProcessor
            from initial_conditions import get_fv_initial_condition
            from animation import (
                create_fv_animation_from_history,
                plot_fv_solution_snapshots_from_history,
            )
            from amr.amr_factory import AMRFactory
            from matplotlib.gridspec import GridSpec

            # Create mesh
            mesh = FVMesh(
                nx_cells=config["nx_cells"],
                ny_cells=config["ny_cells"],
                plate_length=config["plate_length"],
                plate_width=config["plate_width"],
            )

            # Create solver
            solver = FVHeatSolver(
                mesh=mesh,
                alpha=config["alpha"],
                spatial_order=config["spatial_order"],
                time_integration=config["time_integration"],
                enable_reactions=config["enable_reactions"],
            )

            # Set up AMR if enabled
            amr_system = None
            if config["enable_amr"]:
                amr_config = self.get_amr_config()

                try:
                    # Create AMR system with base grid only
                    self.status_label.config(text="Creating AMR base grid...")
                    self.master.update()

                    amr_system = AMRFactory.create_amr(
                        backend=config["amr_backend"],
                        base_solver=solver,
                        config=amr_config,
                    )

                    if amr_system is not None:
                        solver.set_amr_system(amr_system)
                        print(f"AMR system initialized: {type(amr_system).__name__}")
                        print("Base grid created - ready for initial condition")
                    else:
                        print("Failed to create AMR system")

                except Exception as e:
                    print(f"Error initializing AMR: {e}")
                    messagebox.showwarning(
                        "AMR Warning",
                        f"Failed to initialize AMR: {e}\nContinuing without AMR.",
                    )
                    amr_system = None

            # Enable centerline collection if requested
            if config["show_centerlines"]:
                solver.enable_centerline_collection(True)
                print("Centerline data collection enabled")

            # Set initial condition
            self.status_label.config(text="Setting initial condition...")
            self.master.update()

            ic_params = {
                "background_temp": config["background_temp"],
                "hotspot_temp": config["hotspot_temp"],
                "center_x": config["center_x"],
                "center_y": config["center_y"],
                "hotspot_radius": config["hotspot_radius"],
                "smooth_transition": config["smooth_transition"],
                "transition_width": config["transition_width"],
                "image_path": config["image_path"],
                "use_constant_temp": config["use_constant_temp"],
                "constant_temp": config["constant_temp"],
            }

            ic = get_fv_initial_condition(config["ic_type"], ic_params)
            ic.set(solver)
            print("Initial condition set on uniform grid")

            # Setup reactions if enabled
            if config["enable_reactions"]:
                messagebox.showinfo("Info", "Reaction models not yet implemented")

            # Create post-processor
            postprocessor = FVPostProcessor(mesh)

            # Get hotspot parameters for visualization
            hotspot_params = None
            if config["ic_type"] == "circular" and (
                config.get("show_hotspot", False)
                or config.get("show_centerlines", False)
            ):
                hotspot_params = {
                    "center_x": config["center_x"],
                    "center_y": config["center_y"],
                    "radius": config["hotspot_radius"],
                }

            # AMR Grid Adaptation Phase
            if amr_system is not None and config.get("adapt_after_ic", True):
                # Show uniform grid before adaptation if requested
                if config.get("show_before_adapt", False):
                    print("\nShowing uniform grid before adaptation...")
                    fig_uniform = plt.figure(figsize=(10, 8))
                    gs = GridSpec(1, 2, figure=fig_uniform, width_ratios=[1, 1])

                    # Left: Grid structure
                    ax_grid = fig_uniform.add_subplot(gs[0])
                    amr_system.plot_grid_structure(ax_grid, show_levels=True)
                    ax_grid.set_title("Uniform Grid (Before Adaptation)")

                    # Right: Temperature field
                    ax_temp = fig_uniform.add_subplot(gs[1])
                    sol = postprocessor.get_solution_on_grid(
                        solver, config["nx_plot"], config["ny_plot"], smooth=True
                    )
                    X, Y = np.meshgrid(sol["x"], sol["y"])
                    cs = ax_temp.contourf(X, Y, sol["T"], levels=50, cmap="hot")
                    plt.colorbar(cs, ax=ax_temp, label="Temperature (K)")
                    ax_temp.set_xlabel("X (m)")
                    ax_temp.set_ylabel("Y (m)")
                    ax_temp.set_title("Initial Condition on Uniform Grid")
                    ax_temp.set_aspect("equal")

                    plt.tight_layout()
                    plt.show()

                # Adapt grid to initial condition
                self.status_label.config(text="Adapting grid to initial condition...")
                self.master.update()

                print("\nAdapting AMR grid to initial condition...")
                amr_system.adapt_to_initial_condition()

                # Show adapted grid if requested
                if config.get("show_amr_grid", False):
                    print("\nShowing adapted grid...")
                    fig_adapted = plt.figure(figsize=(12, 8))
                    gs = GridSpec(1, 2, figure=fig_adapted, width_ratios=[1, 1])

                    # Left: Grid structure
                    ax_grid = fig_adapted.add_subplot(gs[0])
                    amr_system.plot_grid_structure(ax_grid, show_levels=True)
                    ax_grid.set_title("Adapted AMR Grid")

                    # Right: Composite solution
                    ax_temp = fig_adapted.add_subplot(gs[1])
                    composite = amr_system.get_composite_solution("T")
                    X, Y = np.meshgrid(composite["x"], composite["y"])
                    cs = ax_temp.contourf(
                        X, Y, composite["data"], levels=50, cmap="hot"
                    )
                    plt.colorbar(cs, ax=ax_temp, label="Temperature (K)")

                    # Overlay grid structure
                    amr_system.plot_grid_structure(ax_temp, show_levels=False)
                    ax_temp.set_xlabel("X (m)")
                    ax_temp.set_ylabel("Y (m)")
                    ax_temp.set_title("Initial Condition with AMR Grid")
                    ax_temp.set_aspect("equal")

                    plt.tight_layout()
                    plt.show()

                    # Also show level map if available
                    if "level_map" in composite and composite["level_map"] is not None:
                        fig_levels = plt.figure(figsize=(8, 8))
                        ax_levels = fig_levels.add_subplot(111)
                        im = ax_levels.imshow(
                            composite["level_map"],
                            origin="lower",
                            extent=[
                                0,
                                solver.mesh.plate_length,
                                0,
                                solver.mesh.plate_width,
                            ],
                            cmap="tab10",
                            interpolation="nearest",
                        )
                        cbar = plt.colorbar(
                            im, ax=ax_levels, ticks=range(amr_system.max_levels)
                        )
                        cbar.set_label("Refinement Level")
                        ax_levels.set_xlabel("X (m)")
                        ax_levels.set_ylabel("Y (m)")
                        ax_levels.set_title("AMR Refinement Levels")
                        ax_levels.set_aspect("equal")
                        plt.show()

            # Print AMR statistics if enabled
            if amr_system is not None:
                amr_stats = amr_system.get_statistics()
                print(f"\nInitial AMR statistics:")
                print(f"  Total cells: {amr_stats['total_cells']:,}")
                for level, count in amr_stats["cells_per_level"].items():
                    print(f"  Level {level}: {count:,} cells")
                efficiency = (1.0 - amr_stats["efficiency"]) * 100
                print(f"  Memory savings vs uniform fine grid: {efficiency:.1f}%")

            # Determine time step
            if config["dt"]:
                dt = float(config["dt"])
            else:
                dt = solver.compute_stable_timestep(config["cfl"])
                print(f"Auto-computed time step: {dt:.3e} s")

            # Print diagnostics
            print(f"\nSimulation parameters:")
            print(f"  Thermal diffusivity α = {config['alpha']:.2e} m²/s")
            print(f"  Cell size: Δx = {mesh.dx:.3e} m, Δy = {mesh.dy:.3e} m")
            print(f"  Time step: Δt = {dt:.3e} s")
            print(f"  Diffusion number: α*Δt/Δx² = {config['alpha']*dt/mesh.dx**2:.3f}")
            print(
                f"  Expected diffusion length at t=10s: L ≈ √(4αt) = {np.sqrt(4*config['alpha']*10):.3f} m"
            )

            # Main simulation loop
            self.status_label.config(text="Running simulation...")
            self.master.update()

            time_points = config["time_points"]
            solution_snapshots = []

            # Store initial solution
            print("Storing initial solution...")
            if amr_system is not None:
                # Get composite solution from AMR
                composite = solver.get_solution_for_visualization(
                    use_amr_composite=True
                )
                sol = {
                    "T": composite["data"],
                    "x": composite["x"],
                    "y": composite["y"],
                    "level_map": composite.get("level_map"),
                }
            else:
                # Regular solution
                sol = postprocessor.get_solution_on_grid(
                    solver, config["nx_plot"], config["ny_plot"], smooth=True
                )

            solution_snapshots.append((0.0, sol))

            # Prepare for animation if requested
            animation_frames = []
            if config["create_animation"]:
                frame_data = postprocessor.prepare_animation_frame(
                    solver, config["nx_plot"], config["ny_plot"]
                )
                frame_data["time"] = 0.0
                animation_frames.append(frame_data)

            # Run simulation
            print("\nRunning simulation...")

            # Choose advance method based on whether AMR is enabled
            if amr_system is not None:
                advance_method = solver.advance_to_time_with_amr
            else:
                advance_method = solver.advance_to_time

            # Animation frame collection logic
            if config["create_animation"]:
                frame_dt = dt * config["frame_skip"]
                all_frame_times = np.arange(
                    0, config["total_time"] + frame_dt, frame_dt
                )
                if all_frame_times[-1] < config["total_time"]:
                    all_frame_times = np.append(all_frame_times, config["total_time"])

                all_times = sorted(set(list(time_points[1:]) + list(all_frame_times)))
                frame_time_set = set(all_frame_times)
                output_time_set = set(time_points[1:])

                for target_time in all_times:
                    is_output_time = target_time in output_time_set

                    advance_method(
                        target_time,
                        dt,
                        show_progress=is_output_time,
                        collect_interval=None,
                        collect_at_target=is_output_time and config["show_centerlines"],
                    )

                    if is_output_time:
                        print(
                            f"Storing solution snapshot at t = {solver.current_time:.3f} s"
                        )

                        # Get solution (AMR composite or regular)
                        if amr_system is not None:
                            composite = solver.get_solution_for_visualization(
                                use_amr_composite=True
                            )
                            sol = {
                                "T": composite["data"],
                                "x": composite["x"],
                                "y": composite["y"],
                                "level_map": composite.get("level_map"),
                            }
                        else:
                            sol = postprocessor.get_solution_on_grid(
                                solver,
                                config["nx_plot"],
                                config["ny_plot"],
                                smooth=True,
                            )

                        solution_snapshots.append((solver.current_time, sol))

                        # Print AMR statistics if enabled
                        if amr_system is not None:
                            amr_stats = amr_system.get_statistics()
                            print(
                                f"  AMR: {amr_stats['total_cells']:,} cells, "
                                f"efficiency: {amr_stats['efficiency']:.1%}"
                            )

                    if target_time in frame_time_set:
                        frame_data = postprocessor.prepare_animation_frame(
                            solver, config["nx_plot"], config["ny_plot"]
                        )
                        frame_data["time"] = solver.current_time
                        animation_frames.append(frame_data)
            else:
                # No animation - just output times
                for target_time in time_points[1:]:
                    print(f"\nAdvancing to t = {target_time} s")

                    advance_method(
                        target_time,
                        dt,
                        show_progress=True,
                        collect_interval=None,
                        collect_at_target=config["show_centerlines"],
                    )

                    print(
                        f"Storing solution snapshot at t = {solver.current_time:.3f} s"
                    )

                    if amr_system is not None:
                        composite = solver.get_solution_for_visualization(
                            use_amr_composite=True
                        )
                        sol = {
                            "T": composite["data"],
                            "x": composite["x"],
                            "y": composite["y"],
                            "level_map": composite.get("level_map"),
                        }

                        # Print AMR statistics
                        amr_stats = amr_system.get_statistics()
                        print(
                            f"  AMR: {amr_stats['total_cells']:,} cells, "
                            f"efficiency: {amr_stats['efficiency']:.1%}"
                        )
                    else:
                        sol = postprocessor.get_solution_on_grid(
                            solver, config["nx_plot"], config["ny_plot"], smooth=True
                        )

                    solution_snapshots.append((solver.current_time, sol))

            print("\nSimulation completed!")
            print(f"Collected {len(solution_snapshots)} solution snapshots")
            if config["create_animation"]:
                print(f"Collected {len(animation_frames)} frames for animation")

            # Final AMR statistics
            if amr_system is not None:
                print("\nFinal AMR statistics:")
                final_stats = amr_system.get_statistics()
                print(f"  Total cells: {final_stats['total_cells']:,}")
                for level, count in final_stats["cells_per_level"].items():
                    print(f"  Level {level}: {count:,} cells")
                print(f"  Efficiency: {final_stats['efficiency']:.1%}")

                # Memory usage
                memory = amr_system.get_memory_usage()
                print(f"  Memory usage: {memory['total_mb']:.1f} MB")

            # Visualization
            from animation import plot_fv_solution_snapshots_from_history

            print("\nPlotting solution snapshots...")
            fig = plot_fv_solution_snapshots_from_history(
                solution_snapshots,
                solver,
                postprocessor,
                T_min_fixed=config["T_min_fixed"],
                T_max_fixed=config["T_max_fixed"],
                show_mesh=config.get("show_mesh", False),
                show_hotspot=config.get("show_hotspot", False),
                hotspot_params=hotspot_params,
            )

            plt.show()

            # Plot centerlines if collected
            if config["show_centerlines"]:
                centerline_history = solver.get_centerline_history()
                if centerline_history is not None:
                    print("\nPlotting centerline evolution...")
                    fig_centerlines, axes = postprocessor.plot_centerlines_from_history(
                        centerline_history,
                        hotspot_params=hotspot_params,
                        show_mesh_lines=config.get("show_mesh", False),
                    )
                    plt.show()

            # Create animation if requested
            if config["create_animation"] and len(animation_frames) > 0:
                print("\nCreating animation...")
                anim = create_fv_animation_from_history(
                    solver,
                    postprocessor,
                    animation_frames,
                    T_min_fixed=config["T_min_fixed"],
                    T_max_fixed=config["T_max_fixed"],
                    save_animation=config["save_animation"],
                )
                plt.show()

            # Show final AMR grid if requested
            if amr_system is not None and config.get("show_amr_grid", False):
                print("\nShowing final AMR grid structure...")
                fig_final = plt.figure(figsize=(8, 8))
                ax_final = fig_final.add_subplot(111)
                amr_system.plot_grid_structure(ax_final, show_levels=True)
                ax_final.set_title("Final AMR Grid Structure")
                plt.show()

            self.status_label.config(text="Simulation completed!")

        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed:\n{str(e)}")
            self.status_label.config(text="Simulation failed!")
            raise

        finally:
            self.progress.stop()
            self.enable_controls_after_simulation()

    def preview_amr_grid_org(self):
        """Preview AMR grid structure without running simulation."""
        if not self.var_enable_amr.get():
            messagebox.showinfo("Info", "Please enable AMR first")
            return

        try:
            # Get configuration
            config = self.get_config_dict()

            # Create mesh and solver
            from mesh import FVMesh
            from solver import FVHeatSolver
            from initial_conditions import get_fv_initial_condition
            from amr.amr_factory import AMRFactory

            # Create mesh
            mesh = FVMesh(
                nx_cells=config["nx_cells"],
                ny_cells=config["ny_cells"],
                plate_length=config["plate_length"],
                plate_width=config["plate_width"],
            )

            # Create solver
            solver = FVHeatSolver(
                mesh=mesh,
                alpha=config["alpha"],
                spatial_order=config["spatial_order"],
                time_integration=config["time_integration"],
                enable_reactions=config["enable_reactions"],
            )

            # Set initial condition
            ic_params = {
                "background_temp": config["background_temp"],
                "hotspot_temp": config["hotspot_temp"],
                "center_x": config["center_x"],
                "center_y": config["center_y"],
                "hotspot_radius": config["hotspot_radius"],
                "smooth_transition": config["smooth_transition"],
                "transition_width": config["transition_width"],
            }

            ic = get_fv_initial_condition(config["ic_type"], ic_params)
            ic.set(solver)

            # Create AMR system
            amr_config = self.get_amr_config()
            amr_system = AMRFactory.create_amr(
                backend=config["amr_backend"], base_solver=solver, config=amr_config
            )

            if amr_system is not None:
                # Plot grid structure
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

                # Left: Grid structure
                amr_system.plot_grid_structure(ax1, show_levels=True)

                # Right: Initial temperature with grid overlay
                sol = amr_system.get_composite_solution("T")
                X, Y = np.meshgrid(sol["x"], sol["y"])

                cs = ax2.contourf(X, Y, sol["data"], levels=50, cmap="hot")
                plt.colorbar(cs, ax=ax2, label="Temperature (K)")

                # Overlay grid structure
                amr_system.plot_grid_structure(ax2, show_levels=True)
                ax2.set_title("Initial Temperature with AMR Grid")

                # Show statistics
                stats = amr_system.get_statistics()
                info_text = (
                    f"Total cells: {stats['total_cells']:,}\n"
                    f"Efficiency: {stats['efficiency']:.1%}\n"
                )
                for level, count in stats["cells_per_level"].items():
                    info_text += f"Level {level}: {count:,} cells\n"

                ax1.text(
                    0.02,
                    0.02,
                    info_text,
                    transform=ax1.transAxes,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

                plt.tight_layout()
                plt.show()

                # Also show refinement criteria
                fig2, ax = plt.subplots(figsize=(8, 8))
                indicators = amr_system.compute_refinement_indicators(0)

                im = ax.imshow(indicators, cmap="viridis", origin="lower")
                plt.colorbar(im, ax=ax, label="Refinement Indicator")
                ax.set_title("Refinement Criteria (Base Level)")
                ax.set_xlabel("X cells")
                ax.set_ylabel("Y cells")

                # Add threshold line
                threshold = amr_config.get("refine_threshold", 100.0)
                ax.axhline(
                    y=threshold,
                    color="red",
                    linestyle="--",
                    label=f"Threshold: {threshold}",
                )

                plt.show()
            else:
                messagebox.showerror("Error", "Failed to create AMR system")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview AMR grid:\n{str(e)}")

    def verify_amr_adaptation(self):
        """Show verification of AMR adaptation."""
        if not self.var_enable_amr.get():
            messagebox.showinfo("Info", "AMR is not enabled")
            return

        messagebox.showinfo(
            "AMR Verification",
            "AMR Adaptation Verification:\n\n"
            "1. Set initial condition on uniform grid\n"
            "2. Compute error indicators (gradients)\n"
            "3. Flag cells exceeding threshold\n"
            "4. Create refined grids around flagged cells\n"
            "5. Interpolate solution to refined grids\n\n"
            "Check console output for detailed information.",
        )

    def preview_amr_grid(self):
        """Preview AMR grid structure without running simulation."""
        if not self.var_enable_amr.get():
            messagebox.showinfo("Info", "Please enable AMR first")
            return

        try:
            # Get configuration
            config = self.get_config_dict()
            backend = self.var_amr_backend.get()

            print(f"\n=== AMR Preview Debug ===")
            print(f"Backend requested: {backend}")

            # Import and check AMR factory
            try:
                from amr.amr_factory import AMRFactory

                print("✓ AMRFactory imported successfully")
            except ImportError as e:
                print(f"✗ Failed to import AMRFactory: {e}")
                raise

            # Check available backends
            try:
                available = AMRFactory.get_available_backends()
                print(f"Available backends: {available}")
            except Exception as e:
                print(f"✗ Failed to get available backends: {e}")
                available = []

            # Check if requested backend is available
            if backend not in available:
                print(f"⚠ Backend '{backend}' not in available list")
                if "simple" in available:
                    print("  → Falling back to 'simple' backend")
                    backend = "simple"
                else:
                    raise ValueError(
                        f"No AMR backends available! Available: {available}"
                    )

            # Create mesh and solver
            print("\nCreating mesh and solver...")
            from mesh import FVMesh
            from solver import FVHeatSolver
            from initial_conditions import get_fv_initial_condition

            mesh = FVMesh(
                nx_cells=config["nx_cells"],
                ny_cells=config["ny_cells"],
                plate_length=config["plate_length"],
                plate_width=config["plate_width"],
            )
            print(f"✓ Mesh created: {config['nx_cells']}×{config['ny_cells']} cells")

            solver = FVHeatSolver(
                mesh=mesh,
                alpha=config["alpha"],
                spatial_order=config["spatial_order"],
                time_integration=config["time_integration"],
                enable_reactions=config["enable_reactions"],
            )
            print("✓ Solver created")

            # Set initial condition
            print("\nSetting initial condition...")
            ic_params = {
                "background_temp": config["background_temp"],
                "hotspot_temp": config["hotspot_temp"],
                "center_x": config["center_x"],
                "center_y": config["center_y"],
                "hotspot_radius": config["hotspot_radius"],
                "smooth_transition": config["smooth_transition"],
                "transition_width": config["transition_width"],
            }

            ic = get_fv_initial_condition(config["ic_type"], ic_params)
            ic.set(solver)
            print("✓ Initial condition set")

            # Create AMR system
            print(f"\nCreating AMR system with backend: {backend}")
            amr_config = self.get_amr_config()
            print(f"AMR config: {amr_config}")

            try:
                amr_system = AMRFactory.create_amr(
                    backend=backend, base_solver=solver, config=amr_config
                )
                print(f"AMR system created: {amr_system}")
            except Exception as e:
                print(f"✗ Failed to create AMR system: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                raise

            if amr_system is None:
                raise ValueError("AMRFactory.create_amr returned None")

            print("✓ AMR system created successfully")

            # Now do the visualization...
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(10, 10))
            amr_system.plot_grid_structure(ax, show_levels=True)
            ax.set_title(f"AMR Grid Structure - {backend} backend")
            plt.show()

            # Show statistics
            stats = amr_system.get_statistics()
            info = f"AMR Statistics:\n"
            info += f"Total cells: {stats['total_cells']:,}\n"
            info += f"Efficiency: {stats.get('efficiency', 0):.1%}\n"
            messagebox.showinfo("AMR Preview Success", info)

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            print(f"\n✗ Preview failed with error: {type(e).__name__}: {e}")
            print(error_detail)
            messagebox.showerror(
                "Error",
                f"Failed to preview AMR grid:\n\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {str(e)}\n\n"
                f"Check console for details.",
            )

    def save_config(self):
        """Save current configuration to file."""
        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            defaultextension=".ini",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")],
        )
        if filename:
            # Implementation to save config
            self.update_status(f"Configuration saved to {filename}")

    def load_config_dialog(self):
        """Load configuration from file."""
        from tkinter import filedialog

        filename = filedialog.askopenfilename(
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
        )
        if filename:
            self.load_config(filename)
            self.update_status(f"Configuration loaded from {filename}")

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        from tkinter import messagebox

        if messagebox.askyesno(
            "Reset to Defaults",
            "Are you sure you want to reset all settings to defaults?",
        ):
            self.load_config("config_fv.ini")
            self.update_status("Settings reset to defaults")

    def update_status(self, message):
        """Update status text area."""
        self.status_text.insert("end", f"{message}\n")
        self.status_text.see("end")  # Auto-scroll to bottom

    def initialize_and_adapt_amr(self):
        """Initialize solution and perform AMR adaptation to show refined mesh."""
        try:
            # Check if AMR is enabled
            if not self.var_enable_amr.get():
                messagebox.showwarning("AMR Disabled", 
                    "Please enable AMR first to use adaptive mesh refinement.")
                return
            
            # Get mesh configuration
            nx = self.var_nx_cells.get()
            ny = self.var_ny_cells.get()
            
            # Warn if base grid is too fine
            if nx > 100 or ny > 100:
                response = messagebox.askyesno("Large Base Grid", 
                    f"Base grid is {nx}×{ny}. AMR works best with coarser base grids (e.g., 50×50).\n\n"
                    "A finer base grid may result in excessive refinement.\n\n"
                    "Continue anyway?")
                if not response:
                    return
            
            # Warn if base grid is not set properly
            if nx < 10 or ny < 10:
                messagebox.showwarning("Base Grid Too Coarse", 
                    f"Base grid is only {nx}×{ny}. Please set a reasonable base grid size "
                    "in the Grid tab first (e.g., 25×25 to 50×50).")
                return
            
            print("\n" + "="*50)
            print("Starting Initialize & Adapt process...")
            print("="*50)
            
            # Create mesh
            from mesh import FVMesh
            mesh = FVMesh(
                nx_cells=nx, 
                ny_cells=ny,
                plate_length=self.var_plate_length.get(), 
                plate_width=self.var_plate_width.get()
            )
            
            # Create solver
            from solver import FVHeatSolver
            solver = FVHeatSolver(
                mesh=mesh,
                alpha=self.var_alpha.get()
            )
            
            # Get AMR configuration
            amr_config = self.get_amr_config()
            backend = amr_config['backend']
            
            # Create AMR system
            print(f"\nCreating AMR system with backend: {backend}")
            print(f"AMR config: {amr_config}")
            
            if backend == 'simple':
                from amr.simple_amr import SimpleAMRRefactored
                amr_system = SimpleAMRRefactored(
                    base_solver=solver,
                    config=amr_config
                )
            elif backend == 'amrex':
                from amr.amrex_amr import AMReXAMR
                amr_system = AMReXAMR(
                    base_solver=solver,
                    config=amr_config
                )
            
            # Set initial condition
            print("\nSetting initial condition...")
            ic_params = {
                'background_temp': self.var_background_temp.get(),
                'hotspot_temp': self.var_hotspot_temp.get(),
                'center_x': self.var_center_x.get(),
                'center_y': self.var_center_y.get(),
                'hotspot_radius': self.var_hotspot_radius.get(),
            }
            
            from initial_conditions import get_fv_initial_condition
            ic = get_fv_initial_condition(self.var_ic_type.get(), ic_params)
            
            # Always set IC on the base solver
            ic.set(solver)
            print("✓ Initial condition set")
            
            # Initialize AMR system if needed
            if backend == 'simple':
                amr_system.base_initialized = True
            elif backend == 'amrex':
                # AMReX will handle initialization during adapt
                pass
            
            print("✓ Initial condition set")
            
            # Perform adaptation
            print("\nPerforming AMR adaptation based on initial condition...")
            
            if backend == 'simple':
                amr_system.adapt_to_initial_condition()
            elif backend == 'amrex':
                # AMReX adaptation
                amr_system.regrid(0)  # Regrid at level 0
            
            print("✓ AMR adaptation complete")
            
            # Visualize the adapted mesh with temperature field
            print("\nVisualizing adapted mesh with temperature field...")
            self.visualize_amr_solution(amr_system, amr_config, ic_params)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize and adapt:\n{str(e)}")
            import traceback
            traceback.print_exc()
        
    def visualize_amr_solution(self, amr_system, amr_config, ic_params):
        """Visualize the AMR-adapted mesh with the temperature field."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        from matplotlib.colors import ListedColormap
        
        backend = amr_config['backend']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Always start by showing the base solver's temperature field
        mesh = amr_system.base_solver.mesh
        T = amr_system.base_solver.T
        
        # Extract interior temperature
        T_interior = mesh.extract_interior(T)
        
        # Plot base level temperature
        X, Y = np.meshgrid(mesh.x_centers, mesh.y_centers)
        im = ax.pcolormesh(X, Y, T_interior, cmap='hot', shading='auto',
                        vmin=ic_params['background_temp'], 
                        vmax=ic_params['hotspot_temp'])
        
        if backend == 'simple':
            # Overlay refined patches for Simple AMR
            colors = ['blue', 'green', 'red', 'purple']
            for level in range(1, amr_system.max_levels):
                if level in amr_system.patches and amr_system.patches[level]:
                    for i, patch in enumerate(amr_system.patches[level]):
                        rect = patches.Rectangle(
                            (patch['x_min'], patch['y_min']),
                            patch['x_max'] - patch['x_min'],
                            patch['y_max'] - patch['y_min'],
                            linewidth=2, edgecolor=colors[min(level-1, len(colors)-1)],
                            facecolor='none', 
                            label=f'Level {level}' if i == 0 else ""
                        )
                        ax.add_patch(rect)
            
        elif backend == 'amrex':
            # For AMReX, show a message that detailed visualization is not available
            # But we can still show the base grid structure
            info_text = "AMReX Backend\nDetailed AMR visualization requires\nrunning the full simulation.\nShowing base grid with initial condition."
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # Show base grid structure if requested
            if self.var_show_amr_grid.get():
                # Draw a few grid lines to show base structure
                dx = mesh.plate_length / amr_config.get('max_grid_size', 32)
                dy = mesh.plate_width / amr_config.get('max_grid_size', 32)
                
                for x in np.arange(0, mesh.plate_length + dx/2, dx):
                    ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
                for y in np.arange(0, mesh.plate_width + dy/2, dy):
                    ax.axhline(y, color='gray', linewidth=0.5, alpha=0.5)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Temperature (K)')
        
        # Add hotspot circle
        if self.var_show_hotspot.get():
            circle = patches.Circle(
                (ic_params['center_x'], ic_params['center_y']), 
                ic_params['hotspot_radius'],
                fill=False, edgecolor='green', linewidth=2, linestyle='--',
                label='Initial Hotspot'
            )
            ax.add_patch(circle)
        
        # Add mesh lines if requested
        if self.var_show_mesh.get():
            # Show base mesh
            for x in mesh.x_faces[::5]:  # Show every 5th line to avoid clutter
                ax.axvline(x, color='gray', linewidth=0.5, alpha=0.3)
            for y in mesh.y_faces[::5]:
                ax.axhline(y, color='gray', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'AMR-Adapted Mesh with Temperature Field\n'
                    f'Backend: {backend}, Max Levels: {amr_config["max_levels"]}')
        ax.set_aspect('equal')
        ax.set_xlim(0, mesh.plate_length)
        ax.set_ylim(0, mesh.plate_width)
        
        if ax.get_legend_handles_labels()[0]:  # Only add legend if there are items
            ax.legend()
        
        # Add text info about adaptation
        info_text = f"Base Grid: {mesh.nx}×{mesh.ny}\n"
        
        if backend == 'simple' and hasattr(amr_system, 'patches'):
            total_cells = mesh.nx * mesh.ny
            # Count refined cells
            for level in range(1, amr_system.max_levels):
                if level in amr_system.patches:
                    for patch in amr_system.patches[level]:
                        if 'nx' in patch and 'ny' in patch:
                            refined_cells = ((patch['nx'] * patch['ny']) * 
                                        (amr_system.refinement_ratio ** (2 * level)))
                            total_cells += refined_cells
            info_text += f"Total Cells: ~{total_cells:,}\n"
        
        info_text += f"Temp Range: {ic_params['background_temp']:.0f}K - {ic_params['hotspot_temp']:.0f}K"
        
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    def setup_physics_based_amr(self):
        """Setup AMR with physics-based refinement."""
        # Create AMR configuration
        amr_config = {
            # Basic AMR parameters
            'max_levels': self.var_max_levels.get(),
            'refinement_ratio': self.var_refinement_ratio.get(),
            'max_grid_size': 16,  # Smaller for better interface capture
            'blocking_factor': 4,
            'n_error_buf': 1,
            'regrid_interval': self.var_regrid_interval.get(),
            
            # Physics-based refinement
            'refine_mode': 'physics_based',
            
            # Gradient indicator settings
            'gradient_factor': 2.0,  # How many std devs above mean
            'use_gradient': True,
            
            # Curvature indicator settings  
            'use_curvature': True,
            'curvature_threshold': 0.5,
            
            # Jump indicator settings
            'use_jump': True,
            'jump_threshold': 0.3,
            
            # Coherence indicator (optional - more expensive)
            'use_coherence': False,
            'coherence_threshold': 0.8,
            
            # Interface buffer
            'interface_buffer': 2,  # cells around interface
            
            # Thresholding strategy
            'use_statistical_threshold': True,
            'sigma_factor': 2.0,
            'use_otsu_threshold': True,
            'refine_threshold': 0.3,  # Fallback threshold
            'threshold_combine': 'or',  # 'or', 'and', 'majority'
            
            # Morphological cleanup
            'apply_morphology': True,
            'dilation_iter': 1,
            
            # Visualization
            'show_error_indicator': self.var_show_error.get(),
        }
    
        return amr_config

def main():
    """Main entry point."""
    root = tk.Tk()
    app = FVHeatSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
