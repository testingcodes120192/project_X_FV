# animation.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def create_fv_animation_from_history(solver, postprocessor, frame_data_list,
                                   T_min_fixed=None, T_max_fixed=None,
                                   save_animation=True, filename=None, writer='gif'):
    """
    Create animation from pre-collected frame data.
    
    Parameters
    ----------
    solver : FVHeatSolver
        Solver object (for configuration info)
    postprocessor : FVPostProcessor
        Post-processor object
    frame_data_list : list
        List of frame data dictionaries
    T_min_fixed : float, optional
        Fixed minimum temperature for colorbar
    T_max_fixed : float, optional
        Fixed maximum temperature for colorbar
    save_animation : bool
        Save animation to file
    filename : str, optional
        Output filename (auto-generated if None)
    writer : str
        Animation writer ('gif' or 'mp4')
        
    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object
    """
    print("Creating animation from collected data...")
    
    frames = frame_data_list
    n_frames = len(frames)
    
    if n_frames == 0:
        raise ValueError("No frame data provided")
    
    # Compute temperature range if not fixed
    if T_min_fixed is None:
        T_min_fixed = min(np.min(frame['T']) for frame in frames)
    if T_max_fixed is None:
        T_max_fixed = max(np.max(frame['T']) for frame in frames)
        
    # Add some margin to fixed ranges
    T_range = T_max_fixed - T_min_fixed
    T_min_fixed -= 0.01 * T_range
    T_max_fixed += 0.01 * T_range
    
    # Create figure
    if solver.enable_reactions:
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[20, 1])
        ax_temp = fig.add_subplot(gs[0, 0])
        ax_cbar_temp = fig.add_subplot(gs[0, 1])
        ax_lambda = fig.add_subplot(gs[1, 0])
        ax_cbar_lambda = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[20, 1])
        ax_temp = fig.add_subplot(gs[0, 0])
        ax_cbar_temp = fig.add_subplot(gs[0, 1])
    
    # Setup temperature plot
    x_plot = frames[0]['x']
    y_plot = frames[0]['y']
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
    
    # Initial temperature plot
    levels_temp = np.linspace(T_min_fixed, T_max_fixed, 51)
    cf_temp = ax_temp.contourf(X_plot, Y_plot, frames[0]['T'], 
                               levels=levels_temp, cmap='hot', extend='both')
    cbar_temp = plt.colorbar(cf_temp, cax=ax_cbar_temp)
    cbar_temp.set_label('Temperature (K)', fontsize=12)
    
    ax_temp.set_xlabel('X (m)', fontsize=12)
    ax_temp.set_ylabel('Y (m)', fontsize=12)
    ax_temp.set_aspect('equal')
    title_temp = ax_temp.set_title('Temperature Distribution - t = 0.000 s', fontsize=14)
    
    # Stats text
    stats_text = ax_temp.text(0.02, 0.98, '', transform=ax_temp.transAxes,
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Setup reaction plot if enabled
    if solver.enable_reactions:
        levels_lambda = np.linspace(0, 1, 21)
        cf_lambda = ax_lambda.contourf(X_plot, Y_plot, frames[0].get('lambda_rxn', np.zeros_like(frames[0]['T'])),
                                       levels=levels_lambda, cmap='viridis', extend='neither')
        cbar_lambda = plt.colorbar(cf_lambda, cax=ax_cbar_lambda)
        cbar_lambda.set_label('Reaction Progress', fontsize=12)
        
        ax_lambda.set_xlabel('X (m)', fontsize=12)
        ax_lambda.set_ylabel('Y (m)', fontsize=12)
        ax_lambda.set_aspect('equal')
        title_lambda = ax_lambda.set_title('Reaction Progress - t = 0.000 s', fontsize=14)
    
    plt.tight_layout()
    
    def update(frame_idx):
        """Update function for animation."""
        frame = frames[frame_idx]
        t = frame['time']
        
        # Clear and redraw temperature
        ax_temp.clear()
        cf_temp = ax_temp.contourf(X_plot, Y_plot, frame['T'],
                                   levels=levels_temp, cmap='hot', extend='both')
        
        # Format time for title
        if t < 1e-6:
            time_str = f'{t*1e9:.1f} ns'
        elif t < 1e-3:
            time_str = f'{t*1e6:.1f} μs'
        elif t < 1:
            time_str = f'{t*1e3:.1f} ms'
        else:
            time_str = f'{t:.3f} s'
            
        ax_temp.set_title(f'Temperature Distribution - t = {time_str}', fontsize=14)
        ax_temp.set_xlabel('X (m)', fontsize=12)
        ax_temp.set_ylabel('Y (m)', fontsize=12)
        ax_temp.set_aspect('equal')
        
        # Update stats
        stats = frame['stats']
        stats_str = (f"Max T: {stats['T_max']:.1f} K\n"
                    f"Min T: {stats['T_min']:.1f} K\n"
                    f"Avg T: {stats['T_mean']:.1f} K")
        
        stats_text = ax_temp.text(0.02, 0.98, stats_str, transform=ax_temp.transAxes,
                                 verticalalignment='top', fontsize=10,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Update reaction plot if enabled
        if solver.enable_reactions and 'lambda_rxn' in frame:
            ax_lambda.clear()
            cf_lambda = ax_lambda.contourf(X_plot, Y_plot, frame['lambda_rxn'],
                                          levels=levels_lambda, cmap='viridis', extend='neither')
            ax_lambda.set_title(f'Reaction Progress - t = {time_str}', fontsize=14)
            ax_lambda.set_xlabel('X (m)', fontsize=12)
            ax_lambda.set_ylabel('Y (m)', fontsize=12)
            ax_lambda.set_aspect('equal')
            
            # Add reaction stats
            if 'lambda_mean' in stats:
                rxn_str = (f"Mean λ: {stats['lambda_mean']:.3f}\n"
                          f"Complete: {stats['reaction_complete_percent']:.1f}%")
                ax_lambda.text(0.02, 0.98, rxn_str, transform=ax_lambda.transAxes,
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return [ax_temp, ax_lambda] if solver.enable_reactions else [ax_temp]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=200, blit=False, repeat=True)
    
    # Save animation
    if save_animation:
        if filename is None:
            if solver.enable_reactions:
                filename = f'fv_heat_diffusion_reactions_{solver.spatial_order}th_order'
            else:
                filename = f'fv_heat_diffusion_{solver.spatial_order}th_order'
                
        if writer == 'gif':
            writer_obj = animation.PillowWriter(fps=5)
            filename += '.gif'
        else:  # mp4
            writer_obj = animation.FFMpegWriter(fps=10, bitrate=2000)
            filename += '.mp4'
            
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer=writer_obj)
        print("Animation saved!")
    
    return anim


def create_fv_animation(solver, postprocessor, total_time, dt, 
                       nx_plot=100, ny_plot=100,
                       frame_skip=10, T_min_fixed=None, T_max_fixed=None,
                       save_animation=True, filename=None, writer='gif'):
    """
    Create animation of FV heat diffusion solution.
    
    DEPRECATED: Use simulation with frame collection instead.
    """
    print("WARNING: This function performs duplicate computation.")
    print("Consider using the main simulation with frame collection enabled.")
    
    # Original implementation kept for compatibility
    print("Preparing animation data...")
    
    # Calculate frame times
    total_steps = int(total_time / dt)
    frame_steps = list(range(0, total_steps + 1, frame_skip))
    if total_steps not in frame_steps:
        frame_steps.append(total_steps)
    
    times = [min(step * dt, total_time) for step in frame_steps]
    n_frames = len(times)
    
    # Pre-compute all frames
    frames = []
    actual_times = []
    
    # Reset solver to initial condition
    solver.current_time = 0.0
    solver.step_count = 0
    
    # Store initial frame
    frame_data = postprocessor.prepare_animation_frame(solver, nx_plot, ny_plot)
    frame_data['time'] = 0.0
    frames.append(frame_data)
    actual_times.append(0.0)
    
    # Generate frames
    print(f"Generating {n_frames} frames...")
    for target_time in tqdm(times[1:]):
        # Advance to target time
        actual_time = solver.advance_to_time(target_time, dt, show_progress=False)
        
        # Store frame
        frame_data = postprocessor.prepare_animation_frame(solver, nx_plot, ny_plot)
        frame_data['time'] = actual_time
        frames.append(frame_data)
        actual_times.append(actual_time)
    
    # Use the new function
    return create_fv_animation_from_history(solver, postprocessor, frames,
                                          T_min_fixed, T_max_fixed,
                                          save_animation, filename, writer)


def plot_fv_solution_snapshots_from_history(solutions, solver, postprocessor,
                                           T_min_fixed=None, T_max_fixed=None,
                                           show_mesh=False, show_hotspot=False,
                                           hotspot_params=None):
    """
    Plot solution snapshots from pre-collected data.
    
    Parameters
    ----------
    solutions : list
        List of (time, solution_dict) tuples
    solver : FVHeatSolver
        Solver object (for mesh info)
    postprocessor : FVPostProcessor
        Post-processor object
    T_min_fixed : float, optional
        Fixed minimum temperature for colorbar
    T_max_fixed : float, optional
        Fixed maximum temperature for colorbar
    show_mesh : bool
        Show mesh lines overlay
    show_hotspot : bool
        Show initial hotspot boundary
    hotspot_params : dict
        Hotspot parameters (center_x, center_y, radius)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.patches as patches
    
    n_times = len(solutions)
    time_points = [t for t, _ in solutions]
    
    # Create figure
    if solver.enable_reactions:
        fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 8))
        if n_times == 1:
            axes = axes.reshape(2, 1)
    else:
        fig, axes = plt.subplots(1, n_times, figsize=(4*n_times, 4))
        if n_times == 1:
            axes = axes.reshape(1, 1)
    
    # Determine temperature range
    all_temps = []
    for _, sol in solutions:
        all_temps.extend(sol['T'].flatten())
    
    if T_min_fixed is None:
        T_min_fixed = min(all_temps)
    if T_max_fixed is None:
        T_max_fixed = max(all_temps)
        
    # Plot each snapshot
    X_plot, Y_plot = np.meshgrid(solutions[0][1]['x'], solutions[0][1]['y'])
    levels = np.linspace(T_min_fixed, T_max_fixed, 51)
    
    for idx, (t, sol) in enumerate(solutions):
        # Temperature plot
        if solver.enable_reactions:
            ax_temp = axes[0, idx]
            ax_lambda = axes[1, idx]
        else:
            ax_temp = axes[idx] if n_times > 1 else axes[0]
            
        cf = ax_temp.contourf(X_plot, Y_plot, sol['T'], 
                             levels=levels, cmap='hot', extend='both')
        
        # Add mesh overlay if requested
        if show_mesh:
            # Draw vertical lines
            for i in range(solver.mesh.nx + 1):
                ax_temp.axvline(x=solver.mesh.x_faces[i], color='black', 
                               linewidth=0.5, alpha=0.3)
            # Draw horizontal lines
            for j in range(solver.mesh.ny + 1):
                ax_temp.axhline(y=solver.mesh.y_faces[j], color='black', 
                               linewidth=0.5, alpha=0.3)
        
        # Add hotspot boundary if requested
        if show_hotspot and hotspot_params is not None:
            center_x = hotspot_params.get('center_x', solver.mesh.plate_length/2)
            center_y = hotspot_params.get('center_y', solver.mesh.plate_width/2)
            radius = hotspot_params.get('radius', 0.05)
            
            circle = patches.Circle((center_x, center_y), radius,
                                  linewidth=2, edgecolor='lime',
                                  facecolor='none', linestyle='--')
            ax_temp.add_patch(circle)
        
        # Format time for title
        if t == 0:
            time_str = 't = 0 s (Initial)'
        elif t < 1e-6:
            time_str = f't = {t*1e9:.1f} ns'
        elif t < 1e-3:
            time_str = f't = {t*1e6:.1f} μs'
        elif t < 1:
            time_str = f't = {t*1e3:.1f} ms'
        else:
            time_str = f't = {t:.3f} s'
            
        ax_temp.set_title(time_str, fontsize=12)
        ax_temp.set_xlabel('X (m)')
        ax_temp.set_ylabel('Y (m)')
        ax_temp.set_aspect('equal')
        
        # Colorbar for last plot
        if idx == n_times - 1:
            cbar = plt.colorbar(cf, ax=ax_temp)
            cbar.set_label('Temperature (K)')
            
        # Reaction plot if enabled
        if solver.enable_reactions and 'lambda_rxn' in sol:
            cf_lambda = ax_lambda.contourf(X_plot, Y_plot, sol['lambda_rxn'],
                                          levels=np.linspace(0, 1, 21),
                                          cmap='viridis')
            ax_lambda.set_title(f'λ at {time_str}', fontsize=12)
            ax_lambda.set_xlabel('X (m)')
            ax_lambda.set_ylabel('Y (m)')
            ax_lambda.set_aspect('equal')
            
            if idx == n_times - 1:
                cbar = plt.colorbar(cf_lambda, ax=ax_lambda)
                cbar.set_label('Reaction Progress')
                
    plt.tight_layout()
    return fig


def plot_fv_solution_snapshots(solver, postprocessor, time_points, 
                              nx_plot=101, ny_plot=101,
                              T_min_fixed=None, T_max_fixed=None,
                              show_mesh=False, show_hotspot=False,
                              hotspot_params=None):
    """
    Plot solution at specific time points.
    
    Parameters
    ----------
    solver : FVHeatSolver
        Solver object
    postprocessor : FVPostProcessor
        Post-processor object
    time_points : list
        List of times to plot
    nx_plot : int
        Plotting resolution in x
    ny_plot : int
        Plotting resolution in y
    T_min_fixed : float, optional
        Fixed minimum temperature for colorbar
    T_max_fixed : float, optional
        Fixed maximum temperature for colorbar
    show_mesh : bool
        Show mesh lines overlay
    show_hotspot : bool
        Show initial hotspot boundary
    hotspot_params : dict
        Hotspot parameters (center_x, center_y, radius)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.patches as patches
    
    n_times = len(time_points)
    
    # Create figure
    if solver.enable_reactions:
        fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 8))
        if n_times == 1:
            axes = axes.reshape(2, 1)
    else:
        fig, axes = plt.subplots(1, n_times, figsize=(4*n_times, 4))
        if n_times == 1:
            axes = axes.reshape(1, 1)
            
    # Store solutions
    solutions = []
    
    # Reset solver
    solver.current_time = 0.0
    solver.step_count = 0
    
    # Get initial solution
    sol = postprocessor.get_solution_on_grid(solver, nx_plot, ny_plot, smooth=True)
    solutions.append(sol)
    
    # Determine temperature range
    if T_min_fixed is None:
        T_min_fixed = min(np.min(sol['T']) for sol in solutions)
    if T_max_fixed is None:
        T_max_fixed = max(np.max(sol['T']) for sol in solutions)
        
    # Advance to each time point
    dt = solver.compute_stable_timestep()
    for t in time_points[1:]:
        solver.advance_to_time(t, dt, show_progress=True)
        sol = postprocessor.get_solution_on_grid(solver, nx_plot, ny_plot, smooth=True)
        solutions.append(sol)
        
    # Plot each snapshot
    X_plot, Y_plot = np.meshgrid(solutions[0]['x'], solutions[0]['y'])
    levels = np.linspace(T_min_fixed, T_max_fixed, 51)
    
    for idx, (t, sol) in enumerate(zip(time_points, solutions)):
        # Temperature plot
        if solver.enable_reactions:
            ax_temp = axes[0, idx]
            ax_lambda = axes[1, idx]
        else:
            ax_temp = axes[idx] if n_times > 1 else axes[0]
            
        cf = ax_temp.contourf(X_plot, Y_plot, sol['T'], 
                             levels=levels, cmap='hot', extend='both')
        
        # Add mesh overlay if requested
        if show_mesh:
            # Draw vertical lines
            for i in range(solver.mesh.nx + 1):
                ax_temp.axvline(x=solver.mesh.x_faces[i], color='black', 
                               linewidth=0.5, alpha=0.3)
            # Draw horizontal lines
            for j in range(solver.mesh.ny + 1):
                ax_temp.axhline(y=solver.mesh.y_faces[j], color='black', 
                               linewidth=0.5, alpha=0.3)
        
        # Add hotspot boundary if requested
        if show_hotspot and hotspot_params is not None:
            center_x = hotspot_params.get('center_x', solver.mesh.plate_length/2)
            center_y = hotspot_params.get('center_y', solver.mesh.plate_width/2)
            radius = hotspot_params.get('radius', 0.05)
            
            circle = patches.Circle((center_x, center_y), radius,
                                  linewidth=2, edgecolor='lime',
                                  facecolor='none', linestyle='--')
            ax_temp.add_patch(circle)
        
        # Format time for title
        if t == 0:
            time_str = 't = 0 s (Initial)'
        elif t < 1e-6:
            time_str = f't = {t*1e9:.1f} ns'
        elif t < 1e-3:
            time_str = f't = {t*1e6:.1f} μs'
        elif t < 1:
            time_str = f't = {t*1e3:.1f} ms'
        else:
            time_str = f't = {t:.3f} s'
            
        ax_temp.set_title(time_str, fontsize=12)
        ax_temp.set_xlabel('X (m)')
        ax_temp.set_ylabel('Y (m)')
        ax_temp.set_aspect('equal')
        
        # Colorbar for last plot
        if idx == n_times - 1:
            cbar = plt.colorbar(cf, ax=ax_temp)
            cbar.set_label('Temperature (K)')
            
        # Reaction plot if enabled
        if solver.enable_reactions and 'lambda_rxn' in sol:
            cf_lambda = ax_lambda.contourf(X_plot, Y_plot, sol['lambda_rxn'],
                                          levels=np.linspace(0, 1, 21),
                                          cmap='viridis')
            ax_lambda.set_title(f'λ at {time_str}', fontsize=12)
            ax_lambda.set_xlabel('X (m)')
            ax_lambda.set_ylabel('Y (m)')
            ax_lambda.set_aspect('equal')
            
            if idx == n_times - 1:
                cbar = plt.colorbar(cf_lambda, ax=ax_lambda)
                cbar.set_label('Reaction Progress')
                
    plt.tight_layout()
    return fig


def plot_fv_centerlines(solver, postprocessor, time_points, direction='both',
                       show_mesh_lines=False, show_hotspot_marker=False,
                       hotspot_params=None):
    """
    DEPRECATED: Use solver with centerline collection instead.
    
    This function performs duplicate computation.
    """
    print("WARNING: This function performs duplicate computation.")
    print("Use the main simulation with centerline collection enabled instead.")
    
    # Collect centerline data
    solver_states = []
    
    # Reset solver
    solver.current_time = 0.0
    solver.step_count = 0
    
    # Initial data - store full field for centerline extraction
    solver_states.append({
        'T': solver.T.copy(),
        'time': 0.0
    })
    
    # Advance and collect data
    dt = solver.compute_stable_timestep()
    for t in time_points[1:]:
        solver.advance_to_time(t, dt, show_progress=False)
        solver_states.append({
            'T': solver.T.copy(),
            'time': solver.current_time
        })
    
    # Plot using postprocessor
    fig, axes = postprocessor.plot_centerlines_evolution(
        solver_states, time_points, direction,
        show_mesh_lines, show_hotspot_marker, hotspot_params
    )
    
    return fig


def plot_convergence_study(convergence_data):
    """
    Plot convergence study results.
    
    Parameters
    ----------
    convergence_data : dict
        Output from postprocessor.create_convergence_data()
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # L2 error
    if 'errors_L2' in convergence_data:
        ax1.loglog(convergence_data['cell_sizes'], convergence_data['errors_L2'], 
                   'bo-', label='L2 Error', markersize=8)
        
        # Add reference slopes
        h = np.array(convergence_data['cell_sizes'])
        ax1.loglog(h, h**2 * convergence_data['errors_L2'][0] / h[0]**2, 
                   'k--', alpha=0.5, label='2nd order')
        ax1.loglog(h, h**5 * convergence_data['errors_L2'][0] / h[0]**5, 
                   'k:', alpha=0.5, label='5th order')
        
        ax1.set_xlabel('Cell Size (m)')
        ax1.set_ylabel('L2 Error')
        ax1.set_title('L2 Error Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
    # L∞ error
    if 'errors_Linf' in convergence_data:
        ax2.loglog(convergence_data['cell_sizes'], convergence_data['errors_Linf'], 
                   'ro-', label='L∞ Error', markersize=8)
        
        # Add reference slopes
        h = np.array(convergence_data['cell_sizes'])
        ax2.loglog(h, h**2 * convergence_data['errors_Linf'][0] / h[0]**2, 
                   'k--', alpha=0.5, label='2nd order')
        ax2.loglog(h, h**5 * convergence_data['errors_Linf'][0] / h[0]**5, 
                   'k:', alpha=0.5, label='5th order')
        
        ax2.set_xlabel('Cell Size (m)')
        ax2.set_ylabel('L∞ Error')
        ax2.set_title('L∞ Error Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    return fig