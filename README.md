# project_X_FV

# FV Heat Diffusion Simulator

A Python-based Finite Volume solver for 2D heat diffusion problems with optional chemical reactions and adaptive mesh refinement.

## Features

- **Finite Volume Method** with 1st, 2nd, and 5th order (WENO) spatial discretization
- **Time Integration**: Forward Euler and 3rd order Runge-Kutta
- **Initial Conditions**: 
  - Circular hotspot
  - Gaussian pulse
  - Multiple hotspots
  - Image-based temperature fields
- **Optional Chemical Reactions** (prepared for plugin system)
- **Simple Adaptive Mesh Refinement** (experimental)
- **GUI Interface** with tabbed configuration
- **Visualization**: Snapshots and animations

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt