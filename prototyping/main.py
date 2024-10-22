from mesh import ComputationalMesh
from visualize import animate_smooth_simulation
import numpy as np

mesh = ComputationalMesh("../data/rectangle_mesh.msh")
mesh.initialize_conditions(init_velocity=np.array([1.0, 0.0]), init_pressure=1.0)

# # Generate and save the series of pressure field images
# generate_simulation_visualization(mesh, total_timesteps=10, save_path="simulation_output")


# Run the smooth field animation
animate_smooth_simulation(mesh, total_timesteps=100, field="pressure", grid_size=200)