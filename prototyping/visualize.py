import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation

def plot_pressure_field(mesh, timestep, save_path=None):
    """
    Plots the pressure field on the mesh for a given time step.
    
    :param mesh: ComputationalMesh object
    :param timestep: Current time step (for labeling)
    :param save_path: Path to save the figure (optional)
    """
    centroids = mesh.centroids
    pressure = mesh.pressure

    plt.figure(figsize=(8, 6))
    
    # Create a scatter plot of pressure at each centroid
    scatter = plt.scatter(centroids[:, 0], centroids[:, 1], c=pressure, cmap='viridis', s=50)
    
    plt.colorbar(scatter, label='Pressure')
    plt.title(f'Pressure Field at Timestep {timestep}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    if save_path:
        plt.savefig(f"{save_path}/pressure_timestep_{timestep}.png")
    else:
        plt.show()

    plt.close()  # Close the figure to avoid memory issues when plotting in a loop

def plot_smooth_field(mesh, timestep, field="pressure", save_path=None, grid_size=100):
    """
    Plots a smooth interpolated field (pressure or velocity) over the mesh.
    
    :param mesh: ComputationalMesh object
    :param timestep: Current time step (for labeling)
    :param field: The field to plot, either 'pressure' or 'velocity'
    :param save_path: Path to save the figure (optional)
    :param grid_size: Number of points in the grid for interpolation (higher = smoother)
    """
    centroids = mesh.centroids
    if field == "pressure":
        values = mesh.pressure
    elif field == "velocity":
        values = np.linalg.norm(mesh.velocity, axis=1)  # Use magnitude of velocity for visualization

    # Create a grid over the mesh domain
    x_min, x_max = centroids[:, 0].min(), centroids[:, 0].max()
    y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
    
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    # Interpolate the field values on the regular grid
    grid_values = griddata(centroids, values, (grid_x, grid_y), method='cubic')

    # Plot the interpolated field as a contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y, grid_values, cmap='viridis', levels=100)
    plt.colorbar(contour, label=field.capitalize())
    plt.title(f'{field.capitalize()} Field at Timestep {timestep}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    if save_path:
        plt.savefig(f"{save_path}/{field}_timestep_{timestep}.png")
    else:
        plt.show()

    plt.close()


def generate_simulation_visualization(mesh, total_timesteps, save_path=None):
    """
    Generates and saves a series of plots that show the evolution of the pressure field.
    
    :param mesh: ComputationalMesh object
    :param total_timesteps: Total number of time steps
    :param save_path: Path to save images (optional, default is None)
    """
    for t in range(total_timesteps):
        # Simulate one time step (this assumes your simulate method advances by one step)
        mesh.simulate(time_steps=1, dt=0.01)
        
        # Plot and save the pressure field for each time step
        plot_smooth_field(mesh, t, save_path=save_path)

def animate_smooth_simulation(mesh, total_timesteps, field="pressure", grid_size=100, interval=100):
    """
    Creates an animation of the smooth simulation field (pressure or velocity).
    
    :param mesh: ComputationalMesh object
    :param total_timesteps: Total number of time steps
    :param field: Field to visualize ('pressure' or 'velocity')
    :param grid_size: Number of points in the grid for interpolation
    :param interval: Interval between frames in milliseconds
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    centroids = mesh.centroids

    # Create a grid over the mesh domain
    x_min, x_max = centroids[:, 0].min(), centroids[:, 0].max()
    y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    def update(t):
        # Run simulation for one time step
        mesh.simulate(time_steps=1, dt=0.01)
        
        if field == "pressure":
            values = mesh.pressure
        elif field == "velocity":
            values = np.linalg.norm(mesh.velocity, axis=1)

        # Interpolate the field values
        grid_values = griddata(centroids, values, (grid_x, grid_y), method='cubic')
        
        # Clear the axis and plot the new field
        ax.clear()
        contour = ax.contourf(grid_x, grid_y, grid_values, cmap='viridis', levels=100)
        ax.set_title(f'{field.capitalize()} Field at Timestep {t}')
        return contour,

    # plt.colorbar(contour, ax=ax, label=field.capitalize())

    ani = FuncAnimation(fig, update, frames=total_timesteps, interval=interval, blit=False)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be

    ani.save(f"../data/{field}_simulation.mp4", writer='ffmpeg', fps=10)

    plt.show()

