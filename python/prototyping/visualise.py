import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def plot_solution(centroids, solution):

    # drop the z coordinate from the centroids

    centroids = centroids[:, :2]

    # Create a grid
    grid_x, grid_y = np.mgrid[
        np.min(centroids[:, 0]):np.max(centroids[:, 0]):100j,
        np.min(centroids[:, 1]):np.max(centroids[:, 1]):100j
    ]
    
    # Interpolate the solution values on the grid
    grid_z = griddata(centroids, solution, (grid_x, grid_y), method='cubic')
    
    # Create a heatmap of the interpolated solution values
    plt.figure(figsize=(10, 4))
    heatmap = plt.imshow(grid_z.T, extent=(np.min(centroids[:, 0]), np.max(centroids[:, 0]),
                                           np.min(centroids[:, 1]), np.max(centroids[:, 1])),
                         origin='lower', cmap='viridis', aspect='auto')
    
    # Add a color bar
    plt.colorbar(heatmap, label='Solution Value (e.g., velocity or pressure)')
    
    # Set plot labels and title
    plt.title("CFD Simulation Results")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis("equal")
    plt.savefig("../data/solution_plot.png")
    
    # Show the plot
    plt.show()