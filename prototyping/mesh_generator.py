import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Step 1: Define the rectangle dimensions (width, height)
width, height = 2.0, 1.0

# Step 2: Create a grid of points over the rectangle
x = np.linspace(0, width, 30)  # 10 points along the x-axis
y = np.linspace(0, height, 15)  # 5 points along the y-axis
points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

# Step 3: Perform Delaunay triangulation to create a triangular mesh
tri = Delaunay(points)

# Step 4: Visualize the mesh (for checking the mesh)
fig, ax = plt.subplots()
ax.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')
ax.plot(points[:, 0], points[:, 1], 'o', color='red')
ax.set_aspect('equal', adjustable='box')
plt.title('Triangular Mesh of the Rectangle')
plt.savefig('rectangle_mesh_plot.png')  # Save plot as image
plt.close(fig)  # Close figure to avoid interactive backend issues

# Step 5: Save the mesh in a format compatible with C++
# Cells as a list of tuples ("triangle", simplices)
cells = [("triangle", tri.simplices)]

# Create mesh object for saving
mesh = meshio.Mesh(points, cells)

# Save the mesh as a .msh file (Gmsh format, widely used and supported by C++)
meshio.write("../data/rectangle_mesh.msh", mesh)

print("Mesh created and saved to 'rectangle_mesh.msh' and plot saved to 'rectangle_mesh_plot.png'")
