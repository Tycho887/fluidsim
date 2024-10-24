import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def make_rectangle(width, height, x_dim, y_dim):
    x = np.linspace(0, width, x_dim)  # 10 points along the x-axis
    y = np.linspace(0, height, y_dim)  # 5 points along the y-axis
    points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    return points

def make_rectangle_with_hole(width, height, x_dim, y_dim, hole_radius):
    # Rectangle with a hole in the center, to be used to study vortex formation in the pipe
    x = np.linspace(0, width, x_dim)  # 10 points along the x-axis
    y = np.linspace(0, height, y_dim)  # 5 points along the y-axis


def Delunay_triangulation(points):
    tri = Delaunay(points)
    return tri

def plot_mesh(points, tri, title):
    fig, ax = plt.subplots()
    ax.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')
    ax.plot(points[:, 0], points[:, 1], 'o', color='red')
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"{title}")
    plt.show()

def save_mesh(points, tri, filename):
    cells = [("triangle", tri.simplices)]
    mesh = meshio.Mesh(points, cells)
    meshio.write(filename, mesh)

if __name__ == "__main__":
    # Make rectangle mesh
    points = make_rectangle(width=1.0, height=1.0, x_dim=20, y_dim=5)
    tri = Delunay_triangulation(points)
    plot_mesh(points, tri, "Rectangle Mesh")
    save_mesh(points, tri, "rectangle_mesh.msh")


print("Mesh created and saved to 'rectangle_mesh.msh' and plot saved to 'rectangle_mesh_plot.png'")
