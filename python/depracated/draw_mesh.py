import matplotlib.pyplot as plt
import meshio


def visualise_mesh(file):
    msh = meshio.read(file)
    points = msh.points
    cells = msh.cells
    for cell in cells:
        if cell.type == "triangle":
            triangles = points[cell.data]
            plt.plot(triangles[:, 0], triangles[:, 1], 'k-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    visualise_mesh("../data/msh/pipe.msh")