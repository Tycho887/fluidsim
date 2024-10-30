import numpy as np
import meshio
import scipy.sparse as sps
from detect_neighbours import FindNeighbours


def load_mesh(file):
    """
    Reads the mesh file and returns a meshio object.
    """
    mesh = meshio.read(file)

    return mesh

def boundary_nodes(mesh):
        
    boundary_tags = mesh.cell_data_dict['gmsh:physical']

    inlet_cells = mesh.cells_dict["line"][boundary_tags["line"] == 1]  # For inlet
    outlet_cells = mesh.cells_dict["line"][boundary_tags["line"] == 2]  # For outlet
    wall_cells = mesh.cells_dict["line"][boundary_tags["line"] == 3]  # For walls

    # Unique boundary nodes for each type
    inlet_nodes = np.unique(inlet_cells)
    outlet_nodes = np.unique(outlet_cells)
    wall_nodes = np.unique(wall_cells)

    return inlet_nodes, outlet_nodes, wall_nodes

def generate_sparse_matrices(mesh):

    num_cells = len(mesh.cells_dict["triangle"])

    # Initialize the sparse matrices

    flux_matrix = sps.lil_matrix((num_cells, num_cells))

    inlet_matrix = sps.lil_matrix((num_cells, num_cells))
    outlet_matrix = sps.lil_matrix((num_cells, num_cells))
    wall_matrix = sps.lil_matrix((num_cells, num_cells))

    return flux_matrix, inlet_matrix, outlet_matrix, wall_matrix



def timing_decorator(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken: {end - start}")
        return result
    return wrapper



@timing_decorator
def find_neighbours_simple(mesh):
    """
    Finds the neighbours of each cell in the mesh.
    """

    triangle_cells = mesh.cells_dict["triangle"]

    num_cells = len(triangle_cells)

    print(f"Number of cells: {num_cells}")

    neighbour_dict = {i: list() for i in range(num_cells)}

    for id, cell_points in enumerate(triangle_cells):
        # if a cell has three neighbours, then it is an interior cell
        if len(neighbour_dict[id]) == 3:
            continue
        for id2, cell_points2 in enumerate(triangle_cells):
            if id != id2:
                if np.intersect1d(cell_points, cell_points2).size == 2:
                    neighbour_dict[id].append(id2)

    return neighbour_dict


def detect_neighbours(mesh):
    """
    Finds the neighbours of each cell in the mesh.
    """
    return FindNeighbours(mesh).get_neighbour_matrix()



if __name__ == "__main__":
    mesh = load_mesh("../data/msh/simple.msh")

    print(find_neighbours_simple(mesh))

    # print(mesh.cells_dict["triangle"])
    