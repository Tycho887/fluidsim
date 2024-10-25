import numpy as np
import meshio
import scipy.sparse as sps


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