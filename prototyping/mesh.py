from precompute import Topology
import meshio
import numpy as np

class ComputationalMesh:
    """
    Object that stores the vectorised mesh data in object-of-vectors format
    :param file: The file path to the msh file
    :attr normal_vector_tensor: The normal vector tensor
    :attr neighbour_matrix: The neighbour matrix
    :attr area_vector: Stores area for each cell
    :attr density_vector: Stores the density for each cell
    :attr pressure_vector: Stores the pressure for each cell
    :attr velocity_matrix: stores the velocity vector for each cell
    """
    def __init__(self, file):

        msh = self.read_mesh(file)

        topology = Topology(msh)
        topology.find_neighbours().find_constants()

        self._normal_vector_tensor = topology._normal_vector_tensor
        self._neighbour_matrix = topology._neighbour_matrix
        self._area_vector = topology._area_vector
        self._density_vector = np.zeros(len(topology.cells), dtype=np.float64)
        self._pressure_vector = np.zeros(len(topology.cells), dtype=np.float64)
        self._velocity_matrix = np.zeros((len(topology.cells), 2), dtype=np.float64)

    def read_mesh(self,file):
        """
        Reads the mesh file
        :param file: The file path
        :return: The mesh object
        """
        msh = meshio.read(file)
        return msh

    @property
    def normal_vectors(self):
        return self._normal_vector_tensor
    
    @property
    def neighbour_ids(self):
        return self._neighbour_matrix
    
    @property
    def area(self):
        return self._area_vector
    
    @property
    def density(self):
        return self._density_vector
    
    @property
    def pressure(self):
        return self._pressure_vector
    
    @property
    def velocity(self):
        return self._velocity_matrix
    
    def update_pressure(self, pressure_vector):
        """
        Updates the pressure vector
        :param pressure_vector: The pressure vector
        """
        self._pressure_vector = pressure_vector

    def update_velocity(self, velocity_matrix):
        """
        Updates the velocity matrix
        :param velocity_matrix: The velocity matrix
        """
        self._velocity_matrix = velocity_matrix

    def update_density(self, density_vector):
        """
        Updates the density vector
        :param density_vector: The density vector
        """
        self._density_vector = density_vector