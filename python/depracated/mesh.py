from precompute import Topology
import numpy as np
import meshio

class ComputationalMesh:
    def __init__(self, file):
        msh = self.read_mesh(file)
        topology = Topology(msh)
        topology.find_neighbours_and_compute_normals()

        self._centroids = topology._centroids
        self._points = topology._points

        self._normal_vector_tensor = topology._normal_vectors
        self._neighbour_matrix = topology._neighbour_matrix
        self._area_vector = topology._area_vector
        
        # Fields (density, pressure, velocity)
        self._density_vector = np.ones(len(topology.cells))  # Initialize with some default values
        self._pressure_vector = np.zeros(len(topology.cells))
        self._velocity_matrix = np.zeros((len(topology.cells), 2))  # 2D velocity (x, y)
        self._viscosity = 1.0  # Default viscosity value

    def set_boundary_conditions

    def read_mesh(self, file):
        """
        Reads the mesh file and returns a meshio object.
        """
        return meshio.read(file)
    
    @property
    def centroids(self):
        return self._centroids
    
    @property
    def points(self):
        return self._points
    
    @property
    def normal_vectors(self):
        return self._normal_vector_tensor
    
    @property
    def neighbours(self):
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
    
    @property
    def viscosity(self):
        return self._viscosity
    
    def update_density(self, density):
        self._density_vector = density

    def update_pressure(self, pressure):
        self._pressure_vector = pressure
    
    def update_velocity(self, velocity):
        self._velocity_matrix = velocity
    
    # Other methods omitted for brevity...
