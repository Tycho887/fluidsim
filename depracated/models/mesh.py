from abc import abstractmethod
import src.utils as utils
from src.models.cell import Cell
import meshio
import numpy as np

def vectorize_neighbour_data(cell):
    """
    Vectorize the neighbour data
    :param cell: The cell
    :param neighbour: The neighbour
    :return: The vectorized neighbour data
    """
    # we cant use isinstance because of circular imports
    assert hasattr(cell, "neighbour_data"), "The cell must have a neighbour_data attribute"
    assert hasattr(cell, "type"), "The cell must have a type attribute"
    assert hasattr(cell, "area"), "The cell must have an area attribute"
    
    normal_matrix = np.zeros((3, 2), dtype=np.float64)
    neighbor_index_row = np.full(3, -1, dtype=np.int64)
    area = 0

    for i, neighbour_index in enumerate(cell.neighbour_data.keys()):
        if cell.type != "triangle":
            continue
        normal_matrix[i] = cell.neighbour_data[neighbour_index]['normal']
        neighbor_index_row[i] = neighbour_index
        area = cell.area

    return normal_matrix, neighbor_index_row, area

class Mesh:
    """
    The mesh class
    :param meshio_obj: The meshio object
    :method cell_factory: Creates the cells from the mesh file
    :method find_neighbours: Finds the neighbours of the cells
    :method find_constants: Finds the dot product between the neighbours
    :method _vectorise_constants: Finds the dot product between the neighbours
    :attr cells: The cells
    :attr points: The points
    :attr dot_matrix: The dot matrix
    :attr neighbour_matrix: The neighbour matrix
    :attr area_vector: The area vector
    """
    def __init__(self, meshio_obj):

        self._points = meshio_obj.points
        self.cell_factory(meshio_obj.cells)

        # normal vector tensor stores the normal vector of the cell and its neighbour
        self._normal_vector_tensor = np.zeros((len(self.cells), 3, 2), dtype=np.float64)
        self._neighbour_matrix = np.full((len(self.cells), 3), -1, dtype=np.int64)
        self._area_vector = np.zeros(len(self.cells), dtype=np.float64)

        assert isinstance(self.cells, dict), f"Expected cells to be of type dict, got {self.cells.__class__} instead"
        assert isinstance(self.points, np.ndarray), f"Expected points to be of type np.ndarray, got {self.points.__class__} instead"
        assert isinstance(self.dot_matrix, np.ndarray), f"Expected dot_matrix to be of type np.ndarray, got {self.dot_matrix.__class__} instead"
        assert isinstance(self.neighbour_matrix, np.ndarray), f"Expected neighbour_matrix to be of type np.ndarray, got {self.neighbour_matrix.__class__} instead"
        assert isinstance(self.area_vector, np.ndarray), f"Expected area_vector to be of type np.ndarray, got {self.area_constant_vector.__class__} instead"


    @abstractmethod
    def cell_factory(self, cells):
        """Creates the cells from the mesh file"""
        self._cells = {}
        id = 0
        for cellblock in cells:
            for cell in cellblock.data:
                self.cells[id] = Cell(*cell, points=self._points, id=id)
                id += 1

    def find_neighbours(self):
        """
        Finds the neighbours of the cells
        returns: self
        """
        indexes = list(self.cells.keys())

        for i, cell_id in enumerate(indexes):

            cell = self.cells[cell_id]

            for neighbour_id in indexes[i:]:

                # stop the search if a cell has 3 neighbours
                if len(cell.neighbour_data) == 3: break
                neighbour = self.cells[neighbour_id]
                cell.check_if_neighour(neighbour)

        return self

    def find_constants(self):
        """
        Finds the dot product between the neighbours
        returns: self
        """
        for cell_id in self.cells.keys():

            cell = self.cells[cell_id]

            # if the cell is not a triangle, we do not want to calculate the dot product
            if cell.type != "triangle": continue
            
            for neighbour_id in cell.neighbour_data.keys():
                # if the cell is not a triangle, we cannot calculate the dot product

                neighbour = self.cells[neighbour_id]
                cell.update_edge_normal(neighbour)

        # vectorise the constants
        self._vectorise_constants()

        return self
    
    def _vectorise_constants(self):
        """
        Vectorises the constants obtained from the find_constants method
        """
        for cell_index in self.cells.keys():
            cell = self.cells[cell_index]

            normal_matrix, neighbour_row, area = vectorize_neighbour_data(cell)

            self._normal_vector_tensor[cell_index] = normal_matrix
            self._neighbour_matrix[cell_index] = neighbour_row
            self._area_vector[cell_index] = area

    @property
    def cells(self):
        return self._cells
    
    @property
    def points(self):
        return self._points
                    
    @property
    def dot_matrix(self):
        return self._dot_matrix
    
    @property
    def neighbour_matrix(self):
        return self._neighbour_matrix
    
    @property
    def area_vector(self):
        return self._area_vector
    
class ComputationalMesh:
    """
    Object that stores the vectorised mesh data
    :param file: The file path to the msh file
    :attr normal_vector_tensor: The normal vector tensor
    :attr neighbour_matrix: The neighbour matrix
    :attr area_vector: Stores area for each cell
    :attr density_vector: Stores the density for each cell
    :attr pressure_vector: Stores the pressure for each cell
    :attr velocity_matrix: stores the velocity vector for each cell
    """
    def __init__(self, msh):

        mesh = Mesh(msh)
        mesh.find_neighbours().find_constants()

        self._normal_vector_tensor = mesh._normal_vector_tensor
        self._neighbour_matrix = mesh._neighbour_matrix
        self._area_vector = mesh._area_vector
        self._density_vector = np.zeros(len(mesh.cells), dtype=np.float64)
        self._pressure_vector = np.zeros(len(mesh.cells), dtype=np.float64)
        self._velocity_matrix = np.zeros((len(mesh.cells), 2), dtype=np.float64)

        # inherent to the topology of the mesh
        assert isinstance(self.normal_vector_tensor, np.ndarray), f"Expected normal_vector_tensor to be of type np.ndarray, got {self.normal_vector_tensor.__class__} instead"
        assert isinstance(self.neighbour_matrix, np.ndarray), f"Expected neighbour_matrix to be of type np.ndarray, got {self.neighbour_matrix.__class__} instead"
        assert isinstance(self.area_vector, np.ndarray), f"Expected area_vector to be of type np.ndarray, got {self.area_constant_vector.__class__} instead"
        
        # properties that will be updated
        assert isinstance(self.density_vector, np.ndarray), f"Expected density_vector to be of type np.ndarray, got {self.density_vector.__class__} instead"
        assert isinstance(self.pressure_vector, np.ndarray), f"Expected pressure_vector to be of type np.ndarray, got {self.pressure_vector.__class__} instead"
        assert isinstance(self.velocity_matrix, np.ndarray), f"Expected velocity_matrix to be of type np.ndarray, got {self.velocity_matrix.__class__} instead"

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

# Read mesh is stored here to avoid circular imports

def read_mesh(file="data/bay.msh"):
    """
    Reads the mesh file
    :param file: The file path
    :return: The mesh object
    """
    msh = meshio.read(file)
    return Mesh(msh)