import numpy as np
from abc import ABC, abstractmethod
import meshio

types = {1: "vertex", 2: "line", 3: "triangle"}

class Cell(ABC):
    def __init__(self, *args, points: dict, id: int):
        self._id = id
        self._points = {i: points[i][:2] for i in args}  # store only x, y coordinates
        self._type = types[len(self._points)]
        self._centroid = np.mean(list(self._points.values()), axis=0)
        self._area = self._compute_area()
        self._neighbours = {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self._id})"

    def check_if_neighbour(self, other_cell: "Cell") -> bool:
        common_points = set(self._points).intersection(set(other_cell._points))
        if len(common_points) == 2:
            self._neighbours[other_cell._id] = {"common_points": list(common_points), 'normal': np.zeros(2)}
            other_cell._neighbours[self._id] = {"common_points": list(common_points), 'normal': np.zeros(2)}
            return True
        return False

    def update_edge_normal(self, neighbour: "Cell"):
        common_pts = [self._points[i] for i in self._neighbours[neighbour._id]["common_points"]]
        edge_vector = np.array(common_pts[1]) - np.array(common_pts[0])
        normal = np.array([[0, -1], [1, 0]]) @ edge_vector  # rotate vector to find normal
        normal = normal / np.linalg.norm(normal)
        if np.dot(normal, np.array(common_pts[0]) - self._centroid) < 0:
            normal = -normal
        self._neighbours[neighbour._id]['normal'] = normal

    def _compute_area(self) -> float:
        if len(self._points) != 3:
            return 0
        pts = list(self._points.values())
        x1, y1, x2, y2, x3, y3 = pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1]
        return 0.5 * abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

    @property
    def area(self):
        return self._area

    @property
    def id(self):
        return self._id

    @property
    def points(self):
        return self._points

    @property
    def centroid(self):
        return self._centroid

    @property
    def neighbours(self):
        return self._neighbours

    @property
    def type(self):
        return self._type

class Topology:
    """
    Array of object structure to store the mesh topology
    """
    def __init__(self, meshio_obj):
        # Store only the x and y coordinates of the points
        self._points = {i: meshio_obj.points[i][:2] for i in range(len(meshio_obj.points))}
        self._create_cells(meshio_obj.cells)
        self._normal_vectors = np.zeros((len(self._cells), 3, 2), dtype=np.float64)
        self._neighbour_matrix = np.full((len(self._cells), 3), -1, dtype=np.int64)
        self._area_vector = np.zeros(len(self._cells), dtype=np.float64)
        self._centroids = np.array([cell.centroid for cell in self._cells.values()])

    @abstractmethod
    def _create_cells(self, cells):
        """Creates the cells from the mesh file"""
        self._cells = {}
        id = 0
        for cellblock in cells:
            for cell in cellblock.data:
                self.cells[id] = Cell(*cell, points=self._points, id=id)
                id += 1

    def find_neighbours_and_compute_normals(self):
        for cell_id, cell in self._cells.items():
            print(cell_id)
            for other_cell_id, other_cell in list(self._cells.items())[cell_id:]:
                if len(cell.neighbours) == 3:
                    break
                if cell.check_if_neighbour(other_cell):
                    cell.update_edge_normal(other_cell)
        self._vectorize_constants()

    def _vectorize_constants(self):
        for cell_id, cell in self._cells.items():
            normals, neighbour_ids, area = self._vectorize_neighbour_data(cell)
            self._normal_vectors[cell_id] = normals
            self._neighbour_matrix[cell_id] = neighbour_ids
            self._area_vector[cell_id] = area

    @staticmethod
    def _vectorize_neighbour_data(cell):
        normal_matrix = np.zeros((3, 2), dtype=np.float64)
        neighbour_ids = np.full(3, -1, dtype=np.int64)
        if cell.type == "triangle":
            for i, (neighbour_id, data) in enumerate(cell.neighbours.items()):
                normal_matrix[i] = data['normal']
                neighbour_ids[i] = neighbour_id
        return normal_matrix, neighbour_ids, cell.area

    @property
    def cells(self):
        return self._cells
    
    def read_mesh(self,file):
        """
        Reads the mesh file
        :param file: The file path
        :return: The mesh object
        """
        msh = meshio.read(file)
        return msh
