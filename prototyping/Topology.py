import numpy as np
from collections import defaultdict
import meshio
import time
from abc import ABC, abstractmethod

types = {1: "vertex", 2: "line", 3: "triangle"}

class Element(ABC):
    def __init__(self, *args, vertices: dict, id: int):
        self._id = id
        self._vertices = {i: vertices[i][:2] for i in args}  # store only x, y coordinates
        self._type = types[len(self._vertices)]
        self._centroid = np.mean(list(self._vertices.values()), axis=0)
        self._surface_area = self._compute_area()
        self._adjacent_elements = {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self._id})"

    def check_if_neighbour(self, other_element: "Element") -> bool:
        common_vertices = set(self._vertices).intersection(set(other_element._vertices))
        if len(common_vertices) == 2:
            self._adjacent_elements[other_element._id] = {"common_vertices": list(common_vertices), 'normal': np.zeros(2)}
            other_element._adjacent_elements[self._id] = {"common_vertices": list(common_vertices), 'normal': np.zeros(2)}
            return True
        return False

    def update_edge_normal(self, adjacent_element: "Element"):
        common_pts = [self._vertices[i] for i in self._adjacent_elements[adjacent_element._id]["common_vertices"]]
        edge_vector = np.array(common_pts[1]) - np.array(common_pts[0])
        normal = np.array([[0, -1], [1, 0]]) @ edge_vector  # rotate vector to find normal
        normal = normal / np.linalg.norm(normal)
        if np.dot(normal, np.array(common_pts[0]) - self._centroid) < 0:
            normal = -normal
        self._adjacent_elements[adjacent_element._id]['normal'] = normal

    def _compute_area(self) -> float:
        if len(self._vertices) != 3:
            return 0
        pts = list(self._vertices.values())
        x1, y1, x2, y2, x3, y3 = pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1]
        return 0.5 * abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

    @property
    def surface_area(self):
        return self._surface_area

    @property
    def id(self):
        return self._id

    @property
    def vertices(self):
        return self._vertices

    @property
    def centroid(self):
        return self._centroid

    @property
    def adjacent_elements(self):
        return self._adjacent_elements

    @property
    def type(self):
        return self._type


class UnstructuredMesh:
    """
    Array of object structure to store the mesh topology
    """
    def __init__(self, file, autocompute=True):

        self.meshio_obj = meshio.read(file)

        # Store only the x and y coordinates of the points
        self._vertices = {i: self.meshio_obj.points[i][:2] for i in range(len(self.meshio_obj.points))}
        self._create_elements(self.meshio_obj.cells)
        self._normal_vectors = np.zeros((len(self._elements), 3, 2), dtype=np.float64)
        self._adjacency_matrix = np.full((len(self._elements), 3), -1, dtype=np.int64)
        self._surface_area_vector = np.zeros(len(self._elements), dtype=np.float64)
        self._centroids = np.array([element.centroid for element in self._elements.values()])

        if autocompute:
            self.find_neighbours_and_compute_normals()

    @abstractmethod
    def _create_elements(self, cells):
        """Creates the elements from the mesh file"""
        self._elements = {}
        id = 0
        for cellblock in cells:
            for cell in cellblock.data:
                self._elements[id] = Element(*cell, vertices=self._vertices, id=id)
                id += 1

    def find_neighbours_and_compute_normals(self):
        for element_id, element in self._elements.items():
            for other_element_id, other_element in list(self._elements.items())[element_id:]:
                if len(element.adjacent_elements) == 3:
                    break
                if element.check_if_neighbour(other_element):
                    element.update_edge_normal(other_element)
        self._vectorize_constants()

    def _vectorize_constants(self):
        for element_id, element in self._elements.items():
            normals, neighbour_ids, surface_area = self._vectorize_adjacent_data(element)
            self._normal_vectors[element_id] = normals
            self._adjacency_matrix[element_id] = neighbour_ids
            self._surface_area_vector[element_id] = surface_area

    @staticmethod
    def _vectorize_adjacent_data(element):
        normal_matrix = np.zeros((3, 2), dtype=np.float64)
        neighbour_ids = np.full(3, -1, dtype=np.int64)
        if element.type == "triangle":
            for i, (neighbour_id, data) in enumerate(element.adjacent_elements.items()):
                normal_matrix[i] = data['normal']
                neighbour_ids[i] = neighbour_id
        return normal_matrix, neighbour_ids, element.surface_area

    @property
    def elements(self):
        return self._elements
    
    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix
    
    @property
    def surface_area_vector(self):
        return self._surface_area_vector
    
    @property
    def face_normals(self):
        return self._normal_vectors
    
    @property
    def centroids(self):
        return self._centroids
    
    @property
    def data_dict(self):
        return self.meshio_obj.cell_data_dict

if __name__ == "__main__":
    start = time.time()
    UM = UnstructuredMesh("../data/msh/pipe_mesh.msh")
    print(UM.elements)
    end = time.time() - start
    print(f"Time taken: {end}")