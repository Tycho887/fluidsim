import meshio
import numpy as np
import time
import numpy as np
from collections import defaultdict

class Triangle:
    def __init__(self, *args, points: dict, id: int):
        self._id = id
        self._points = {i: points[i][:2] for i in args}  # Use first 2 coordinates only
        self.neighbours = []

    @property
    def id(self):
        return self._id
    
    @property
    def points(self):
        return self._points

    def add_neighbour(self, other: "Triangle"):
        """Adds the given triangle as a neighbour if not already added."""
        if other not in self.neighbours:
            self.neighbours.append(other)
            other.neighbours.append(self)

class FindNeighbours:
    def __init__(self, mesh):
        self.points = mesh.points
        self.triangles = [Triangle(*cell, points=self.points, id=i) for i, cell in enumerate(mesh.cells_dict["triangle"])]
        self._find_neighbours()

    def _get_edges(self, triangle: Triangle):
        """Returns the edges (as sets of point indices) of a triangle."""
        points = list(triangle.points.keys())
        return {(points[i], points[(i + 1) % 3]) for i in range(3)}  # Cyclic edges

    def _find_neighbours(self):
        """Efficiently finds and sets neighbours for triangles using edge lookups."""
        edge_to_triangles = defaultdict(list)

        # Step 1: Create edge-to-triangle mapping
        for triangle in self.triangles:
            edges = self._get_edges(triangle)
            for edge in edges:
                sorted_edge = tuple(sorted(edge))  # Sort points in edge for consistent comparison
                edge_to_triangles[sorted_edge].append(triangle)

        # Step 2: Find neighbours by checking shared edges
        for triangles in edge_to_triangles.values():
            if len(triangles) > 1:
                for i in range(len(triangles)):
                    for j in range(i + 1, len(triangles)):
                        triangles[i].add_neighbour(triangles[j])

    def get_neighbour_matrix(self):
        """Returns a matrix where each row corresponds to a triangle and contains its neighbour IDs."""
        self._find_neighbours()

        neighbour_matrix = []
        for triangle in self.triangles:
            neighbours_ids = [neighbour.id for neighbour in triangle.neighbours]
            # Fill up to 3 neighbours with -1 if there are fewer
            neighbours_ids += [-1] * (3 - len(neighbours_ids))
            neighbour_matrix.append(neighbours_ids)

        return np.array(neighbour_matrix)

if __name__ == "__main__":
    mesh = meshio.read("../data/msh/simple.msh")
    start = time.time()
    print(FindNeighbours(mesh).get_neighbour_matrix())    
    end = time.time()
    print(f"Time taken: {end - start}")