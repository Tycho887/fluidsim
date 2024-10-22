from abc import ABC, abstractmethod
import numpy as np

types = {1: "vertex", 2: "line", 3: "triangle"}

def centroid(points):
    """
    Compute the centroid of a cell
    :param points: The points of the cell, in the form of a dictionary
    :return: The centroid of the cell
    """
    assert isinstance(points, dict), f"Expected points to be a dictionary, got {type(points)}"
    points = np.array(list(points.values()))
    return np.mean(points, axis=0)

def area(points):
    """
    Compute the area of a cell
    :param points: The points of the cell, in the form of a dictionary
    """
    assert isinstance(points, dict), f"Expected points to be a dictionary, got {type(points)}"

    if 0 < len(points) < 3: 
        return 0
    elif len(points) == 3:

        # compute the area of a cell
        p1, p2, p3 = points.values()
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        return 0.5 * abs((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1))
    else:
        raise NotImplementedError("Only triangles are supported")
    
def find_edge_normal(cell, neighbour):
    """
    Find the dot product between the normal vector and the average velocity of the cell and its neighbour
    :param cell: The cell
    :param neighbour: The neighbour
    :return: The dot product
    """
    assert hasattr(cell, "points"), "The cell must have a points attribute"
    assert hasattr(cell, "neighbour_data"), "The cell must have a neighbour_data attribute"
    assert neighbour.id in cell.neighbour_data, "The neighbour must be a neighbour of the cell"
    assert isinstance(cell.points, dict), "The points must be a dictionary"

    # find the dot product between the normal vector and the average velocity of the cell and its neighbour

    p1, p2 = [cell.points[i] for i in cell.neighbour_data[neighbour.id]["common_points"]]

    if isinstance(p1, list): p1 = np.array(p1)
    if isinstance(p2, list): p2 = np.array(p2)

    e = p2 - p1

    assert isinstance(e, np.ndarray), "The points must be numpy arrays"

    normal = np.array([[0, -1], [1, 0]]) @ e  # matrix multiplication by the rotation matrix to get the normal vector

    scaled_normal = normal * np.linalg.norm(p2-p1) / np.linalg.norm(normal)

    # correct the direction of the normal vector

    if np.dot(p1 - cell.centroid, scaled_normal) < 0: scaled_normal = -scaled_normal

    # check if the normal vector is orthogonal to the line between the points

    assert np.isclose(np.dot(p2-p1, scaled_normal), 0), f"Normal vector is not orthogonal to the line between the points {p1} and {p2}"

    return scaled_normal

class Cell(ABC):
    """
    The base class for all cells
    :param args: The points of the cell
    :param points: The points of the mesh
    :param id: The id of the cell
    :attr fluid_flow: The fluid flow vector at the centroid
    :attr area: The area of the cell
    :attr id: The id of the cell
    :attr points: The points of the cell
    :attr centroid: The centroid of the cell
    :attr neighbour_data: The neighbour data
    """
    def __init__(self, *args, points, id):
        
        self._id = id
        self._neighbour_data = {}

        # we only want to store the x and y coordinates

        try: 
            self._points = {i: points[i][0:2] for i in args}
        except IndexError:
            self._points = {i: points[i] for i in args}

        self._type = types[len(self.points)] 
        self._centroid = centroid(self.points)
        self._area = area(self.points)

        assert len(self.points) == len(args), "The number of points must be equal to the number of arguments"
        assert isinstance(self.id, int), f"Expected id to be of type int, got {self.id.__class__} instead"
        assert isinstance(self.points, dict), f"Expected points to be of type dict, got {self.points.__class__} instead"
        assert isinstance(self.centroid, np.ndarray), f"Expected centroid to be of type np.ndarray, got {self.centroid.__class__} instead"
        assert isinstance(self.area, (float,int)), f"Expected area to be of type float, got {self.area.__class__} instead"

    def __repr__(self):
        """"""
        return f"{self.__class__.__name__}({self.id})"
    
    def check_if_neighour(self, cell):
        """
        Check if the cell is a neighbour
        :param cell: The cell object
        :return: True if the cell is a neighbour
        """
        intersection = set(self.points.keys()).intersection(set(cell.points.keys()))

        if len(intersection) == 2:
            # Save the neighbour information to the neighbour data of the cell, 
            # and the cell to the neighbour data of the neighbour
            self._neighbour_data[cell.id] = {"common_points": list(intersection), 'normal': np.array([0, 0])}
            cell._neighbour_data[self.id] = {"common_points": list(intersection), 'normal': np.array([0, 0])}
            return True
        return False

    def update_edge_normal(self, neighbour):
        """
        Updates the edge normal of the cell and its neighbour
        :param neighbour: The neighbour cell object
        :return: None
        """
        self._neighbour_data[neighbour.id]['normal'] = find_edge_normal(self, neighbour)
    
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
    def neighbour_data(self):
        return self._neighbour_data
    
    @property
    def type(self):
        return self._type
    
