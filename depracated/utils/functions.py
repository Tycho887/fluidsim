import numpy as np
import pandas as pd

def centroid(points):
    """
    Compute the centroid of a cell
    :param points: The points of the cell, in the form of a dictionary
    :return: The centroid of the cell
    """
    assert isinstance(points, dict), f"Expected points to be a dictionary, got {type(points)}"
    points = np.array(list(points.values()))
    return np.mean(points, axis=0)


def area_constant(points):
    """
    Compute the area constant of a cell
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

        area = 0.5 * abs((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1))

        return 1 / area
    else:
        raise NotImplementedError("Only triangles are supported")

def flow_function(Pm):
    """
    flow function for the fluid flow
    :param Pm: The centroid of the cell
    :return: The fluid flow vector at the centroid
    """
    assert isinstance(Pm, np.ndarray), f"Expected Pm to be a numpy array, got {type(Pm)}"

    x, y = Pm
    return np.array([y - 0.2*x, -x])


def initial_distribution(centroid, x0=np.array([0.35, 0.45])):
    """
    initial distribution of oil content
    :param centroid: The centroid of the cell
    :param x0: The source of the oil
    :return: oil concentration for the cell
    """
    assert isinstance(centroid, np.ndarray), f"Expected centroid to be a numpy array, got {type(centroid)}"
    assert isinstance(x0, np.ndarray), f"Expected x0 to be a numpy array, got {type(x0)}"

    Pm = centroid
    return np.exp((np.linalg.norm(Pm-x0)**2)/-0.01)

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

def find_nearest_value_in_array(array, value):
    """
    Find the nearest value in an array
    :param array: The array
    :param value: The value
    :return: The nearest value
    """
    assert isinstance(value, (int, float)), "The value must be an integer or a float"

    # first convert the column to a numpy array
    array = np.asarray(array).astype(float)

    idx = (np.abs(array - value)).argmin()
    assert array[idx] in array, "The value is not in the array"
    return str(array[idx])

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
    assert hasattr(cell, "area_constant"), "The cell must have an area_constant attribute"
    
    dot_row = np.zeros(3)
    neighbor_index_row = np.full(3, -1, dtype=np.int64)
    area_constant = 0

    for i, neighbour_index in enumerate(cell.neighbour_data.keys()):
        if cell.type != "triangle":
            continue
        dot_row[i] = cell.neighbour_data[neighbour_index]['dot']
        neighbor_index_row[i] = neighbour_index
        area_constant = cell.area_constant

    return dot_row, neighbor_index_row, area_constant


