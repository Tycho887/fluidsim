import numpy as np
import src.utils.decorators as decos


@decos.logging_decorator
def object_solver(mesh, cell_oil_content_vector, dt, area_constant):
    """
    Solve the system of equations using object oriented programming
    :param mesh: The mesh object
    :param cell_oil_content_vector: The oil content vector
    :param dt: The time step
    :param area_constant: The area constant
    :return: The updated oil content vector
    """

    flux_vector = np.zeros(len(cell_oil_content_vector))

    for cell_id in cell_oil_content_vector.index:

        cell = mesh.cells[cell_id]
    
        # calculate the flux vector
        for neighbour_id in mesh.cells[cell_id].neighbour_data.keys():

            oil_content_main = cell_oil_content_vector[cell_id]
            oil_content_neighbour = cell_oil_content_vector[neighbour_id]
            dot = cell.neighbour_data[neighbour_id]['dot']

            if dot > 0:
                flux_vector[cell_id] += oil_content_main * dot
            else:
                flux_vector[cell_id] += oil_content_neighbour * dot

    return cell_oil_content_vector - dt * area_constant * flux_vector