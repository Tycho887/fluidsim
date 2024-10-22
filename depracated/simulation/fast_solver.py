import numpy as np
import src.utils.decorators as decos

def vector_solver(neighbour_matrix, dot_matrix, oil_content_vector, area_constant_vector, dt):
    """
    Solve the system of equations using vectorized operations
    :param neighbour_matrix: The neighbour matrix
    :param dot_matrix: The dot matrix
    :param oil_content_vector: The oil content vector
    :param area_constant_vector: The area constant vector
    :param dt: The time step
    :return: Oil content vector for next iteration"""

    # Initialize valid neighbour oil content array
    valid_neighbour_oil_content = np.zeros(neighbour_matrix.shape, dtype=np.float64)

    for i in range(valid_neighbour_oil_content.shape[1]): # Loop over the columns of the valid_neighbour_oil_content array

        indices = neighbour_matrix[:, i]                                                            # get the indices of the neighbours
        valid_indices = indices >= 0                                                                # get the valid indices, i.e. the indices that are greater than or equal to 0
        valid_neighbour_oil_content[valid_indices, i] = oil_content_vector[indices[valid_indices]]  # Fill the valid neighbour oil content


    # Compute fluxes using vectorized operations
    flux_vector = np.sum(np.where(dot_matrix > 0,                                       # If the dot product is positive
                                  dot_matrix * oil_content_vector[:, None],             # Multiply the oil content with the dot product
                                  dot_matrix * valid_neighbour_oil_content), axis=1)    # Otherwise multiply the valid neighbour oil content with the dot product

    # [:, None] allows us to broadcast the oil content vector to the shape of the dot matrix

    return oil_content_vector - dt * area_constant_vector * flux_vector