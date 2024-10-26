from precompute import Topology
import numpy as np
import meshio

def initialize_conditions(self, init_velocity=np.array([1.0, 0.0]), init_pressure=1.0):
    """
    Initializes the velocity and pressure for the entire domain.
    """
    self._velocity_matrix[:, :] = init_velocity  # Initialize velocity in x-direction
    self._pressure_vector[:] = init_pressure  # Set uniform pressure initially


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

    def initialize_conditions(self, init_velocity=np.array([1.0, 0.0]), init_pressure=1.0):
        """
        Initializes the velocity and pressure for the entire domain.
        """
        self._velocity_matrix[:, :] = init_velocity  # Initialize velocity in x-direction
        self._pressure_vector[:] = init_pressure  # Set uniform pressure initially

    def apply_boundary_conditions(self, boundary_type="no-slip", inlet_velocity=None):
        """
        Applies boundary conditions. For simplicity, assuming certain boundary cells.
        """
        for i, neighbours in enumerate(self._neighbour_matrix):
            if np.any(neighbours == -1):  # Identify boundary cells
                if boundary_type == "no-slip":
                    self._velocity_matrix[i] = np.array([0.0, 0.0])  # No-slip condition
                elif boundary_type == "inlet" and inlet_velocity is not None:
                    self._velocity_matrix[i] = inlet_velocity  # Apply inlet velocity

    def update_velocity(self, dt):
        """
        Updates velocity using a basic forward Euler method, considering pressure gradients and viscosity.
        """
        new_velocity = np.copy(self._velocity_matrix)
        for i in range(len(self._velocity_matrix)):
            velocity = self._velocity_matrix[i]
            pressure = self._pressure_vector[i]
            area = self._area_vector[i]
            flux_sum = np.zeros(2)

            # Iterate over neighbours
            for j, neighbour_id in enumerate(self._neighbour_matrix[i]):
                if neighbour_id == -1:
                    continue  # Skip if no neighbour
                
                neighbour_velocity = self._velocity_matrix[neighbour_id]
                normal = self._normal_vector_tensor[i, j]
                pressure_diff = pressure - self._pressure_vector[neighbour_id]

                # Pressure force
                pressure_force = pressure_diff * normal
                
                # Viscous term
                viscous_flux = self._viscosity * (neighbour_velocity - velocity)

                # Sum up the fluxes
                flux_sum += pressure_force + viscous_flux
            
            # Update velocity based on fluxes
            new_velocity[i] += dt * flux_sum / area
        
        # Apply velocity updates
        self._velocity_matrix = new_velocity

    def update_pressure(self, dt):
        """
        Updates pressure based on the continuity equation or a Poisson pressure correction.
        """
        new_pressure = np.copy(self._pressure_vector)
        for i in range(len(self._pressure_vector)):
            velocity = self._velocity_matrix[i]
            divergence = 0.0
            
            # Calculate divergence using neighbours
            for j, neighbour_id in enumerate(self._neighbour_matrix[i]):
                if neighbour_id == -1:
                    continue  # Skip if no neighbour
                
                neighbour_velocity = self._velocity_matrix[neighbour_id]
                normal = self._normal_vector_tensor[i, j]
                
                # Compute divergence
                divergence += np.dot(neighbour_velocity - velocity, normal)
            
            # Update pressure based on divergence (simplified)
            new_pressure[i] -= dt * divergence / self._area_vector[i]
        
        # Apply pressure updates
        self._pressure_vector = new_pressure

    def simulate(self, time_steps, dt, boundary_type="no-slip", inlet_velocity=None):
        """
        Runs the simulation for a given number of time steps.
        """
        for t in range(time_steps):
            print(f"Time step {t+1}/{time_steps}")
            
            # Apply boundary conditions at each step
            self.apply_boundary_conditions(boundary_type, inlet_velocity)

            # Update velocity field
            self.update_velocity(dt)

            # Update pressure field
            self.update_pressure(dt)

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
    def normal_vector_tensor(self):
        return self._normal_vector_tensor
    
    @property
    def neighbour_matrix(self):
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
    def velocity_matrix(self):
        return self._velocity_matrix
    
    @property
    def viscosity(self):
        return self._viscosity    
    
    # Other methods omitted for brevity...
