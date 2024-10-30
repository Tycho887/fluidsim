import numpy as np
import CFD_functions as cfd
import visualise as vis
import Topology as topo
import scipy.sparse as sps
import scipy.sparse.linalg as spla

class SIMPLE:
    def __init__(self,file, fluid='air', autosetup=True):
        self._mesh = topo.UnstructuredMesh(file)
        self.face_normals = self._mesh.face_normals
        self.areas = self._mesh.surface_area_vector
        self.centroids = self._mesh.centroids
        self.adjenency_matrix = self._mesh.adjacency_matrix
        self.num_cells = len(self._mesh.elements)

        # Define inlet, outlet, and wall nodes
        self._inlet_nodes, self._outlet_nodes, self._wall_nodes = cfd.boundary_nodes(self._mesh.meshio_obj)

        # Set up matrices
        self._flux_matrix = sps.lil_matrix((self.num_cells, self.num_cells))
        self._inlet_matrix = sps.lil_matrix((self.num_cells, self.num_cells))
        self._outlet_matrix = sps.lil_matrix((self.num_cells, self.num_cells))
        self._wall_matrix = sps.lil_matrix((self.num_cells, self.num_cells))

        if autosetup:
            # Set fluid properties like density and viscosity
            self.set_fluid_properties(fluid)

            # Apply boundary conditions and fill the flux matrix
            self.apply_boundary_conditions()
            self.fill_flux_matrix()

    def get_adjacents(self, cell_id):
        """Return the neighboring cells for the given cell ID, handling boundary cells properly."""
        # Get the list of neighboring cells from the adjacency matrix
        neighbors = self.adjenency_matrix[cell_id]

        # Filter out invalid neighbors (e.g., -1 or any other marker for non-existent neighbors)
        valid_neighbors = [neighbor for neighbor in neighbors if neighbor != -1]

        return valid_neighbors

    
    def get_face_normal(self, cell_id, neighbor_id):
        """Return the normal vector for the face between two cells, handling boundary conditions."""
        # Check if the neighbor is valid (non-boundary)
        if neighbor_id == -1:  # Boundary face case
            # For boundary cells, return a predefined normal (e.g., for a wall or inlet)
            # Here you could determine the correct normal based on the boundary type
            if cell_id in self._wall_nodes:
                return self.compute_wall_normal(cell_id)  # Custom method for wall normals
            elif cell_id in self._inlet_nodes:
                return np.array([1.0, 0.0])  # Example: Inlet normal (assume it's in x-direction)
            elif cell_id in self._outlet_nodes:
                return np.array([-1.0, 0.0])  # Example: Outlet normal (assume it's in -x direction)
            else:
                raise ValueError("Unhandled boundary condition")
        
        # Otherwise, compute the normal vector between the two neighboring cells
        adjecent_cells = self.get_adjacents(cell_id)
        neighbor_index = np.where(adjecent_cells == neighbor_id)[0][0]
        return self.face_normals[cell_id, neighbor_index]

    
    def set_fluid_properties(self, fluid):
        """Set realistic values for fluid properties based on the fluid type."""
        if fluid == 'air':
            self.density = 1.225  # kg/m³ (air at sea level)
            self.viscosity = 1.81e-5  # Pa.s (dynamic viscosity of air)
        elif fluid == 'water':
            self.density = 998.0  # kg/m³ (water at room temperature)
            self.viscosity = 1.0e-3  # Pa.s (dynamic viscosity of water)
        else:
            raise ValueError(f"Unknown fluid type: {fluid}")

    def set_up_starting_values(self):
        """Initialize velocity and pressure fields and apply boundary conditions."""
        velocity_x = np.zeros(self.num_cells)
        velocity_y = np.zeros(self.num_cells)
        pressure = np.zeros(self.num_cells)

        u_inlet = 1.0  # Example inlet velocity (m/s)

        # Apply inlet velocity boundary condition for velocity
        for node in self._inlet_nodes:
            velocity_x[node] = u_inlet

        # Set outlet pressure to 101 kPa (Pa)
        for node in self._outlet_nodes:
            pressure[node] = 101000  # Pa (101 kPa)

        # Apply no-slip boundary condition (zero velocity) on walls
        for node in self._wall_nodes:
            velocity_x[node] = 0.0
            velocity_y[node] = 0.0

        return velocity_x, velocity_y, pressure

    def apply_boundary_conditions(self):
        """Apply boundary conditions to the system matrices."""
        for node in self._inlet_nodes:
            self._inlet_matrix[node, node] = 1.0  # For velocity inlet

        for node in self._outlet_nodes:
            self._outlet_matrix[node, node] = 1.0  # For pressure outlet

        for node in self._wall_nodes:
            self._wall_matrix[node, node] = 1.0  # No-slip boundary on walls

    def fill_flux_matrix(self):
        """Fill the flux matrix based on neighboring cells and diffusion, handling boundaries properly."""
        for i in range(self.num_cells):
            if i in self._inlet_nodes or i in self._outlet_nodes or i in self._wall_nodes:
                # Skip boundary cells: Boundary conditions are applied separately
                continue

            # Get valid neighbors (excluding boundaries)
            neighbors = self.get_adjacents(i)
            num_valid_neighbors = len(neighbors)

            # Set diagonal element for cell i (number of valid neighbors, including diffusion terms)
            self._flux_matrix[i, i] = -num_valid_neighbors * self.viscosity

            # Loop over valid neighbors and compute fluxes
            for neighbor in neighbors:
                # Compute flux between cells i and its neighbor
                self._flux_matrix[i, neighbor] = self.viscosity


    def solve(self, max_iterations=100, tolerance=1e-5):
        # Initialize velocity and pressure
        velocity_x, velocity_y, pressure = self.set_up_starting_values()
        
        # Iterative SIMPLE loop
        for iteration in range(max_iterations):
            #
            print(f"Iteration {iteration + 1}...")
            # Step 1: Solve momentum equations for u*, v* (intermediate velocity)
            A_u, b_u = self.momentum_equation_x(pressure, velocity_x)
            velocity_x_new, _ = spla.gmres(A_u, b_u)

            A_v, b_v = self.momentum_equation_y(pressure, velocity_y)
            velocity_y_new, _ = spla.gmres(A_v, b_v)
            
            # Step 2: Solve pressure correction equation
            A_p, b_p = self.pressure_correction_equation(velocity_x_new, velocity_y_new)
            pressure_correction, _ = spla.gmres(A_p, b_p)

            # Step 3: Correct pressure and velocity
            pressure += pressure_correction
            velocity_x_new, velocity_y_new = self.correct_velocity(velocity_x_new, velocity_y_new, pressure_correction)
            
            # Check for convergence (residuals)
            if np.linalg.norm(velocity_x_new - velocity_x) < tolerance and np.linalg.norm(velocity_y_new - velocity_y) < tolerance:
                print(f"Converged in {iteration} iterations.")
                break

            # Update velocity fields for next iteration
            velocity_x, velocity_y = velocity_x_new, velocity_y_new
            
        return velocity_x, velocity_y, pressure
    
    def momentum_equation_x(self, pressure, velocity_x):
        # LHS for the x-momentum equation
        A_u = self._flux_matrix.copy()  # Use your existing flux matrix setup
        
        # RHS for the x-momentum equation (pressure gradient and body forces)
        b_u = -self.pressure_gradient(pressure)  # Compute pressure gradient in x
        b_u += self.body_forces()[0]  # Add x-component of the body forces
        
        return A_u, b_u

    def momentum_equation_y(self, pressure, velocity_y):
        A_v = self._flux_matrix.copy()
        b_v = -self.pressure_gradient(pressure, direction='y')
        b_v += self.body_forces()[1]  # Add y-component of the body forces
        return A_v, b_v

    def pressure_correction_equation(self, velocity_x_new, velocity_y_new):
        # LHS for pressure correction equation (Poisson equation)
        A_p = self.pressure_poisson_matrix()  # Setup based on flux and element volumes
        
        # RHS: Divergence of the velocity field
        b_p = self.velocity_divergence(velocity_x_new, velocity_y_new)
        
        return A_p, b_p

    def correct_velocity(self, velocity_x, velocity_y, pressure_correction):
        velocity_x_corrected = velocity_x - self.pressure_gradient(pressure_correction, direction='x')
        velocity_y_corrected = velocity_y - self.pressure_gradient(pressure_correction, direction='y')
        
        return velocity_x_corrected, velocity_y_corrected

    def pressure_gradient(self, pressure, direction='x'):
        """
        Compute the pressure gradient for each cell in the given direction.
        
        :param pressure: Array of pressure values for each cell.
        :param direction: 'x' or 'y' indicating the direction of the gradient.
        :return: Array of pressure gradients in the specified direction for each cell.
        """
        grad_p = np.zeros(self.num_cells)  # Initialize the pressure gradient array

        # Loop over each cell
        for i in range(self.num_cells):
            #
            print(f"Computing pressure gradient for cell {i}...")
            # Get the neighboring cells
            adjecents = self.get_adjacents(i)  # Returns a list of neighboring cell indices
            cell_center = self.centroids[i]
            
            # Initialize the gradient contribution for the current cell
            gradient_contrib = 0.0

            # Loop over neighbors
            for j in adjecents:
                adjacent_center = self.centroids[j]
                
                # Compute the distance between the centroids of cell i and neighbor j
                d_ij = np.linalg.norm(adjacent_center - cell_center)
                
                # Pressure difference between cell i and neighbor j
                delta_p = pressure[j] - pressure[i]

                # Compute the normal vector for the face between cell i and neighbor j
                face_normal = self.get_face_normal(i, j)  # Returns the normal vector between the cells

                # Select the x or y component of the normal vector based on the direction
                if direction == 'x':
                    normal_component = face_normal[0]  # x-component of the normal vector
                elif direction == 'y':
                    normal_component = face_normal[1]  # y-component of the normal vector
                else:
                    raise ValueError("Unknown direction: must be 'x' or 'y'")
                
                # Accumulate the contribution from this neighbor to the pressure gradient
                gradient_contrib += (delta_p / d_ij) * normal_component

            # Store the computed gradient contribution for cell i
            grad_p[i] = gradient_contrib
        
        return grad_p


    def pressure_poisson_matrix(self):
        """Return the Poisson matrix for the pressure correction equation."""
        A_p = sps.lil_matrix((self.num_cells, self.num_cells))

        for i in range(self.num_cells):
            # Boundary cells should have specific treatment
            if i in self._inlet_nodes or i in self._wall_nodes:
                A_p[i, i] = 1.0  # Fix the pressure correction at boundaries
                continue
            elif i in self._outlet_nodes:
                A_p[i, i] = 1.0  # Outlet: Enforce specific pressure condition
                continue

            # Internal cells: Calculate based on valid neighbors
            neighbors = self.get_adjacents(i)
            A_p[i, i] = len(neighbors)  # Diagonal element

            for j in neighbors:
                if j != -1:
                    A_p[i, j] = -1  # Off-diagonal: neighbor 

        return A_p


    def velocity_divergence(self, velocity_x, velocity_y):
        """Compute the divergence of the velocity field."""
        divergence = np.zeros(self.num_cells)

        # Loop over each cell
        for i in range(self.num_cells):
            adjacents = self.get_adjacents(i)
            cell_center = self.centroids[i]

            # Sum contributions from neighboring cells
            for j in adjacents:
                adjacent_center = self.centroids[j]

                # Compute distance between the centroids of cell i and neighbor j
                d_ij = np.linalg.norm(adjacent_center - cell_center)

                # Get face normal vector
                face_normal = self.get_face_normal(i, j)

                # Velocity difference between cell i and neighbor j
                delta_u_x = velocity_x[j] - velocity_x[i]
                delta_u_y = velocity_y[j] - velocity_y[i]

                # Compute divergence (dot product of velocity difference and face normal)
                divergence[i] += (delta_u_x * face_normal[0] + delta_u_y * face_normal[1]) / d_ij

        return divergence


        
    def body_forces(self):
        """Return the body force vector for each cell, assuming gravity is the only force."""
        g = np.array([0.0, -9.81])  # Gravity vector in the negative y-direction
        forces = self.density * g    # Multiply by density to get the force per unit volume
        return forces  # Return the same force for each cell


    @property
    def system_matrix(self):
        """Return the combined system matrix for velocity and pressure."""
        return self._flux_matrix + self._inlet_matrix + self._outlet_matrix + self._wall_matrix


if __name__ == "__main__":
    cfd = SIMPLE("../data/msh/pipe_mesh.msh")
    velocity_x, velocity_y, pressure = cfd.solve()
    vis.plot_solution(cfd.centroids, velocity_y)

    