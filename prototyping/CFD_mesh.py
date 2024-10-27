import numpy as np
import CFD_functions as cfd
import visualise as vis
import scipy.sparse as sps
import scipy.sparse.linalg as spla


class Mesh:
    def __init__(self, file, fluid='air'):
        self._mesh = cfd.load_mesh(file)
        self.points = self._mesh.points
        self.num_cells = len(self._mesh.cells_dict["triangle"])

        # Define inlet, outlet, and wall nodes
        self._inlet_nodes, self._outlet_nodes, self._wall_nodes = cfd.boundary_nodes(self._mesh)

        # Set up matrices
        self._flux_matrix = sps.lil_matrix((self.num_cells, self.num_cells))
        self._inlet_matrix = sps.lil_matrix((self.num_cells, self.num_cells))
        self._outlet_matrix = sps.lil_matrix((self.num_cells, self.num_cells))
        self._wall_matrix = sps.lil_matrix((self.num_cells, self.num_cells))

        # Set fluid properties like density and viscosity
        self.set_fluid_properties(fluid)

        # Apply boundary conditions and fill the flux matrix
        self.apply_boundary_conditions()
        self.fill_flux_matrix()

    def get_cell_centroids(self):
        """Return the centroids of the mesh cells."""
        return np.array([np.mean([self.points[i] for i in cell], axis=0) for cell in self._mesh.cells_dict["triangle"]])

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
        """Fill the flux matrix based on neighboring cells and diffusion."""
        neighbour_matrix = cfd.detect_neighbours(self._mesh)

        # Loop over each cell and compute flux with neighbors
        for i in range(self.num_cells):
            if i not in self._inlet_nodes and i not in self._outlet_nodes and i not in self._wall_nodes:
                num_valid_neighbors = np.count_nonzero(neighbour_matrix[i] != -1)
                self._flux_matrix[i, i] = -num_valid_neighbors * self.viscosity

                for neighbor in neighbour_matrix[i]:
                    if neighbor != -1:
                        # Compute flux between neighbors based on viscosity
                        self._flux_matrix[i, neighbor] = self.viscosity

    def solve(self):
        """Solve the system of equations for pressure and velocity."""
        system_matrix = self.system_matrix
        velocity_x, velocity_y, pressure = self.set_up_starting_values()

        # For now, solve for pressure alone (later extend to velocity-pressure coupling)
        rhs = pressure

        # Solve the system (Ax = b)
        solution = spla.spsolve(system_matrix, rhs)

        return solution

    @property
    def system_matrix(self):
        """Return the combined system matrix for velocity and pressure."""
        return self._flux_matrix + self._inlet_matrix + self._outlet_matrix + self._wall_matrix


if __name__ == '__main__':
    # Solve for air
    mesh_air = Mesh('../data/msh/pipe_mesh_high_res.msh', fluid='air')
    pressure_air = mesh_air.solve()

    # Visualize the pressure solution
    vis.plot_solution(mesh_air.get_cell_centroids(), pressure_air)
