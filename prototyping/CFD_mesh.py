import numpy as np
import meshio
import scipy.sparse as sps
import CFD_functions as cfd
import visualise as vis

class Mesh:
    def __init__(self, file):

        self._mesh = cfd.load_mesh(file)
        self.points = self._mesh.points
        self.num_cells = len(self._mesh.cells_dict["triangle"])

        self._inlet_nodes, self._outlet_nodes, self._wall_nodes = cfd.boundary_nodes(self._mesh)
        self._flux_matrix, self._inlet_matrix, self._outlet_matrix, self._wall_matrix = cfd.generate_sparse_matrices(self._mesh)

        self.apply_boundary_conditions()
        self.fill_flux_matrix()

    def set_up_starting_values(self, density=1.0, pressure=0.0):
        rhs = np.zeros(self.num_cells)

        u_inlet = 1.0
        for node in self._inlet_nodes:
            rhs[node] = density * u_inlet

        return rhs

    def apply_boundary_conditions(self):
        u_inlet = 1.0

        for node in self._inlet_nodes:
            self._inlet_matrix[node, node] = 1.0

        for node in self._outlet_nodes:
            self._outlet_matrix[node, node] = 1.0

        for node in self._wall_nodes:
            self._wall_matrix[node, node] = 1.0

    def fill_flux_matrix(self):
        for i in range(self.num_cells):
            if i not in self._inlet_nodes and i not in self._outlet_nodes and i not in self._wall_nodes:
                self._flux_matrix[i, i] = -2.0
                if i - 1 >= 0:
                    self._flux_matrix[i, i - 1] = 1.0
                if i + 1 < self.num_cells:
                    self._flux_matrix[i, i + 1] = 1.0

    def check_rank(self):
        rank = np.linalg.matrix_rank(self.system_matrix.todense())
        print(f"Rank of the system matrix: {rank}")
        return rank

    def solve(self):
        system_matrix = self.system_matrix
        start_state = self.set_up_starting_values()
        solution = sps.linalg.spsolve(system_matrix, start_state)
        return solution

    @property
    def system_matrix(self):        
        return self._flux_matrix + self._inlet_matrix + self._outlet_matrix + self._wall_matrix

    
if __name__ == '__main__':
    mesh = Mesh('../data/pipe_mesh.msh')

    # mesh.check_rank()

    solution = mesh.solve()

    print(solution)
