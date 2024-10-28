import numpy as np
import scipy.sparse

# We want to implement the gauss seidel method to solve a linear system of equations
# The matrix A is a sparse matrix

class GaussSeidel:
    def __init__(self,A,b,iterations,verbose=False,threshold=1e-6):
        
        self.iterations = iterations
        self.n = A.shape[0]
        self.verbose = verbose
        self.threshold = threshold

        self.A = A
        self.b = b
        self.x = np.zeros(self.n)

        self.L = scipy.sparse.tril(A)
        self.U = A - self.L

    def solve(self):
        for _ in range(self.iterations):
            if self.verbose: print(self.x)
            if self.residual() < self.threshold:
                break
            self.x = scipy.sparse.linalg.spsolve(self.L,self.b - self.U @ self.x)
        return self
    
    def residual(self):
        return np.linalg.norm(self.x - self.b)
    
    def get_solution(self):
        return self.x

if __name__ == "__main__":
    A = scipy.sparse.csr_matrix(np.array([[4,1],[1,3]]))
    b = np.array([1,2])
    gauss_seidel = GaussSeidel(A,b,10,verbose=True)
    gauss_seidel.solve()
    print(gauss_seidel.get_solution())