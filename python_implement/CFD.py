import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PipeFlowSimulation:
    def __init__(self, Lx=1.0, Ly=1.0, nx=50, ny=50, rho=100.0, nu=10.0, g=10.0, dt=1e-5, nt=500):
        # Simulation parameters
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.dx, self.dy = Lx / (nx - 1), Ly / (ny - 1)
        self.rho, self.nu = rho, nu
        self.g = -g
        self.dt, self.nt = dt, nt

        assert self.nx > 2, "nx must be greater than 2"
        assert self.ny > 2, "ny must be greater than 2"

        # Check the stability condition
        
        assert self.dt <= min(self.dx, self.dy)**2 / (4 * self.nu), "Time step is too large"
        
        # Initialize fields
        self.u = np.zeros((ny, nx))   # x-velocity
        self.v = np.zeros((ny, nx))   # y-velocity
        self.p = np.zeros((ny, nx))   # Pressure
        self.b = np.zeros((ny, nx))   # RHS of the pressure-Poisson equation

        # Apply boundary conditions
        self.apply_boundary_conditions()

    def build_up_b(self):
        """
        Compute the RHS for the pressure Poisson equation.
        """
        b = self.b
        u, v = self.u, self.v
        dx, dy, dt, rho = self.dx, self.dy, self.dt, self.rho
        b[1:-1, 1:-1] = (rho * (1 / dt * 
                        ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                         (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                        ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                        2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                             (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                        ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
        return b

    def pressure_poisson(self, max_iterations=100):
        """
        Solve the pressure Poisson equation with gravity consistency.
        """
        p, b = self.p, self.b
        dx, dy, rho = self.dx, self.dy, self.rho
        pn = np.empty_like(p)
        
        for _ in range(max_iterations):
            pn[:] = p
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                            (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) -
                            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

            # Apply pressure boundary conditions with gravity consistency
            p[:, 0] = p[:, 1]   # Zero-gradient on left wall
            p[:, -1] = p[:, -2] # Zero-gradient on right wall
            p[0, :] = 0         # Constant pressure at the top
            p[-1, :] = p[-2, :] + rho * self.g * dy  # Hydrostatic balance at the bottom
        
        return p


    def update_velocities(self):
        """
        Update the velocity fields u and v based on pressure and boundary conditions.
        """
        u, v, p = self.u, self.v, self.p
        un, vn = u.copy(), v.copy()
        dx, dy, dt, rho, nu = self.dx, self.dy, self.dt, self.rho, self.nu

        # Update u velocity
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        # Update v velocity
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        self.apply_boundary_conditions()
        return u, v

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions:
        - No-slip on top and bottom walls (u = v = 0)
        - Zero-gradient on left wall
        - Right wall with specified inlet velocity for rotating flow
        - Constant pressure at top boundary
        """
        # No-slip on top and bottom
        self.u[-1, :] = -.5
        self.v[-1, :] = 0
        self.u[0, :] = .5
        self.v[0, :] = 0

        # Rotating flow on the sides

        self.u[:, -1] = 0
        self.v[:, -1] = .5
        self.u[:, 0] = 0
        self.v[:, 0] = -.5

        # Pressure boundary condition (constant at top boundary to balance gravity)
        self.p[0, :] = 0


    def run_simulation(self):
        for _ in range(self.nt):
            self.b = self.build_up_b()
            self.p = self.pressure_poisson()
            self.u, self.v = self.update_velocities()

    def animate_simulation(self):
        """
        Animate the simulation results with both velocity and pressure fields.
        """
        fig, (ax_vel, ax_pres) = plt.subplots(1, 2, figsize=(12, 6))  # Create two subplots side by side
        x, y = np.linspace(0, self.Lx, self.nx), np.linspace(0, self.Ly, self.ny)
        
        # Initial plot of the velocity magnitude
        vel_magnitude = np.sqrt(self.u**2 + self.v**2)
        CS_vel = ax_vel.contourf(x, y, vel_magnitude, alpha=0.5)
        cbar_vel = fig.colorbar(CS_vel, ax=ax_vel)
        cbar_vel.ax.set_ylabel('Velocity Magnitude')
        
        # Initial plot of the pressure field
        CS_pres = ax_pres.contourf(x, y, self.p, alpha=0.5)
        cbar_pres = fig.colorbar(CS_pres, ax=ax_pres)
        cbar_pres.ax.set_ylabel('Pressure')
        
        def animate(i):
            ax_vel.clear()
            ax_pres.clear()
            
            # Update velocity plot
            vel_magnitude = np.sqrt(self.u**2 + self.v**2)
            ax_vel.quiver(x, y, self.u, self.v, scale=10)
            CS_vel = ax_vel.contourf(x, y, vel_magnitude, alpha=0.5)
            ax_vel.set_title(f"Velocity Field (Time step: {i})")
            
            # Update pressure plot
            CS_pres = ax_pres.contourf(x, y, self.p, alpha=0.5)
            ax_pres.set_title("Pressure Field")
            
        ani = animation.FuncAnimation(fig, animate, frames=range(0, self.nt, self.nt // 100), repeat=False)
        plt.tight_layout()
        plt.show()


# Initialize and run the simulation
simulation = PipeFlowSimulation()
simulation.run_simulation()
simulation.animate_simulation()
