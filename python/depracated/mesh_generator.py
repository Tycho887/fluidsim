import gmsh
import numpy as np

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("2D_pipe")

# Define parameters for the rectangular pipe
length = 10.0  # Length of the pipe
height = 2.0   # Height of the pipe
mesh_size = 0.02  # Approximate size of mesh elements

# Create points for the rectangle
p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
p2 = gmsh.model.geo.addPoint(length, 0, 0, mesh_size)
p3 = gmsh.model.geo.addPoint(length, height, 0, mesh_size)
p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

# Create lines (edges of the rectangle) and define them as inlet, outlet, and walls
bottom_wall = gmsh.model.geo.addLine(p1, p2)      # Bottom wall
outlet = gmsh.model.geo.addLine(p2, p3)           # Right side (outlet)
top_wall = gmsh.model.geo.addLine(p3, p4)         # Top wall
inlet = gmsh.model.geo.addLine(p4, p1)            # Left side (inlet)

# Create a closed loop and surface
closed_loop = gmsh.model.geo.addCurveLoop([bottom_wall, outlet, top_wall, inlet])
surface = gmsh.model.geo.addPlaneSurface([closed_loop])

# Define physical groups for boundary conditions
gmsh.model.geo.addPhysicalGroup(1, [inlet], tag=1)    # Tag 1 for inlet
gmsh.model.geo.addPhysicalGroup(1, [outlet], tag=2)   # Tag 2 for outlet
gmsh.model.geo.addPhysicalGroup(1, [bottom_wall, top_wall], tag=3)  # Tag 3 for walls
gmsh.model.geo.addPhysicalGroup(2, [surface], tag=4)  # Tag 4 for the surface (fluid domain)

# Synchronize and generate the mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

# Save the mesh to file
gmsh.write("../data/msh/pipe_mesh_high_res.msh")

# Finalize Gmsh
gmsh.finalize()
