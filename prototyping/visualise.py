import matplotlib.pyplot as plt
import numpy as np

def plot_solution(points,solution):
    # Create a scatter plot of the solution values
    plt.figure(figsize=(10, 4))
    sc = plt.scatter(points[:, 0], points[:, 1], c=solution, cmap='viridis', s=5)
    plt.colorbar(sc, label='Solution Value (e.g., velocity or pressure)')
    plt.title("CFD Simulation Results")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis("equal")
    plt.show()