import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define the function f(x, y)
def function_f(x, y):
    return (1/(2*x**2-1)) + 2*y-5

# Compute the gradient of the function symbolically
def compute_gradient_symbolic(x, y):
    x_sym, y_sym = sp.symbols('x y')
    f = function_f(x_sym, y_sym)
    gradient = [sp.diff(f, x_sym), sp.diff(f, y_sym)]
    return np.array([float(gradient[0].subs({x_sym: x, y_sym: y})), float(gradient[1].subs({x_sym: x, y_sym: y}))])

# Given point for calculating gradient
x_point = 1.0
y_point = 1/2

# Compute the gradient vector at the given point
gradient_vector = compute_gradient_symbolic(x_point, y_point)

# Compute the angle of the gradient vector
# theta = np.arctan2(float(gradient_vector[1]), float(gradient_vector[0]))
# print(theta)

# Generate unit vectors in the specified angle range
num_vectors = 100
unit_vectors = []

for i in range(num_vectors):
    alpha_i = np.pi/2 + (i / num_vectors) * np.pi
    rotation_matrix = np.array([[np.cos(alpha_i), -np.sin(alpha_i)],
                                [np.sin(alpha_i), np.cos(alpha_i)]])
    unit_vector_i = np.dot(rotation_matrix, gradient_vector)
    unit_vectors.append(unit_vector_i)

# Plot the unit vectors
plt.figure(figsize=(8, 8))
origin = np.zeros(2)

for i in range(num_vectors):
    plt.quiver(origin[0], origin[1], unit_vectors[i][0], unit_vectors[i][1], angles='xy', scale_units='xy', scale=1, color='b')

# Plot the gradient vector
plt.quiver(origin[0], origin[1], gradient_vector[0], gradient_vector[1], angles='xy', scale_units='xy', scale=1, color='r', label='Gradient Vector')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
