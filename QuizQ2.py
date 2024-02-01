import numpy as np
import sympy as sp

# Initialize the starting point
x_initial = np.array([1, 1, 1])
u_0 = np.array([1, 1, -4])
u_1 = np.array([2,2,1])

# Set the maximum number of iterations and tolerance
max_iterations = 5
tolerance = 1e-10

# Define the objective function f(x) = 0.5 * x^T A x - b^T x
def My_func(x, y, z):
    return 7 * x**2 + 4 * x * y + 6 * y**2 + 4 * y * z + 5 * z**2

# Define the gradient and Hessian of the objective function
def get_Grad_Hessian(x, y, z):
    x_sym, y_sym, z_sym = sp.symbols('x y z')
    f = My_func(x_sym, y_sym, z_sym)
    gradient = [sp.diff(f, x_sym), sp.diff(f, y_sym), sp.diff(f, z_sym)]
    grad = np.array([float(gradient[0].subs({x_sym: x, y_sym: y, z_sym: z})),
                     float(gradient[1].subs({x_sym: x, y_sym: y, z_sym: z})),
                     float(gradient[2].subs({x_sym: x, y_sym: y, z_sym: z}))])
    
    hessian = np.array([[float(sp.diff(gradient[0], x_sym).subs({x_sym: x, y_sym: y, z_sym: z})),
                         float(sp.diff(gradient[0], y_sym).subs({x_sym: x, y_sym: y, z_sym: z})),
                         float(sp.diff(gradient[0], z_sym).subs({x_sym: x, y_sym: y, z_sym: z}))],
                        
                        [float(sp.diff(gradient[1], x_sym).subs({x_sym: x, y_sym: y, z_sym: z})),
                         float(sp.diff(gradient[1], y_sym).subs({x_sym: x, y_sym: y, z_sym: z})),
                         float(sp.diff(gradient[1], z_sym).subs({x_sym: x, y_sym: y, z_sym: z}))],
                        
                        [float(sp.diff(gradient[2], x_sym).subs({x_sym: x, y_sym: y, z_sym: z})),
                         float(sp.diff(gradient[2], y_sym).subs({x_sym: x, y_sym: y, z_sym: z})),
                         float(sp.diff(gradient[2], z_sym).subs({x_sym: x, y_sym: y, z_sym: z}))]])
    
    return grad, hessian

# Define the Conjugate Gradient Descent algorithm
def conjugate_gradient_descent(x0, tol):
    gradient, A = get_Grad_Hessian(x0[0], x0[1], x0[2])
    x = x0
    r = -gradient  # Initial residual is -gradient
    p = r
    step_count = 0
    while np.linalg.norm(r) >= tol:
        Ap = np.dot(A, p)
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_next = r - alpha * Ap
        beta = np.dot(r_next, r_next) / np.dot(r, r)
        p = r_next + beta * p
        r = r_next
        step_count += 1
    return x, step_count

# Run the Conjugate Gradient Descent algorithm
result, steps_taken = conjugate_gradient_descent(x_initial, tolerance)

# Print the result and step count
print("Optimal Point (x1, x2, x3):", result)
print("Minimum Value of f(x):", My_func(*result))
print("Number of Steps Taken:", steps_taken)
