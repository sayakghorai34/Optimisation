import numpy as np
import sympy as sp

def my_fun(x, y):
    return (x - y + 3*x**2 + 2*x*y + y**2)

initial_point = np.array([2.0, 5.0])

def calculate_gradient_and_hessian(f, x_sym, y_sym):
    # Calculate gradient
    grad = [sp.diff(f, x_sym), sp.diff(f, y_sym)]
    
    # Calculate Hessian matrix
    Hessian = [[sp.diff(grad[0], x_sym), sp.diff(grad[0], y_sym)],
               [sp.diff(grad[1], x_sym), sp.diff(grad[1], y_sym)]]
    
    return grad, Hessian

def Newton(x, y, tol=1e-6, max_iter=100):
    # Define symbols for x and y
    x_sym, y_sym = sp.symbols('x y')
    
    # Convert the function to a symbolic expression
    f_sym = my_fun(x_sym, y_sym)
    
    # Calculate gradient and Hessian
    grad, Hessian = calculate_gradient_and_hessian(f_sym, x_sym, y_sym)
    
    for i in range(max_iter):
        # Evaluate gradient and Hessian at the current point
        grad_val = np.array([float(grad[0].subs({x_sym: x, y_sym: y})),
                             float(grad[1].subs({x_sym: x, y_sym: y}))])
        Hessian_val = np.array([[float(Hessian[0][0].subs({x_sym: x, y_sym: y})),
                                 float(Hessian[0][1].subs({x_sym: x, y_sym: y}))],
                                [float(Hessian[1][0].subs({x_sym: x, y_sym: y})),
                                 float(Hessian[1][1].subs({x_sym: x, y_sym: y}))]])
        
        # Calculate the Newton-Raphson update
        Q_inv = np.linalg.inv(Hessian_val)
        update = np.dot(Q_inv, grad_val)
        
        # Update the point
        new_point = np.array([x, y]) - update
        
        # Check for convergence
        if np.linalg.norm(update) < tol:
            return new_point
        
        x, y = new_point
    
    return None

min_point = Newton(initial_point[0], initial_point[1])

if min_point is not None:
    print("Minimum point: ", min_point)
    print("Minimum value: ", my_fun(*min_point))
else:
    print("Newton-Raphson did not converge.")
