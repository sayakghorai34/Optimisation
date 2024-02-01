import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
count =0
################################################################################################################################ 
def function_f(x, y):
    return (1-x)**2 + 100*(y-x**2)**2
    # return 2*(x+1)**2 + 4*y**2
    # return x**2 + 2*y**2

def gradient(x, y):
    x_sym, y_sym = sp.symbols('x y')
    f = function_f(x_sym, y_sym)
    gradient = [sp.diff(f, x_sym), sp.diff(f, y_sym)]
    grad = np.array([float(gradient[0].subs({x_sym: x, y_sym: y})), float(gradient[1].subs({x_sym: x, y_sym: y}))])
    print(grad)
    return grad


def get_learning_rate(x, y):
    x_sym, y_sym = sp.symbols('x y')
    f = function_f(x_sym, y_sym)
    grad = gradient(x, y)
    
    # Calculate Hessian matrix (Q)
    Q = np.array([[float(sp.diff(sp.diff(f, x_sym), x_sym).subs({x_sym: x, y_sym: y})),
                   float(sp.diff(sp.diff(f, x_sym), y_sym).subs({x_sym: x, y_sym: y}))],
                  [float(sp.diff(sp.diff(f, y_sym), x_sym).subs({x_sym: x, y_sym: y})),
                   float(sp.diff(sp.diff(f, y_sym), y_sym).subs({x_sym: x, y_sym: y}))]])
    
    # Avoid division by zero or very small values
    denominator = np.dot(np.dot(Q, grad), grad)
    if np.abs(denominator) < 1e-6:
        denominator = 1e-6
    
    return abs(np.dot(grad, grad) / denominator)

    
# Steepest gradient descent
def steepest_gradient_descent(initial_point, iterations):
    points = [initial_point]
    current_point = initial_point
    
    for i in range(iterations):
        grad = gradient(current_point[0], current_point[1])
        current_point = current_point - get_learning_rate(current_point[0],current_point[1]) * grad
        points.append(current_point)
    
    return np.array(points)
###############################################################################################################################
# Initial point and parameters
initial_point = np.array([2,2])
learning_rate = get_learning_rate(initial_point[0],initial_point[1])
iterations = 30

# Perform steepest gradient descent
iterative_points = steepest_gradient_descent(initial_point, iterations)


# Calculate the minimum and maximum values for x and y from the iterative points
min_x = iterative_points[:, 0].min()
max_x = iterative_points[:, 0].max()
min_y = iterative_points[:, 1].min()
max_y = iterative_points[:, 1].max()

# Generate contour plot
x = np.linspace(min_x-5, max_x+5, 400)
y = np.linspace(min_y-5, max_y+5, 400)
X, Y = np.meshgrid(x, y)
Z = function_f(X, Y)

# Get function values at the iterative points for contour levels
contour_levels = sorted([function_f(p[0], p[1]) for p in iterative_points])

# Plot contours passing through the iterative points
plt.contour(X, Y, Z, levels=contour_levels, colors='gray')

# Plot the iterative points and line segments
for i in range(1, len(iterative_points)):
    plt.plot([iterative_points[i-1, 0], iterative_points[i, 0]], [iterative_points[i-1, 1], iterative_points[i, 1]], 'bo-')

# Plot the origin (minimizer)
plt.plot(0, 0, 'ro', label='Minimizer (0, 0)')

# Set plot limits based on the min and max values of x and y
plt.xlim(min_x-5, max_x+5)
plt.ylim(min_y-5, max_y+5)

plt.axhline(0, color='black', linewidth=0.2)
plt.axvline(0, color='black', linewidth=0.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Steepest Gradient Descent')
plt.show()
print(count)

