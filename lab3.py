import numpy as np
import matplotlib.pyplot as plt
import math, copy
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
# this is a recreation of machine learning _Linear regression_ lab 3 to practice implementing gradient descent algorithm in python
# Training data
x_train = np.array([1.0, 2.0])  # features
y_train = np.array([300.0, 500.0]) # target value

# Function to compute the cost
def compute_cost(x, y, w, b):
    m = x_train.shape[0]  # number of training examples
    total_cost = 0  # initialize total cost
    
    # Loop over each training example
    for i in range(m): 
        f_wb = w * x[i] + b  # prediction
        cost = (f_wb - y[i])**2  # squared error cost
        total_cost = 1/(2*m) * cost  # accumulate total cost
    return total_cost

# Function to compute the gradient
def compute_gradient(x, y, w, b): 
    m = x_train.shape[0]  # number of training examples
    alpha = 0.1  # learning rate
    dj_dw = 0  # initialize gradient w.r.t. w
    dj_db = 0  # initialize gradient w.r.t. b
    
    # Loop over each training example
    for i in range(m):
        f_wb = w * x[i] + b  # prediction
        dj_dw += (f_wb - y[i]) * x[i]  # accumulate gradient w.r.t. w
        dj_db += (f_wb - y[i])  # accumulate gradient w.r.t. b
    dj_dw = (1/m) * dj_dw  # average gradient w.r.t. w
    dj_db = (1/m) * dj_db  # average gradient w.r.t. b
    
    return dj_dw, dj_db

# Plot the gradients
plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()

# Function to perform gradient descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    b = b_in  # initialize b
    w = w_in  # initialize w
    J_history = []  # to store cost history
    p_history = []  # to store parameter history

    # Loop over the number of iterations
    for i in range(num_iters): 
        dj_dw, dj_db = gradient_function(x, y, w, b)  # compute gradients

        w = w - (alpha * dj_dw)  # update w
        b = b - (alpha * dj_db)  # update b
    return w, b 

# Perform gradient descent
w_final, b_final = gradient_descent(x_train, y_train, 2, 1.5, 0.1, 1000, compute_cost, compute_gradient)
print(w_final, b_final)

# Make predictions
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")



