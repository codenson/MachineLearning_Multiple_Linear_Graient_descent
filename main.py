import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
"""The training set is folows : 
Size (sqft)	Number of Bedrooms	Number of floors	Age of Home	Price (1000s dollars)
2104	5	1	45	460
1416	3	2	40	232
852	    2	1	35	178

it has 4 features and 4 Training Examples (m)

"""
'''X is a matrix with dimensions ( 𝑚,  𝑛) (m rows, n columns).

x(i) is vector containing example i. x(i) =x(i)^(0),x(i)^(1),⋯,x(i)^(𝑛−1))
Matrix X below : 
 X= [
      [2104   5     1   45]
      [1416   3     2   40]
      [852    2     1   35]
                            ]

'''
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# print(X_train.shape)
# # data is stored in numpy array/matrix
# print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
# print(X_train)
# # print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
# print(y_train)

# Initialize the weights and bias with given values. 
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

def calculate_gradient_descient(w, b, alpha, x, y): 
    m = X_train.shape[0]

    f_wb  = 0 
    for i in range(m):   
        bb = np.dot(x, y)
        w = w - alpha * (1/m) * np.dot(x, y)
        b = b - alpha * (1/m) * np.dot(x, y)

    return w, b

# w_final, b_final = calculate_gradient_descient(w_init, b_init, 0.01, X_train[0], y_train)
# print(w_final, b_final)

"""The function below calculates the prediction for a single example."""
def predict_single_loop(x, w, b): 
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = w[i] * x[i] 
        p = p + p_i         
    p = p + b                
    return p

print(X_train[0], "pred for 1 loop : " , predict_single_loop(X_train[0], w_init, b_init))

def predict(x, w, b): 
    p = np.dot(x, w) + b
    return p

print(X_train[0], "pred for 1 loop using the dot vectorization : " , predict(X_train[0], w_init, b_init))

def compute_cost(X, y, w, b): 
    """"Compute Cost With Multiple Variabless"""
    m = X.shape[0]
    cost = 0.0
    for i in range(m): 
        f_wb = np.dot(X[i], w)+b 
        cost  = cost + (f_wb - y[i])**2
    cost = cost /(2*m) 
    return cost
# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

def addition(x, y): 
    return x + y

def compute_gradient(X, y, w, b): 
    m,n = X.shape  
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

print(X_train[0], f"x shape : {X_train[0].shape} and y : {w_init.shape} +  w: { w_init}")

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 

    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

#print(gradient_descent(X_train, y_train, w_init, b_init, compute_cost, compute_gradient, 0.0000001, 1000000))
# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
    