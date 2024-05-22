import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Set a seed for deterministic outputs
SEED = 42
np.random.seed(seed = SEED)

def plotting_GD(times,losses,gradient_norms):
    # Plotting loss vs time
    plt.figure(figsize=(10, 6))
    plt.plot(times, losses)
    plt.xlabel('CPU time (seconds)')
    plt.ylabel('Objective function')
    plt.title('Objective function vs CPU time - Gradient Descent')
    plt.show()

    # Plotting gradient norm across iterations
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_norms, label = 'Gradient Norm')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm across Iterations - Gradient Descent')
    plt.show()

def plotting_RAND(times,losses,gradient_norms):
    # Plotting loss vs time
    plt.figure(figsize=(10, 6))
    plt.plot(times, losses)
    plt.xlabel('CPU time (seconds)')
    plt.ylabel('Objective function')
    plt.title('Objective function vs CPU time - BCGD Randomized')
    plt.show()

    # Plotting gradient norm across iterations
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_norms, label = 'Gradient Norm')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm across Iterations - BCGD Randomized')
    plt.show()

def plotting_GS(times,losses,gradient_norms):
    # Plotting loss vs time
    plt.figure(figsize=(10, 6))
    plt.plot(times, losses)
    plt.xlabel('CPU time (seconds)')
    plt.ylabel('Objective function')
    plt.title('Objective function vs CPU time - BCGD Gauss-Southwell')
    plt.show()

    # Plotting gradient norm across iterations
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_norms, label = 'Gradient Norm')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm across Iterations - BCGD Gauss-Southwell')
    plt.show()

    NUM_SAMPLES = 1000
NUM_FEATURES = 1000
NUM_LABELS = 50
NUM_EXAMPLES = 1000

# A MATRIX
# Generate a 1000x1000 matrix with random samples from a standard normal distribution
# This is our data matrix, which contains 1000 samples (rows) with 1000 features each (columns)
data_matrix = np.random.normal(0, 1, size = (NUM_SAMPLES, NUM_FEATURES))
A = data_matrix 
# 'A' contains random values drawn from N(0,1)
print("A shape: ", A.shape)

# X MATRIX
# This is our weight matrix that we initialize like this ; these weights we want to learn
# it has 1000 features (rows) with 50 labels each (columns)
weight_matrix = np.random.normal(0, 1, size = (NUM_FEATURES, NUM_LABELS))
X = weight_matrix
# 'X' contains random values drawn from N(0,1)
print("X shape: ", X.shape)

# E MATRIX
# This matrix is used to help generating our supervised gold labels 
# It is of size 1000 training examples (rows) and their labels (columns)
# It acts like a sort of bias matrix
generative_matrix = np.random.normal(0, 1, size = (NUM_EXAMPLES, NUM_LABELS))
E = generative_matrix
# 'E' contains random values drawn from N(0,1)
print("E shape: ", E.shape)

# LABEL VECTOR
# Create a vector with numbers from 1 to 50
label_vector = np.arange(1, 51)

# Now calculate AX+E to generate labels for the 1000 training examples (such that we have a supervised learning set) 

result_matrix = A @ X + E

print("AX+E shape: ", result_matrix.shape)

labels = np.argmax(result_matrix, axis=1)
print("Labels shape: ", labels.shape)


m = NUM_SAMPLES # samples
d = NUM_FEATURES # features
k = NUM_LABELS   # labels

# Matrix for the encoding of all classes
I = np.zeros((m,k))
for label_idx in range(k):

    I[:,label_idx] = np.eye(k)[labels][:, label_idx]

# Indicator matrix for classes of samples (Transpose of I)
I_b = I.T


def cost_function(X,A,labels):

    term_1 = -1 * (np.diag((A @ X) @ I_b))
    
    term_2 = np.exp(A @ X) @ np.ones((k,1))

    final = (term_1 + (np.log(term_2)).flatten()) @ np.ones((m,1))

    return final  

def lipschitz(A):
    L = np.linalg.norm(A,2) * np.linalg.norm(A, 'fro')
    return L

EPSILON = 1e-6
ITERATIONS = 3000
LR = 1/lipschitz(A)

def full_gradient(X,A,labels):

    return (-1 * A.T) @ (I - ((np.exp(A @ X)) / (np.exp(A @ X) @ np.ones((k,1)))))


def partial_gradient(X,A,labels,c):

    return (-1 * A.T) @ (I[:,c] - (np.exp(A @ X[:, c]) / ((np.exp(A @ X) @ np.ones((k,1))).flatten())))


def gradient_descent(X,A,labels,lr, iterations):

    gradient_norms = []
    X_values = []
    times = []
    start_time = time.time()

    for i in range(iterations):

        # Keep track of X values 

        X_values.append(X)

        # Calculate gradient

        grad = full_gradient(X,A,labels)

        # Get the gradient norm

        norm = np.linalg.norm(grad)

        # Check the norm stopping criterium

        if norm < EPSILON:
            break

        # Keep track of gradient norms 

        gradient_norms.append(norm)

        # Gradient step

        X = X - lr * grad

        # Keep track of the time
        
        times.append(time.time() - start_time)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return X_values, gradient_norms, times

X_values, gradient_norms_GD, times_GD = gradient_descent(X,A,labels, lr= LR, iterations= ITERATIONS)

losses_GD = [cost_function(X_val, A, labels) for X_val in X_values]

plotting_GD(times_GD,losses_GD,gradient_norms_GD)

# Generate the list of random choices of classes beforehand (For better efficiency)
random_choices = []
for i in range(ITERATIONS):
    random_choices.append(random.randint(0,k-1)) 

def BCGD_randomized(X,A,labels,lr, iterations):

    # Use a copy of X so we can run the algorithm multiple times on the original X
    X_copy = X.copy()

    gradient_norms = []
    X_values = []
    times = []
    start_time = time.time()

    # Calculate the full gradient once
    grad = full_gradient(X_copy,A,labels)

    for i in range(iterations): 
        # Keep track of X values 
        X_values.append(np.copy(X_copy))

        # Get the randomly generated block
        curr_c = random_choices[i] 
        
        # Take the random block
        curr_grad = grad[:, curr_c]
        
        # Recalculate the gradient of the random block
        grad[:, curr_c] = partial_gradient(X_copy,A,labels,curr_c) 
      
        # Calculate the gradient norm
        norm = np.linalg.norm(grad)

        # Check the stopping criterium
        if norm < EPSILON:
            break

        # Keep track of the gradient norm
        gradient_norms.append(norm)

        # Gradient step
        X_copy = X_copy - lr * grad

        # Keep track of the time
        times.append(time.time() - start_time)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return X_values, gradient_norms, times


X_values, gradient_norms_BCGD, times_BCGD = BCGD_randomized(X,A,labels, lr = LR, iterations= ITERATIONS)

losses_BCGD = [cost_function(X_val, A, labels) for X_val in X_values]

plotting_RAND(times_BCGD, losses_BCGD, gradient_norms_BCGD)

def Gauss_Southwell_final(X,A, labels,lr, iterations):
    # Use a copy of X so we can run the algorithm multiple times on the original X
    X_copy = X.copy()

    gradient_norms = []
    X_values = []
    times = []
    start_time = time.time()

    # Calculate the full gradient once
    grad = full_gradient(X_copy,A,labels)

    # Calculate the gradient norm of each block
    norms = np.linalg.norm(grad, axis=0)

    # Initialize the max norm index variable
    max_norm_class_index = None

    for it in range(iterations): # iterations
        # In each iteration, check if there is a new maximal norm : Choose the biggest one for updating

        # Keep track of X values 
        X_values.append(np.copy(X_copy))
        
        # In the first iteration, we already calculated gradients for all blocks, so we do the updating only starting
        # at the second iteration
        if it > 0:
            # Recalculate the gradient of the block gradient we updated
            grad[:, max_norm_class_index] = partial_gradient(X_copy,A,labels,c=max_norm_class_index)
            # Recalculate the norm of the block gradient we updated
            norms[max_norm_class_index] = np.linalg.norm(grad[:, max_norm_class_index])

        # Get sum of norms of each block
        sum_norms = np.sum(norms)

        # Check the stopping criterium
        if sum_norms < EPSILON:
            break

        # Get the index of the column with the largest norm in the current iteration
        max_norm_class_index = np.argmax(norms)
 
        # Select the column with the largest norm
        max_norm_partial_grad = grad[:, max_norm_class_index]

        # Keep track of the gradient norm
        gradient_norms.append(sum_norms) 
        
        # Gradient step
        X_copy = X_copy- lr * grad

        # Keep track of the time
        times.append(time.time() - start_time)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return X_values, gradient_norms, times


# Let's see how it performs
X_values_BCGD_GS, gradient_norms_BCGD_GS, times_BCGD_GS = Gauss_Southwell_final(X,A,labels,lr= LR, iterations= ITERATIONS)

losses_BCGD_GS = [cost_function(X_val, A, labels) for X_val in X_values_BCGD_GS]

plotting_GS(times_BCGD_GS, losses_BCGD_GS, gradient_norms_BCGD_GS)


# Graph comparison of all three methods
plt.figure(figsize=(10, 6))
plt.plot(times_GD, losses_GD, label = 'Gradient Descent')
plt.plot(times_BCGD, losses_BCGD, label = 'Randomized BCGD')
plt.plot(times_BCGD_GS, losses_BCGD_GS, label = 'Gauss Southwell BCGD')
plt.xlabel('CPU time (seconds)')
plt.ylabel('Objective function')
plt.title('Objective function vs CPU time - All methods')
plt.grid(True)
plt.legend()
plt.show()

# !pip install ucimlrepo # If ! produces problems, %pip install ucimlrepo should work. For us !pip worked  

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
# Please note that sometimes the connection to the website fails, so it cannot fetch the data.
# In that case, just rerun this cell, then it should work.
optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 

# data (as pandas dataframes) 
X = optical_recognition_of_handwritten_digits.data.features 
y = optical_recognition_of_handwritten_digits.data.targets 

NUM_SAMPLES_ALL = X.shape[0]
NUM_FEATURES = X.shape[1]
NUM_LABELS = len(np.unique(y))

d = NUM_FEATURES
k = NUM_LABELS 


# Fit and transform the data
#X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

A = np.array(X)
labels = np.array(y)

# Split the data into training and test sets
A_train, A_test, labels_train, labels_test = train_test_split(A, labels, test_size=0.2, random_state=42)

# We will use the train split for learning
m = A_train.shape[0]

# Scale the data
scaler = StandardScaler()

# Fit and transform the training data
A_train = scaler.fit_transform(A_train)

# Transform the test data
A_test = scaler.transform(A_test)

labels_train = np.array(labels_train)
labels_test = np.array(labels_test)

weight_matrix = np.random.normal(0, 1, size = (d,k))
X = weight_matrix

E = np.random.normal(0,1 , size = (A_test.shape[0],k))

I = np.zeros((m,k))
for label_idx in range(k):

    I[:,label_idx] = np.eye(k)[labels_train.flatten()][:, label_idx]

I_b = I.T

# EPSILON and ITERATIONS stay the same
LR = 1/lipschitz(A_train)


X_values_GD, gradient_norms_GD, times_GD = gradient_descent(X,A_train,labels_train, lr= LR, iterations= ITERATIONS)

losses_GD = [cost_function(X_val, A_train, labels_train) for X_val in X_values_GD]

plotting_GD(times_GD,losses_GD,gradient_norms_GD)


# Generate the list of random choices of classes beforehand 
random_choices = []
for i in range(ITERATIONS): 
    random_choices.append(random.randint(0,k-1))


X_values_BCGD, gradient_norms_BCGD, times_BCGD = BCGD_randomized(X,A_train, labels_train,lr= LR, iterations=ITERATIONS)
losses_BCGD = [cost_function(X_val, A_train, labels_train) for X_val in X_values_BCGD]

plotting_RAND(times_BCGD, losses_BCGD, gradient_norms_BCGD)

X_values_BCGD_GS, gradient_norms_BCGD_GS, times_BCGD_GS = Gauss_Southwell_final(X,A_train,labels_train,lr= LR, iterations= ITERATIONS)

losses_BCGD_GS = [cost_function(X_val, A_train, labels_train) for X_val in X_values_BCGD_GS]

plotting_GS(times_BCGD_GS, losses_BCGD_GS, gradient_norms_BCGD_GS)


# We use softmax to calculate the probabilities of being in one of the k classes
def softmax(matrix):
    # Subtract max for numerical stability
    shift_matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    exps = np.exp(shift_matrix)
    softmax_matrix = exps / np.sum(exps, axis=1, keepdims=True)
    return softmax_matrix


def accuracy(feature_matrix, data, labels):

    matrix = (data @ feature_matrix) + E

    softmax_results = softmax(matrix)
    
    # Pick the index (= class) with the largest probability for each sample
    labels_predicted = np.argmax(softmax_results, axis=1)

    same_values = (labels_predicted == labels.flatten())

    num_same_values = np.sum(same_values)

    accuracy = (num_same_values / labels.shape[0]) * 100

    return accuracy

accuracy_GD = accuracy(X_values_GD[-1], A_test, labels_test)
accuracy_BCGD = accuracy(X_values_BCGD[-1], A_test, labels_test)
accuracy_BCGD_GS = accuracy(X_values_BCGD_GS[-1], A_test, labels_test)

names = ['Gradient Descent Normal', 'Block Gradient Descent, Randomized', 'Block Gradient Descent, Gauss-Southwell Rule']
accuracies = [accuracy_GD, accuracy_BCGD, accuracy_BCGD_GS]
for idx,_ in enumerate(accuracies):

    print("Accuracy of {} is {} percent".format(names[idx], accuracies[idx]))


# Graph comparison of all three methods
plt.figure(figsize=(10, 6))
plt.plot(times_GD, losses_GD, label = 'Gradient Descent')
plt.plot(times_BCGD, losses_BCGD, label = 'Randomized BCGD')
plt.plot(times_BCGD_GS, losses_BCGD_GS, label = 'Gauss Southwell BCGD')
plt.xlabel('CPU time (seconds)')
plt.ylabel('Objective function')
plt.title('Objective function vs CPU time - All methods')
plt.grid(True)
plt.legend()
plt.show()