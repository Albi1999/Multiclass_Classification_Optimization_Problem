import numpy as np


# Set a seed for deterministic outputs

np.random.seed(seed = 42)

# Generate a 1000x1000 matrix with random samples from a standard normal distribution
# This is our data matrix, which contains 1000 samples (rows) with 1000 features each (columns)
data_matrix = np.random.normal(0, 1, size = (1000, 1000)) # He refers to it as A 

# Now 'data_matrix' contains random values drawn from N(0,1)
#print(data_matrix)

# This is our weight matrix that we initialize like this ; these weights we want to learn
# it has 1000 features (rows) with 50 labels each (columns)
weight_matrix = np.random.normal(0, 1, size = (1000, 50)) # He refers to it as X 

# This matrix is used to help generating our supervised gold labels 
# It is of size 1000 training examples (rows) and their labels (columns)
generative_matrix = np.random.normal(0, 1, size = (1000, 50)) # He refers to it as E 


# Create a vector with numbers from 1 to 50
label_vector = np.arange(1, 51)

# Print the vector
#print(label_vector)

# Now he wants us to calculate AX+E to generate labels for the 1000 training examples (such that we have a supervised learning set)+


# Calculate the matrix product AX
AX = np.matmul(data_matrix, weight_matrix)  # or simply: AX = A @ X

# Add E to AX element-wise
result_matrix = AX + generative_matrix


print(result_matrix.shape)


# We find our labels by considering the max index in the row as the class label

# Find the column indices of maximum values for each row
max_indices = np.argmax(result_matrix, axis=1)

print(max_indices)

print(result_matrix[2,:])

# 'max_indices' now contains the column indices of maximum values for each row
# print(max_indices)

# We can check if we did correct by looking at the shape which should be (1000,) as we should have a label for each of the inputs

# print(max_indices.shape)

# Print the first few elements of the resulting matrix for verification
# print(result_matrix[:5, :5])  # Adjust the slice as needed



