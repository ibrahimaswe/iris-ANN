# import libraries needed 
import numpy as np              
from sklearn.datasets import load_iris      # loading the iris dataset
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler   # for scaling the data
import time               # for timing complexity 
from sklearn.preprocessing import LabelBinarizer

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets into 50% each
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)


# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN class

class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.rand(self.input_size, self.hidden_size)
        self.weights2 = np.random.rand(self.hidden_size, self.output_size)
        self.bias1 = np.random.rand(self.hidden_size)
        self.bias2 = np.random.rand(self.output_size)

    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    # derivative of sigmoid function
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    # forward propagation
    def forward_propagation(self, X):
        self.hidden_layer = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
        return self.output_layer
    # back propagation 
    def back_propagation(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer)
        # update The weights and biases
        self.weights2 += self.hidden_layer.T.dot(output_delta)
        self.weights1 += X.T.dot(hidden_delta)
        self.bias2 += np.sum(output_delta, axis=0)
        self.bias1 += np.sum(hidden_delta, axis=0)
    # train the model
    def train(self, X, y):
        output = self.forward_propagation(X)
        self.back_propagation(X, y, output)
        
# ann model with 5,2,3 topology
ann = ANN(5, 2, 3)
# ann model with 5,6,3 topology
ann2 = ANN(5, 6, 3)

# Add a bias neuron to the input data
X_train_bias = np.c_[X_train, np.ones(X_train.shape[0])]
X_test_bias = np.c_[X_test, np.ones(X_test.shape[0])]

 #Convert target labels to one-hot encoded format
lb = LabelBinarizer()
y_train_one_hot = lb.fit_transform(y_train)

# Now train the network 
start_time = time.time()
for i in range(1000):
    ann.train(X_train_bias, y_train_one_hot)  
end_time = time.time()
ann_time = end_time - start_time

# Second timer
start_time = time.time()
for i in range(1000):
    ann2.train(X_train_bias, y_train_one_hot)  
end_time = time.time()
ann2_time = end_time - start_time


# now we test our network: 
# Test the network
output1 = ann.forward_propagation(X_test_bias)
predicted1 = np.argmax(output1, axis=1)
accuracy1 = np.mean(predicted1 == y_test)

output2 = ann2.forward_propagation(X_test_bias)
predicted2 = np.argmax(output2, axis=1)
accuracy2 = np.mean(predicted2 == y_test)

print(f'Topology 5-2-3: Accuracy: {accuracy1 * 100}%, Time: {ann_time} seconds')
print(f'Topology 5-6-3: Accuracy: {accuracy2 * 100}%, Time: {ann2_time} seconds')