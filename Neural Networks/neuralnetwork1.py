#!/usr/bin/python
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

#generate dataset and show

np.random.seed(0)
X, y = datasets.make_moons(100, noise=0.3)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()

#plot decision boundary
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.show()

#visualize function
def visualize(X, y, clf):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
plt.title("Logistic Regression")


class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength

'''
#function to train using smart way
#nodes= #of nodes within hidden layer
#passes= number of training attempts
#losses= if true, print losses every 1000 passes 
def smarttrain(nodes, passes, losses):
	#learn random parameters
	np.random.seed(0)
	W1= np.random.randn(nn_input_dim, nodes) / np.sqrt(nn_input_dim)
	b1= np.zeros((1, nodes))
	W1= np.random.randn(nodes, nn_output_dim) / np.sqrt(nodes) 
	b2= np.zeros((1, nn_output_dim))
	
	model= {} #this is returned at the end
	
	for i in xrange(0, passes):
		
		#forward propagation
		z1 = X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)
		probability= expscore / np.sum(expscore, axis= 1, keepdims= True)
		
		#backwards(propagation)
		delta3= probablility
		delta3[range(num_examples), y] -= 1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
		dW1 = np.dot(X.T, delta2)
		db1 = np.sum(delta2, axis=0)
		
		#add regularization
		dW2 += reg_lambda * W1
		dW1 += reg_lambda * W2
		
		#update gradient descend
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

		#assign new values to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        #print loss, resouse-intenisve so only every 1000 iter.
        if print_loss and i % 1000 == 0:
			print "Loss after iteration %i: %f" %(i, calculate_loss(model))
	
	return model
'''
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. 
    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))

    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1

        # Gradient descent parameter update
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))

    return model


clf= linear_model.LogisticRegressionCV()
clf.fit(X, y)
#plot decision boundary (the area between where the program things the seperation area is

plot_decision_boundary(lambda x: clf.predict(x), X, y)
plt.title('Logistic Regression')
plt.show()

num_examples= len(X) #training set size
nn_input_dim= 2 # inner layer dimensions
nn_output_dim= 2 # output layer dimensions

#gradientparameters
epsilon= 0.01 # this defines the learning rate
reg_lambda= 0.01 #regularization

#make the model 

model= build_model(X, y, 3, num_passes= 7000, print_loss=False)

#plot decision boundary
plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.title('hidden layer size 3')
plt.show()

