import numpy as np


# ACTIVATION FUNCTIONS
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z

    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)

    return dZ





# INITIALIZE PARAMETERS
def init_params(layer_dims):
    np.random.seed(1)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

    assert(params['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert(params['b' + str(l)].shape == (layer_dims[l], 1))

    return params


# FORWARD PROPAGATION
def forward_layer(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    linear_cache = (A_prev, W, b)
    activation_cache = (Z)
    cache = (linear_cache, activation_cache)

    return A, cache


def forprop(X, params):
    caches = []
    A = X
    L = len(params) // 2

    # Input and hidden layers
    for l in range(1, L):
        A_prev = A
        A, cache = forward_layer(A_prev, params['W'+str(l)], params['b'+str(l)], 'relu')
        caches.append(cache)

    # Final activation
    AL, cache = forward_layer(A, params['W'+str(L)], params['b'+str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches


# BACKWARD PROPAGATION
def backward_layer(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    # Compute gradients
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def backprop(AL, Y, caches):
    grads = {}
    L = len(caches) # layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # derivative of cost with respect to last layer
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = backward_layer(dAL, current_cache, 'sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_layer(grads['dA'+str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# UDPATE PARAMETERS
def update_params(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]  - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


# COST FUNCTION
def compute_cost(AL, Y):
    m = Y.shape[1]
    # cross-entropy loss
    cost = -(1/m) * float(np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL)))
    cost = np.squeeze(cost)

    return cost




# MODEL CLASS
class NeuralNet:
    def __init__(self):
        self.params = []
        self.cost = []

    def train(self, X, Y, layer_dims, learning_rate, iterations, print_cost=False):
        np.random.seed(1)
        costs = []

        # Initialize parameters
        params = init_params(layer_dims)

        for i in range(0, iterations):
            AL, caches = forprop(X, params)
            cost = compute_cost(AL, Y)
            grads = backprop(AL, Y, caches)
            params = update_params(params, grads, learning_rate)

            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == iterations:
                costs.append(cost)

        self.params = params
        self.cost = cost

    def partial_train(self, X, Y, layer_dims, learning_rate, iterations, print_cost=False):
        np.random.seed(1)
        costs = []

        # Initialize parameters
        if self.params == []:
            params = init_params(layer_dims)
        else:
            params = self.params

        for i in range(0, iterations):
            AL, caches = forprop(X, params)
            cost = compute_cost(AL, Y)
            grads = backprop(AL, Y, caches)
            params = update_params(params, grads, learning_rate)

            if i % 100 == 0 or i == iterations:
                costs.append(cost)

        self.params = params
        self.cost = cost



    def test_accuracy(self, X, y):
        params = self.params
        m = X.shape[1]
        p = np.zeros((1,m))

        # Forward propagation
        probas, caches = forprop(X, params)

        # convert probs to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        print("Accuracy: "  + str(np.sum((p == y)/m)))

        return p


    def predict(self, X):
        params = self.params
        m = X.shape[1]
        p = np.zeros((1,m))

        # Forward propagation
        probas, caches = forprop(X, params)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        return p