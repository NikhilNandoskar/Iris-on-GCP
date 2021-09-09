#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def initialize_parameters_deep(layer_dims, n):
    """
    This function takes the numbers of layers to be used to build our model as
    input and otputs a dictonary containing weights and biases as parameters
    to be learned during training
    The number in the layer_dims corresponds to number of neurons in 
    corresponding layer

    @params

    Input to this function is layer dimensions
    layer_dims = List contains number of neurons in one respective layer
                 and [len(layer_dims) - 1] gives L Layer Neural Network
    
    Returns:
    
    parameters = Dictionary containing parameters "W1", "b1", . . ., "WL", "bL"
                 where Wl = Weight Matrix of shape (layer_dims[l-1],layer_dims[l])
                       bl = Bias Vector of shape (1,layer_dims[l])
    """
    # layers_dims = [250, 128, 128, 5] #  3-layer model
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # Number of layers in the network

    for l in range(1, L):          # It starts with 1 hence till len(layer_dims)
        # Initialize weights randomly according to Xavier initializer in order to avoid linear model
        parameters['W' + str(l)] = np.random.randn(layer_dims[l-1],layer_dims[l])*np.sqrt(n / layer_dims[l-1])
        # Initialize bias vector with zeros
        parameters['b' + str(l)] = np.zeros((1,layer_dims[l]))
        # Making sure the shape is correct
        assert(parameters['W' + str(l)].shape == (layer_dims[l-1], layer_dims[l]))
        assert(parameters['b' + str(l)].shape == (1,layer_dims[l]))

    # parameters = {"W [key]": npnp.random.randn(layer_dims[l-1],layer_dims[l]) [value]}
    return parameters 


# Activation functions and their derivaties:

def sigmoid(Z):
    """
    This function takes the forward matrix Z (Output of the linear layer) as the
    input and applies element-wise Sigmoid activation

    @params

    Z = numpy array of any shape
    
    Returns:

    A = Output of sigmoid(Z), same shape as Z, for the last layer this A is the
        output value from our model

    cache = Z is cached, this is useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z)) # Using numpy apply sigmoid to Z 
    cache = Z           # Cache the matrix Z
    
    return A, cache

def relu(Z):
    """
    This function takes the forward matrix Z as the input and applies element 
    wise Relu activation

    @params

    Z = Output of the linear layer, of any shape

    Returns:

    A = Post-activation parameter, of the same shape as Z
    cache = Z is cached, this is useful during backpropagation
    """
    
    A = np.maximum(0,Z) # Element-wise maximum of array elements
    # Making sure shape of A is same as shape of Z
    assert(A.shape == Z.shape)
    
    cache = Z      # Cache the matrix Z

    return A, cache

def relu_backward(dA, cache):
    """
    This function implements the backward propagation for a single Relu unit

    @params

    dA = post-activation gradient, of any shape
    cache = Retrieve cached Z for computing backward propagation efficiently

    Returns:

    dZ = Gradient of the cost with respect to Z
    """
    
    #Z = cache
    dZ = np.array(dA) # Just converting dz to a correct object.
    #print(dZ.all()==dA.all())
    #print(dZ.shape, Z.shape)
    # When z <= 0, you set dz to 0 as well, as relu sets negative values to 0 
    dZ[cache <= 0] = 0
    # Making sure shape of dZ is same as shape of Z
    assert (dZ.shape == cache.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    This function implements the backward propagation for a single Sigmoid unit

    @params

    dA = post-activation gradient, of any shape
    cache = Retrieve cached Z for computing backward propagation efficiently

    Returns:
    dZ = Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z)) # Using numpy apply Sigmoid to Z 
    dZ = dA * s * (1-s)  # This is derivatie of Sigmoid function

    # Making sure shape of dZ is same as shape of Z
    assert (dZ.shape == Z.shape)
    
    return dZ

# Softmax
def softmax(Z):
    """
    This fucntion caculates the softmax values element wise.
    Here I've implemented a stable softmax function
      
    @params

    Z = Output of the linear layer, of any shape

    Returns:

    Elementwise exponential values of the matriz Z

     """
    exp_val = np.exp(Z - np.max(Z, axis=1,keepdims=True))
    softmax_vals =  exp_val/np.sum(exp_val,axis=1,keepdims=True)
    return softmax_vals

def softmax_loss(Z,y,act_cache):
    """
    This function takes the forward matrix Z as the input and applies element 
    wise Softmax activation. It even calculates the cross entropy loss and
    derivative of cross entropy loss function

    @params

    Z = Output of the linear layer, of any shape
    Y  = Ground Truth/ True "label" vector (containing classes 0 and 1) 
         shape  = (number of examples, 1)

    Returns:

    log_loss = Returns cross entropy loss: −∑ylog(probs)
    dZ = Gradient of the cost with respect to Z
    
    """
    # Forward Pass
    # Here we'll implement a stable softmax
    m = y.shape[0]
    cache = Z
    A = softmax(Z)
    #Z,_ = act_cache
    A_back = softmax(act_cache)
    y = y.flatten()
    log_loss = np.sum(-np.log(A[range(m), y]))/m
    
    # Backward Pass
    
    dZ = A_back.copy()
    dZ[range(m), y] -= 1
    dZ /= m
    
    #dZ = (A - y)/m
    assert(A.shape == Z.shape)
    assert (dZ.shape == Z.shape)

    return A, cache, log_loss, dZ

def linear_forward(A, W, b):
    """
    This function implements the forward propagation equation Z = WX + b

    @params

    A = Activations from previous layer (or input data),
        shape = (number of examples, size of previous layer)
    W = Weight matrix of shape (size of previous layer,size of current layer)
    b = Bias vector of shape (1, size of the current layer)

    Returns:

    Z = The input of the activation function, also called pre-activation 
        parameter, shape = (number of examples, size of current layer)
    cache = Tuple containing "A", "W" and "b"; 
            stored for computing the backward pass efficiently
    """
    # print(A.shape, W.shape)
    Z = A.dot(W) + b # Here b gets broadcasted 
    #print("Debug",Z.shape,A.shape[0],W.shape[1])
    # Making sure shape of Z = (number of examples, size of current layer)
    assert(Z.shape == (A.shape[0], W.shape[1]))

    cache = (A, W, b)   # Cache all the three params 
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, y,keep_prob,predict_result,activation):
    """
    This function implements forward propagation LINEAR -> ACTIVATION layer

    @params

    A_prev = Activations from previous layer (or input data), 
             shape = (number of examples, size of previous layer)
    W = Weight matrix of shape (size of previous layer,size of current layer)
    b = Bias vector of shape (1, size of the current layer)
    Y  = Ground Truth/ True "label" vector (containing classes 0 and 1) 
         shape  = (number of examples, 1)
    keep_prob = Percentage of neurons to be kept active 
    predict_result = False while training, True when predicting the ground truth 
                     values (False only when ground truth values are not present)
                     Must be kept False if you have ground truth values
                     while predicting
    activation = The activation to be used in this layer, 
                 stored as a text string: "sigmoid" or "relu"

    Returns:
    
    When activation is Sigmoid:
    A = The output of the activation function, also called the post-activation 
        value 
    cache = Tuple containing "linear_cache" and "activation_cache";
            stored for computing the backward pass efficiently

    When activation is Softmax and Y is present during training and prediction:
    A = The output of the activation function, also called the post-activation 
        value 
    cache = Tuple containing "linear_cache" and "activation_cache";
            stored for computing the backward pass efficiently
    log_loss = Cross ENtropy loss
    dZ = Derivative of cross entropy softmax 
    
    When activation is Softmax and Y is not present during prediction:
    Z = The input of the activation function, also called pre-activation 
        parameter, shape = (number of examples, size of current layer) 
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache"
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activate_cache = sigmoid(Z)
        D = np.ones((A.shape[0],A.shape[1]))
        A = np.multiply(A,D)
        activation_cache = (activate_cache,D)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache"
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activate_cache = relu(Z)
        D = np.random.rand(A.shape[0],A.shape[1])
        #print("Relu Function ",(A.shape, D.shape))
        D = (D < keep_prob).astype(int)
        #print("Relu D", D.shape)
        A = np.multiply(A,D)
        A /= keep_prob
        activation_cache = (activate_cache,D)
        #print("Relu Activation cache", len(activation_cache))
        
    elif activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache"
        Z, linear_cache = linear_forward(A_prev, W, b)
        #print("Z values",Z)
        if predict_result: return Z
        A, activate_cache, log_loss, dZ = softmax_loss(Z,y,Z.copy())
        
        D = np.ones((A.shape[0],A.shape[1]))
        #print("Softmax D", D.shape)
        A = np.multiply(A,D)
        activation_cache = (activate_cache,D)
        #print("Softmax Activation cache", len(activation_cache))
        #print("A values", A)
        

    # Making sure shape of A = (number of examples, size of current layer)
    assert (A.shape == (A_prev.shape[0],W.shape[1]))
    cache = (linear_cache, activation_cache)
    #print(cache)
    if activation=="softmax":
        return A, cache,log_loss,dZ 
    else: 
        return A, cache
    
def L_model_forward(X, parameters, y,Output_classes,keep_prob,predict_result,activation_type):
    #print(y.shape)
    """
    This function implements forward propagation as following:
    [LINEAR->RELU]*(L-1) -> LINEAR -> SIGMOID computation
    So we apply Relu to all the hidden layers and Sigmoid to the output layer

    @params

    X = Data, numpy array of shape (number of examples, number of features)
    parameters = Output of initialize_parameters_deep() function
    Y  = Ground Truth/ True "label" vector (containing classes 0 and 1) 
         shape  = (number of examples, 1)
    keep_prob = Percentage of neurons to be kept active 
    predict_result = False while training, True when predicting the ground truth 
                     values (False only when ground truth values are not present)
                     Must be kept False if you have ground truth values
                     while predicting
    activation_type = The activation to be used in this layer, 
                      stored as a text string: "bianry" or "multiclass"
    Returns:

    When activation is Binary:
    AL = last post-activation value, also rferred as prediction from model
    caches = list of caches containing:
             every cache of linear_activation_forward() function
             (there are L-1 of them, indexed from 0 to L-1)

    When activation is Mukticlass and Y is present during training and prediction:
    A = The output of the activation function, also called the post-activation 
        value 
    cache = Tuple containing "linear_cache" and "activation_cache";
            stored for computing the backward pass efficiently
    log_loss = Cross Entropy loss
    dZ = Derivative of cross entropy softmax 

    When activation is Multiclass and Y is not present during prediction:
    Z = The input of the activation function, also called pre-activation 
        parameter, shape = (number of examples, size of current layer) 
    """
    #print(np.unique(y).shape[0])
    caches = []
    A = X
    L = len(parameters) // 2            # Number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        # For hidden layers use Relu activation
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],y,keep_prob, predict_result,activation='relu')
        #print("Relu A",A.shape)
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    # For output layer use Sigmoid activation
    if activation_type == "binary":
        AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],y, keep_prob,predict_result,activation='sigmoid')
        caches.append(cache)
        # Making sure shape of AL = (number of examples, 1)
        assert(AL.shape == (X.shape[0],1))
    
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    # For output layer use Sigmoid activation
    
    elif activation_type == "multiclass":
        if not predict_result:
            AL, cache, log_loss, dZ = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],y,keep_prob,predict_result,activation='softmax')
        else:
            Z = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],y,keep_prob,predict_result,activation='softmax')
            return Z
        #print("AL",AL.shape)
        caches.append(cache)
        # Making sure shape of AL = (number of examples, number of classes)
        assert(AL.shape == (X.shape[0],Output_classes))
        #print("Softmax A", AL.shape)
    
    if activation_type=="multiclass":
        return AL, caches, log_loss, dZ  
    else:
        return AL, caches
    
def compute_cost(AL, Y, parameters, lambd, log_loss, reg_type, activation_type):
    """
    When activation is Sigmoid:
    This function implements the Binary Cross-Entropy Cost along with l1/l2 
    regularization
    For l1:
    J = -(1/m)*(ylog(predictions)+(1−y)log(1−predictions)) + (λ/2*m)∑absolute(W)
    For l2:
    J = -(1/m)*(ylog(predictions)+(1−y)log(1−predictions)) + (λ/2*m)∑(W**2)

    When activation is Softmax:
    This function implements the Cross-Entropy Softmax Cost along with L2 
    regularization
    For l1:
    J = -(1/m)*(∑ylog(predictions)) + (λ/2*m)∑absolute(W)
    For l2:
    J = -(1/m)*(ylog(predictions)+(1−y)log(1−predictions)) + (λ/2*m)∑(W**2)
    
    @params

    AL = Probability vector corresponding to our label predictions 
         shape =  (number of examples, 1)
    Y  = Ground Truth/ True "label" vector (containing classes 0 and 1) 
         shape  = (number of examples, 1)
    parameters = Dictionary containing parameters as follwoing:
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    lambd = Regularization parameter, int
    reg_type = Type of regularization to use "l1" or "l2"
    activation_type = The activation to be used in this layer, 
                      stored as a text string: "bianry" or "multiclass"
    Returns:

    cost = Binary or Softmax Cross-Entropy Cost with l1/l2 Regularizaion 
    """
    
    m = Y.shape[0]  # Number of training examples
    if activation_type=="binary":
    # Compute loss from aL and y
        cross_entropy_cost = -(1/m)*(np.dot(np.log(AL).T,Y) + np.dot(np.log(1-AL).T,(1-Y)))
    #print(cost)
    elif activation_type=="multiclass":
        cross_entropy_cost = log_loss
        
    reg_cost = []
    W = 0
    L = len(parameters) // 2                  # number of layers in the neural network
    if reg_type=="l2":
        for l in range(1, L+1):
            W = parameters["W" + str(l)]
            reg_cost.append(lambd*1./(2*m)*np.sum(W**2))
    elif reg_type=="l1":
        for l in range(1, L+1):
            W = parameters["W" + str(l)]
            reg_cost.append(lambd*np.sum(abs(W)))
        

    cross_entropy_cost = np.squeeze(cross_entropy_cost) # To make sure cost's is scalar (e.g. this turns [[cost]] into cost)
    assert(cross_entropy_cost.shape == ())
    cost = cross_entropy_cost + np.sum(reg_cost)
    #print("Cost",(cost,log_loss))

    return cost

def linear_backward(dZ,l_cache,keep_prob,lambd,reg_type):
    """
    This function implements the linear portion of backward propagation for a 
    single layer (layer l)

    @params

    dZ = Gradient of the cost with respect to the linear output of current 
         layer l, shape = (number of examples, size of current layer)
    cache = Tuple of values (A_prev, W, b) coming from the forward propagation 
            in the current layer
    keep_prob = Percentage of neurons to be kept active 
    lambd = Regularization parameter, int
    reg_type = Type of regularization to use "l1" or "l2"
    
    Returns:

    dA_prev = Gradient of the cost with respect to the activation of the 
              previous layer l-1, 
              same shape as A_prev(number of examples, size of previous layer)
    dW = Gradient of the cost with respect to W of current layer l, 
         same shape as W(size of previous layer,size of current layer)
    db = Gradient of the cost with respect to b of current layer l, 
         same shape as b(1,size of current layer)
    """
    
    
    if reg_type=="l2":
        A_prev, W, b = l_cache
        #print("1 Softmax, 2 Relu W", W.shape)
        #print("Backward A_prev for cache",A_prev.shape)
        
        m = A_prev.shape[0] # Number of training examples
        dW = (1/m)*np.dot(A_prev.T,dZ) + (1/m)*lambd*W  # Derivative wrt Weights
        db = (1/m)*np.sum(dZ, axis=0, keepdims=True)  # Derivative wrt Bias
        dA_prev = np.dot(dZ,W.T)

    elif reg_type=="l1":
        A_prev, W, b = l_cache
        m = A_prev.shape[0] # Number of training examples
        #if W.any()>0:
        dW_pos = (W > 0)*lambd # wherever weights are positive(+)lambd from weights
        dW_neg = (W < 0)*-lambd # wherever weights are negative(-)lambd from weights
        dW = (1/m)*np.dot(A_prev.T,dZ) + (dW_pos + dW_neg)
        db = (1/m)*np.sum(dZ, axis=0, keepdims=True)  # Derivative wrt Bias
        dA_prev = np.dot(dZ,W.T)
        
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache,keep_prob, lambd, y, reg_type,first_layer, activation):
    """
    This function implements backward propagation for LINEAR -> ACTIVATION layer
    
    @params

    dA = post-activation gradient for current layer l 
    cache = tuple of values (linear_cache, activation_cache) 
            we store for computing backward propagation efficiently
    keep_prob = Percentage of neurons to be kept active 
    lambd = Regularization parameter, int
    Y  = Ground Truth/ True "label" vector (containing classes 0 and 1) 
         shape  = (number of examples, 1)
    reg_type = Type of regularization to use "l1" or "l2"
    first_layer = True only for first layer i.e. the input layer. It is True 
                  because while unpacking the tuple cache it has only "Two" values
                  "linear" and "activation" cache, cached durinng forward pass.
                  For other layers it is False as it has to unpack "Four"values
                  of "linear" and "activation" cache, from current and next
                  layer during backward class (current and previous layer in 
                  terms of forward pass)
    activation = the activation to be used in this layer, 
                 stored as a text string: "sigmoid" or "relu"
    
    Returns:

    dA_prev = Gradient of the cost with respect to the activation of the 
              previous layer l-1, 
              same shape as A_prev(number of examples, size of previous layer)
    dW = Gradient of the cost with respect to W of current layer l, 
         same shape as W(size of previous layer,size of current layer)
    db = Gradient of the cost with respect to b of current layer l, 
         same shape as b(1,size of current layer)
    """
    
    if activation == "relu":
        if not first_layer:
            # Unpacking Four Tuple Values from Cache
            curr_l_a_cache, next_l_a_cache = cache
            curr_linear_cache, curr_activation_cache = curr_l_a_cache  
            next_linear_cache, next_activation_cache = next_l_a_cache
            Z,_ = curr_activation_cache
            _,D = next_activation_cache 
            dZ = relu_backward(dA,Z)
            dA_prev, dW, db = linear_backward(dZ, curr_linear_cache,keep_prob,lambd,reg_type)
            dA_prev = np.multiply(dA_prev,D)
            dA_prev /= keep_prob
        else: #Unpacking Two Tuple Values from Cache
            curr_linear_cache, curr_activation_cache = cache
            Z,_ = curr_activation_cache
            dZ = relu_backward(dA,Z)
            dA_prev, dW, db = linear_backward(dZ, curr_linear_cache,keep_prob,lambd,reg_type)

    elif activation == "sigmoid":
        # Unpacking Four Tuple Values from Cache
        curr_l_a_cache, next_l_a_cache = cache
        curr_linear_cache, curr_activation_cache = curr_l_a_cache  
        next_linear_cache, next_activation_cache = next_l_a_cache

        Z,_ = curr_activation_cache
        _,D = next_activation_cache
        #print("D",D.shape)
        dZ = sigmoid_backward(dA,Z)
        #print("dZ shape",(dZ.shape,D.shape))
        dA_prev, dW, db = linear_backward(dZ, curr_linear_cache,keep_prob,lambd,reg_type)
        dA_prev = np.multiply(dA_prev,D)
        dA_prev /= keep_prob
        #Z,_ = activation_cache
        #dZ = sigmoid_backward(dA,Z)
        #dA_prev, dW, db = linear_backward(dZ, linear_cache,activation_cache,keep_prob,lambd,reg_type)
    
    elif activation == "softmax":
        # Unpacking Four Tuple Values from Cache
        curr_l_a_cache, next_l_a_cache = cache
        curr_linear_cache, curr_activation_cache = curr_l_a_cache  
        next_linear_cache, next_activation_cache = next_l_a_cache

        Z,_ = curr_activation_cache
        _,D = next_activation_cache
        #print("D",D.shape)
        _,_,_,dZ = softmax_loss(dA, y, Z)
        #print("dZ shape",(dZ.shape,D.shape))
        dA_prev, dW, db = linear_backward(dZ, curr_linear_cache,keep_prob,lambd,reg_type)
        dA_prev = np.multiply(dA_prev,D)
        dA_prev /= keep_prob
        #print("Softmax dA", dA_prev.shape)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, keep_prob, lambd,reg_type, activation_type):
    """
    This function implements the backward propagation as following: 
    [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    @params

    AL = probability vector, output of the L_model_forward function
    Y = Ground Truth/ True "label" vector (containing classes 0 and 1) 
        shape  = (number of examples, 1)
    caches = list of caches containing:
             every cache of linear_activation_forward function with "relu" 
             (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
             the cache of linear_activation_forward function with "sigmoid" 
             (it's caches[L-1])
    keep_prob = Percentage of neurons to be kept active 
    lambd = Regularization parameter, int
    reg_type = Type of regularization to use "l1" or "l2"
    activation_type = The activation to be used in this layer, 
                      stored as a text string: "bianry" or "multiclass"
    
    Returns:

    grads = Dictionary with the gradients
            grads["dA" + str(l)] = ... 
            grads["dW" + str(l+1)] = ...
            grads["db" + str(l+1)] = ... 
    """

    grads = {}
    L = len(caches) # the number of layers
    #print(L)
    m = AL.shape[0] # Number of training examples
    
    
    # Initializing the backpropagation
    # Derivative of Binary Cross Entropy function
    if activation_type=="binary":
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        # Lth layer (SIGMOID -> LINEAR) gradients. 
        # Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = (caches[L-1],caches[L-2]) # Grabbig correct dropout mask of the previous layer (wrt Forward pass)
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, keep_prob,lambd,Y,reg_type,first_layer=False,activation = "sigmoid")
    
    elif activation_type=="multiclass":
        #Y = Y.reshape(AL.shape)
        #curr_cache = caches[L-2]
        current_cache = (caches[L-1],caches[L-2]) # Grabbig correct dropout mask of the previous layer (wrt Forward pass)
        #print("Softmax CC",len(current_cache))
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(AL, current_cache, keep_prob,lambd, Y,reg_type,first_layer=False,activation = "softmax")
        #print("Softmax_grad",grads["dA"+str(L-1)])
        
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        #print("l",l) #l = 1,0
        # lth layer: (RELU -> LINEAR) gradients
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        if l != 0:
            first_layer = False
            current_cache = (caches[l], caches[l-1]) # Grabbig correct dropout mask of the previous layer (wrt Forward pass)
            #print("Relu CC",len(current_cache))
        elif l==0:
            first_layer = True 
            current_cache = caches[l] # No dropout is appkied to the first/input layer
            
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,keep_prob, lambd, Y,reg_type,first_layer,activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        #print(grads)
    return grads

def initialize_adam(parameters) :
    """
    This function Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters
    
    @param
    
    parameters = Dictionary containing parameters as follwoing:
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    
    v = Dictionary that will contain the exponentially weighted average of the gradient
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s = Dictionary that will contain the exponentially weighted average of the squared gradient
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
    
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                              beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    This function updates our model parameters using Adam

    @params
    
    parameters = Dictionary containing our parameters:
                  parameters['W' + str(l)] = Wl
                  parameters['b' + str(l)] = bl
    grads = Dictionary containing our gradients for each parameters:
                  grads['dW' + str(l)] = dWl
                  grads['db' + str(l)] = dbl
    v = Adam variable, moving average of the first gradient, python dictionary
    s = Adam variable, moving average of the squared gradient, python dictionary
    learning_rate = the learning rate, scalar.
    beta1 = Exponential decay hyperparameter for the first moment estimates 
    beta2 = Exponential decay hyperparameter for the second moment estimates 
    epsilon = hyperparameter preventing division by zero in Adam updates

    Returns:
    
    parameters = Dictionary containing our updated parameters 
    v = Adam variable, moving average of the first gradient, python dictionary
    s = Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l+1)] = beta1 * v['dW' + str(l+1)] + (1 - beta1) * grads['dW' + str(l+1)]
        v["db" + str(l+1)] = beta1 * v['db' + str(l+1)] + (1 - beta1) * grads['db' + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] = v['dW' + str(l+1)] / float(1 - beta1**t)
        v_corrected["db" + str(l+1)] = v['db' + str(l+1)] / float(1 - beta1**t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l+1)] = beta2 * s['dW' + str(l+1)] + (1 - beta2) * (grads['dW' + str(l+1)]**2)
        s["db" + str(l+1)] = beta2 * s['db' + str(l+1)] + (1 - beta2) * (grads['db' + str(l+1)]**2)
          ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".  
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / float(1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / float(1 - beta2**t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
    
    return parameters, v, s

def random_mini_batches(X, Y, mini_batch_size):
    """
    This function creates a list of random minibatches from (X, Y)
    
    @params
    
    X = Data, numpy array of shape (number of examples, number of features)
    Y = Ground Truth/ True "label" vector (containing classes 0 and 1) 
        shape = (number of examples, 1)
    mini_batch_size = size of the mini-batches (suggested to use powers of 2)
    
    Returns:
    
    mini_batches = list of synchronous (mini_batch_X, mini_batch_Y)
    
    """
    
    np.random.seed(0)            
    m = X.shape[0]                  # Number of training examples
    mini_batches = []               # List to return synchronous minibatches
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    #print("S_X",shuffled_X.shape)
    shuffled_Y = Y[permutation].reshape((m,1))
    #print("S_Y",shuffled_Y.shape)
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = (m//mini_batch_size) # number of mini batches of size mini_batch_size in our partitionning
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size,:]
        #print("M_X",mini_batch_X.shape)
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)   # Tuple for synchronous minibatches
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size :,: ]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size :,: ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def predict(X, parameters,y,Output_classes, keep_prob,predict_result,activation_type, flags):
    
    """
    This function is used to predict the results of a  L-layer neural network
    
    @ params

    X = Data, numpy array of shape (number of examples, number of features)
    Y = Ground Truth/ True "label" vector (containing classes 0 and 1) 
        shape = (number of examples, 1)
    parameters = Parameters of trained model returned by L_layer_model function
    keep_prob = Percentage of neurons to be kept active 
    predict_result = False while training, True when predicting the ground truth 
                     values (False only when ground truth values are present)
                     Must be kept False if you have ground truth values
                     while predicting
    activation_type = The activation to be used in this layer, 
                      stored as a text string: "bianry" or "multiclass" 
    flags = During prediction sometime we have grounnd truth values and 
            sometime we have to predict ground truth values using learned 
            parameters during training.
            so flags is "y_is_present" or "predict_y"
    Returns:

    Predictions for the given dataset X
    """
    
    m = X.shape[0] # Number of training examples in Dataset
    n = len(parameters) // 2 # Number of layers in the neural network
    
    if activation_type=="multiclass":
        if flags=="y_is_present":
            # Forward propagation
            AL, _, _, _ = L_model_forward(X, parameters, y,Output_classes,keep_prob, predict_result,activation_type)
        elif flags == "predict_y":
            Z = L_model_forward(X, parameters, y,Output_classes,keep_prob, predict_result,activation_type)
            AL = softmax(Z)   # Apply stable Softmax 

        predicted_class = np.argmax(AL, axis=1) # Prediction
        
    elif activation_type=="binary":
        p = np.zeros((m,1))
        #Forward Propagation
        probas, _ = L_model_forward(X, parameters,y,Output_classes,keep_prob, predict_result,activation_type)
        for i in range(probas.shape[0]):
        # As per sigmoid, values greater than 0.5 are categorized as 1
        # and values lesser than 0.5 as categorized as 0
            if probas[i] > 0.5:
                p[i] = 1
            else:
                p[i] = 0
    
    if flags == "y_is_present" and activation_type=="multiclass":
        #acc = np.sum((predicted_class == y)/m)*100
        #print("Accuracy:%.2f%%" % acc)
        #print('Accuracy: {0}%'.format(100*np.mean(predicted_class == y)))
        return predicted_class
    elif flags == "y_is_present" and activation_type=="binary":
        y = y.reshape(p.shape)
        acc = np.sum((p == y)/m)*100
        print("Accuracy:%.2f%%" % acc)
        return p
        
    if flags == "predict_y" and activation_type=="multiclass":
        ret = np.column_stack((y, predicted_class)).astype(int)
        # Saving the Predictions as Multiclass_Predictions.csv  
        pd.DataFrame(ret).to_csv("Multiclass_Predictions.csv", sep = ",", header = ["Id", "label"], index = False)
        return predicted_class
    
    elif flags == "predict_y" and activation_type=="binary":
        ret = np.column_stack((y, p)).astype(int)
        # Saving the Predictions as Binary_Predictions.csv
        pd.DataFrame(ret).to_csv("Binary_Predictions.csv", sep = ",", header = ["Id", "label"], index = False)
        return p
    
def api_prediction(X, parameters,y,Output_classes, keep_prob,predict_result,activation_type, flags):
    m = X.shape[0] # Number of training examples in Dataset
    #n = len(parameters) // 2 # Number of layers in the neural network
    
    if activation_type=="multiclass":
        if flags=="y_is_present":
            # Forward propagation
            AL, _, _, _ = L_model_forward(X, parameters, y,Output_classes,keep_prob, predict_result,activation_type)
        elif flags == "predict_y":
            Z = L_model_forward(X, parameters, y,Output_classes,keep_prob, predict_result,activation_type)
            AL = softmax(Z)   # Apply stable Softmax 

        predicted_class = np.argmax(AL, axis=1) # Prediction
        return predicted_class
    elif activation_type=="binary":
        p = np.zeros((m,1))
        #Forward Propagation
        probas, _ = L_model_forward(X, parameters,y,Output_classes,keep_prob, predict_result,activation_type)
        for i in range(probas.shape[0]):
        # As per sigmoid, values greater than 0.5 are categorized as 1
        # and values lesser than 0.5 as categorized as 0
            if probas[i] > 0.5:
                p[i] = 1
            else:
                p[i] = 0
    
        return p

def L_layer_model(X, Y, Output_classes,layers_dims, activation_type, reg_type, keep_prob=1,learning_rate = 0.01, mini_batch_size = 128,n=2,lambd=0.7,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, predict_result=False,print_cost = True): #lr was 0.009
    """
    This function implements a L-layer neural network: 
    [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    
    Arguments:
    X = Data, numpy array of shape (number of examples, number of features)
    Y = Ground Truth/ True "label" vector (containing classes 0 and 1) 
        shape = (number of examples, 1)
    layers_dims = List contains number of neurons in one respective layer
                  and [len(layer_dims) - 1] gives L Layer Neural Network
    activation_type = The activation to be used in this layer, 
        stored as a text string: "bianry" or "multiclass"
    reg_type = Type of regularization to use "l1" or "l2"
    keep_prob = Percentage of neurons to be kept active 
    learning_rate = learning rate of the gradient descent update rule
    n = 1 or 2, used for random initialization of weights, when 
        n = 1, we get LeCun Initializer
        n = 2, we get He Initializer
    lambd = Regularization parameter, int
    num_epochs = number of epochs
    predict_result = False while training, True when predicting the ground truth 
                     values (False only when ground truth values are present)
                     Must be kept False if you have ground truth values
                     while predicting
    print_cost = if True, it prints the cost every 10 steps
    
    Returns:
    parameters = parameters learnt by the model. They are used during prediction
    """
    np.random.seed(1)
    costs = []                         # keep track of cost
    t = 0                              # Used in Adam
    print(learning_rate)
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims,n)
    v, s = initialize_adam(parameters)
    
    # MiniBatch Gradient Descent
    for i in range(num_epochs):
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            if activation_type == "binary":
                AL, caches = L_model_forward(minibatch_X, parameters, minibatch_Y,Output_classes, keep_prob,predict_result, activation_type)
                # Compute cost
                cost = compute_cost(AL, minibatch_Y, parameters, lambd,0,reg_type,activation_type)
                #print(cost)
                # Backward propagation
                grads = L_model_backward(AL, minibatch_Y, caches,keep_prob,lambd,reg_type,activation_type)
                # Update parameters as per Adam
                t += 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)

            elif activation_type == "multiclass":
                AL, caches, log_loss, dZ = L_model_forward(minibatch_X, parameters,minibatch_Y,Output_classes,keep_prob,predict_result, activation_type)
                # Compute cost
                cost = compute_cost(AL, minibatch_Y, parameters, lambd, log_loss,reg_type,activation_type)
                #print(cost)
                # Backward propagation
                grads = L_model_backward(AL, minibatch_Y, caches,keep_prob,lambd,reg_type, activation_type)
                # Update parameters as per Adam
                t += 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)

            
        # Print the cost every 10 training example
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.grid()
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per tens)')
    plt.title(("Learning rate = {}, Lambda = {} ".format(str(learning_rate),str(lambd))))
    plt.show()
    #result = parameters
    # Saving model to disk
    
    return parameters




