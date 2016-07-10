import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in np.arange(num_train):
    scores = np.exp(X[i,].dot(W))
    y_pred = scores/np.sum(scores)
    for j in np.arange(num_class):
      dW[:,j] += y_pred[j] * X[i,:].T
      if y[i] == j:
        loss += -np.log(y_pred[j])
        dW[:,j] -= X[i,:].T
  
  loss /= num_train
  loss += reg*0.5*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW /= num_train
  dW += reg*W
  return loss, dW

def softmax(x):
  expX = np.exp((x.T - np.max(x, axis = 1).T).T)
  x = expX / np.sum(expX, axis=1, keepdims = True)
  return x

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  idx = xrange(num_train)
  score_softmax = softmax(X.dot(W))
  loss = -np.sum(np.log(score_softmax[idx, y]))
  loss /= num_train
  loss += reg*0.5*np.sum(W*W)
  score_softmax[idx, y] += -1
  
  dW = X.T.dot(score_softmax)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

