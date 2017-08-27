import numpy as np
from random import shuffle
from past.builtins import xrange

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
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_temp = X.dot(W)
  y_temp = y_temp - np.max(y_temp, axis=1).reshape((y_temp.shape[0],1))
  soft = np.exp(y_temp)
  soft_sum = np.sum(soft, axis=1)
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  for i in range(N):
      loss -= np.log(soft[i][y[i]]/soft_sum[i])
  loss /= N
  sum_sq = reg * np.sum(W**2)
  loss += sum_sq
  h = 0.00001
  p = soft/soft_sum.reshape((N,1))
  for i in range(N):
    for j in range(C):
        dW[:,j] +=  (p[i][j] - (y[i] == j)) * X[i,:]
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_temp = X.dot(W)
  y_temp = y_temp - np.max(y_temp, axis=1).reshape((y_temp.shape[0],1))
  soft = np.exp(y_temp)
  soft_sum = np.sum(soft, axis=1)
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  soft_index = soft[(list(np.arange(N)), y)]
  soft_index /= soft_sum
  loss -= np.sum(np.log(soft_index))
  loss /= N
  sum_sq = reg * np.sum(W**2)
  loss += sum_sq
  p = soft/soft_sum.reshape((N,1))
  y_expand = np.zeros_like(p)
  y_expand[(list(np.arange(N)), y)] = 1
  p_final = p - y_expand
  dW = X.T.dot(p_final)
  dW /= N
  dW += reg * W
  #finish loss

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
