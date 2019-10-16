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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  S = X @ W # N by C
  
  for i in range(num_train):
    this_loss = 0.0
    this_dW = np.zeros_like(W)
    for j in range(num_class):
      this_loss += np.exp(S[i,j])
    for m in range(W.shape[0]):
      for n in range(W.shape[1]):
        this_dW[m,n] = np.exp(S[i,n])  / this_loss
        if n == y[i]:
          this_dW[m,n] -= 1
        this_dW[m,n] *= X[i,m]
    this_loss = np.log(this_loss)
    this_loss -= S[i,y[i]]
    loss += this_loss
    dW += this_dW
    
  dW /= num_train
  loss /= num_train
    
  loss += reg * np.sum(W * W)
  dW += 2*reg*W  
##################################
#######MODEL ANSWER##############
#   num_train = X.shape[0]
#   num_class = W.shape[1]
#   scores = X.dot(W)                                                  # N by C
#   p = np.zeros_like(W)
#   dW_each = np.zeros_like(W)
#   p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # N by C
#   loss_selector = np.zeros_like(p)
#   loss_selector[np.arange(num_train),y] = 1.0
#   loss_model = 0
#   for i in range(num_train):
#     for j in range(num_class):
#         loss_model -= loss_selector[i,j] * np.log(p[i,j])
#         dW_each[:, j] = -(loss_selector[i, j] - p[i, j]) * X[i, :].T
#     dW += dW_each
#   loss_model /= num_train
#   loss_model +=  reg * np.sum(W * W)
#   dW /= num_train
#   dW += 2 * reg * W
#   loss = loss_model
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
  num_train = X.shape[0]
  num_class = W.shape[1]
  S = X @ W  # N by C
  E = np.exp(S)
  mask = np.zeros_like(S)
  mask[np.arange(num_train),y] = 1
  loss = np.sum(np.log(np.sum(E,axis = 1))) - np.sum(S * mask)
 
  normalize_vector = np.sum(E,axis = 1) # N dim vector
  Normalize = np.tile(normalize_vector, (num_class,1)).T
  Scale = E / Normalize - mask
  dW = X.T @ Scale

  loss /=  num_train
  loss += np.sum(W*W) * reg

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

