import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] += -X[i, :]  
        dW[:,j] +=  X[i,:]
      

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
  S = X @ W
  
  i_yi = (np.arange(num_train),y)
  s_minus_vector = S[i_yi]
    # R.H.S. equivalently = np.array([S[i, y[i]] for i in range(S.shape[0])]), that is s[i, y[i]]
  S_minus = np.tile( np.reshape(s_minus_vector, [S.shape[0],1]), [1, S.shape[1] ]) #replicating s[i, y[i]] in column form
  outStanding = S  - S_minus + 1
  outStanding_pos = outStanding > 0
#############IMPORTANT OBSERVATION & TREATMENT#############
  outStanding_pos[i_yi] = 0  
###############################################################
  loss = np.sum( outStanding[ outStanding_pos]) / num_train

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

#############IMPORTANT  TREATMENT##############################
  Scale = outStanding_pos.astype(int)
#############################################################################
  updateEntries = - np.sum(outStanding_pos, axis = 1)

  Scale [i_yi] = updateEntries 
    # L.H.S. equivalently = Scale[i, y[i]] for i in range(S.shape[0])
    

  dW = X.T @ Scale
  dW/=num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


############MODEL ANSWER FROM ZHIHU#################
#   scores = X.dot(W)
#   correct_class_score = scores[np.arange(X.shape[0]),y]
#   correct_class_score = np.reshape(correct_class_score, (X.shape[0], -1))
#   margin = scores - correct_class_score +1
#   margin = np.maximum(0, margin)
#   margin[np.arange(X.shape[0]),y] = 0
#   loss = np.sum(margin) / X.shape[0]
#   loss +=  reg * np.sum(W * W)
#   margin[margin > 0] = 1
#   row_sum = np.sum(margin, axis=1)                  # 1 by N
#   margin[np.arange(X.shape[0]), y] = -row_sum  
#   dW += np.dot(X.T, margin)/X.shape[0] + 2 * reg * W     # D by C
###########MODEL ANSWER ENDED########################   
    