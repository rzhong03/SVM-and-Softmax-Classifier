import numpy as np


class Softmax(object):
    """" Softmax classifier """

    def __init__(self, inputDim, outputDim):
        self.W = None
        #########################################################################
        # TODO: 5 points                                                        #
        # - Generate a random softmax weight matrix to use to compute loss.     #
        #   with standard normal distribution and Standard deviation = 0.01.    #
        #########################################################################
        self.W = 0.01 * np.random.randn(inputDim, outputDim)

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss(self, x, y, reg):
        """
        Softmax loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to weights self.W (dW) with the same shape of self.W.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)
        #############################################################################
        # TODO: 20 points                                                           #
        # - Compute the softmax loss and store to loss variable.                    #
        # - Compute gradient and store to dW variable.                              #
        # - Use L2 regularization                                                  #
        # Bonus:                                                                    #
        # - +2 points if done without loop                                          #
        #############################################################################
        num_train = x.shape[0]
        s = x.dot(self.W)

        # Compute the softmax loss
        s1 = s - np.max(s, axis=1, keepdims=True)
        exp_s1 = np.exp(s1)
        sum_f = np.sum(exp_s1, axis=1, keepdims=True)
        p_yi = exp_s1[np.arange(num_train), y] / sum_f
        loss_i = -np.log(p_yi)
        loss = (1/num_train) * np.sum(loss_i)

        # L2 regularization
        R_W = np.sum(np.square(self.W))
        loss += reg * R_W

        # Compute gradient
        prob = exp_s1/sum_f
        ind = np.zeros(prob.shape)
        ind[np.arange(num_train), y] = 1
        ds = prob - ind

        dW = (1/num_train) * np.transpose(x).dot(ds)
        d_reg = reg * 2 * self.W
        dW = dW + d_reg

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train(self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Softmax classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (D, batchSize)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            num_train_o = x.shape[0]

            # Sample select and no different data
            num_train = np.random.choice(num_train_o, batchSize, replace=False)
            xBatch, yBatch = x[num_train], y[num_train]

            # Gradient decent optimization
            loss, dW = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            # Update the weights
            self.W = self.W - lr * dW
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict(self, x, ):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        s = x.dot(self.W)
        # Select the highest score label
        yPred = np.argmax(s, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred

    def calAccuracy(self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        y_pre = self.predict(x)
        acc = np.mean(y == y_pre) * 100
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc
