import numpy as np


class Svm(object):
    """" Svm classifier """

    def __init__(self, inputDim, outputDim):
        self.W = None
        #########################################################################
        # TODO: 5 points                                                        #
        # - Generate a random svm weight matrix to compute loss                 #
        #   with standard normal distribution and Standard deviation = 0.01.    #
        #########################################################################
        self.W = 0.01 * np.random.randn(inputDim, outputDim)

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss(self, x, y, reg):
        """
        Svm loss function
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
        # - Compute the svm loss and store to loss variable.                        #
        # - Compute gradient and store to dW variable.                              #
        # - Use L2 regularization                                                  #
        # Bonus:                                                                    #
        # - +2 points if done without loop                                          #
        #############################################################################
        s = x.dot(self.W)
        num_train = x.shape[0]  # Get the num_train, cause x: N*L
        s_yi = s[np.arange(num_train), y].reshape(-1, 1)
        margin = s - s_yi + 1

        # Compute the svm loss
        loss_i = np.maximum(0, margin)
        loss_i[np.arange(num_train), y] = 0
        loss = np.sum(loss_i) / num_train

        # L2 regularization
        R_W = np.sum(np.square(self.W))
        loss += reg * R_W

        # Compute gradient
        margin[np.arange(num_train), y] = 0
        ds = np.zeros(margin.shape)  # N*K
        ds[margin > 0] = 1
        ds[np.arange(num_train), y] = -np.sum(ds, axis=1)
        dW = (1 / num_train) * (np.transpose(x)).dot(ds)
        d_reg = reg * 2 * self.W
        dW = dW + d_reg

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train(self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Svm classifier using stochastic gradient descent.
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
        A list containing the value of the loss at each training iteration.
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
        acc = np.mean(y == y_pre)*100
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc
