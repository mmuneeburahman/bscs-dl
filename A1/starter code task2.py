# Include libraries which may use in implementation
import numpy as np
import random
import sklearn.datasets as ds
import matplotlib.pyplot as plt


# Create a Neural_Network class
class Neural_Network(object):        
    def __init__(self,inputSize = 2,hiddenlayer = 3 outputSize = 1 ):        
        # size of layers
        self.inputSize = inputSize
        self.outputSize = outputSize 
        self.hiddenLayer = hiddenlayer
        #weights
        self.W1 = ? # randomly initialize W1 using random function of numpy
        # size of the wieght will be (inputSize +1, hiddenlayer) that +1 is for bias    
        self.W2 = ? # randomly initialize W2 using random function of numpy
        # size of the wieght will be (hiddenlayer +1, outputSize) that +1 is for bias    
        
    def feedforward(self, X):
        #forward propagation through our network
        # dot product of X (input) and set of weights
        # apply activation function (i.e. whatever function was passed in initialization)    
    return ? # return your answer with as a final output of the network

    def sigmoid(self, s):
        # activation function
        return ? # apply sigmoid function on s and return it's value

    def sigmoid_derivative(self, s):
        #derivative of sigmoid
        return ? # apply derivative of sigmoid on s and return it's value 
    
    def tanh(self, s):
        # activation function
        return ? # apply tanh function on s and return it's value

    def tanh_derivative(self, s):
        #derivative of tanh
        return ? # apply derivative of tanh on s and return it's value
    
    def relu(self, s):
        # activation function
        return ? # apply relu function on s and return it's value

    def relu_derivative(self, s):
        #derivative of relu
        return ? # apply derivative of relu on s and return it's value

    def backwardpropagate(self,X, Y, y_pred, lr):
        # backward propagate through the network
        # compute error in output which is loss compute cross entropy loss function
        # applying derivative of that applied activation function to the error
        # adjust set of weights
    
    def crossentropy(self, Y, Y_pred):
        # compute error based on crossentropy loss 
        return ? #error

    def train(self, trainX, trainY,epochs = 100, learningRate = 0.001, plot_err = True ,validationX = Null, validationY = Null):
        # feed forward trainX and trainY and recivce predicted value
        # backpropagation with trainX, trainY, predicted value and learning rate.
        # if validationX and validationY are not null than show validation accuracy and error of the model by printing values.
        # plot error of the model if plot_err is true

    def predict(self, testX):
        # predict the value of testX
    
    def accuracy(self, testX, testY):
        # predict the value of trainX
        # compare it with testY
        # compute accuracy, print it and show in the form of picture
        return ? # return accuracy    
        
    def saveModel(self,name):
        # save your trained model, it is your interpretation how, which and what data you store
        # which you will use later for prediction

        
    def loadModel(self,name):
        # load your trained model, load exactly how you stored it.

def main():   
    data, label = ds.make_circles(n_samples=1000, factor=.4, noise=0.05)

    #Lets visualize the dataset
    reds = label == 0
    blues = label == 1
    plt.scatter(data[reds, 0], data[reds, 1], c="red", s=20, edgecolor='k')
    plt.scatter(data[blues, 0], data[blues, 1], c="blue", s=20, edgecolor='k')
    plt.show()

    #Note: shuffle this dataset before dividing it into three parts

    # Distribute this data into three parts i.e. training, validation and testing
    trainX = ?# training data point
    trainY = ?# training lables

    validX = ? # validation data point
    validY = ?# validation lables

    testX = ?# testing data point
    testY = ?# testing lables


    model = Neural_Network(2,1)
    # try different combinations of epochs and learning rate
    model.train(trainX, trainY, epochs = 150, learningRate = 0.001, validationX = validX, validationY = validY)

    #save the best model which you have trained, 
    model.save('bestmodel.mdl')


    # create class object
    mm = Neural_Network()
    # load model which will be provided by you
    mm.load('bestmodel.mdl')
    # check accuracy of that model
    mm.accuracy(testX,testY)


if __name__ == '__main__':
    main()