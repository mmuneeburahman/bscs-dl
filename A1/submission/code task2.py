# Include libraries which may use in implementation
import numpy as np
import random
import sklearn.datasets as ds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.special import xlogy


# Create a Neural_Network class
class Neural_Network(object):        
    def __init__(self,inputSize = 2,hiddenlayer = 3, outputSize = 1 ):       
        # size of layers
        self.inputSize = inputSize
        self.outputSize = outputSize 
        self.hiddenLayer = hiddenlayer
        
        #weights
        self.W1 = np.random.randn(inputSize+1, hiddenlayer)
        self.W2 = np.random.randn(hiddenlayer+1, outputSize)
        self.previous = np.zeros((4,1))

    def feedforward(self, X):
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1) #shape = X.shape[0]x3
        self.z1 = np.dot(X, self.W1) #z1
        self.a1 = self.tanh(self.z1) #a1

        self.a1 = np.append(np.ones((self.a1.shape[0], 1)), self.a1, axis=1) #shape = X.shape[0]x4
        self.z2 = np.dot(self.a1, self.W2) #z2
        self.a2 = self.sigmoid(self.z2) #a2
        return self.a2

    def backwardpropagate(self,X, Y, y_pred, lr):
        #layer23 weights updation
        self.loss_derivative = self.d_ce(Y, y_pred) # loss
        # self.loss
        self.da2dz2 = self.d_sigmoid(self.z2) #z2 sigmoid derivative
        #BP1 completed here
        self.bp1 = self.da2dz2*self.loss_derivative #10000x1
        #BP4 completed here
        deltaW2 = np.dot(self.a1.T, self.bp1)   #3x1
        self.W2 = self.W2-(lr*deltaW2)

        #layer12 weight updation
        self.loss = np.matmul(self.bp1, self.W2.T)[:, 1:]*self.d_tanh(self.z1) #10000x4
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1) #shape = X.shape[0]x3
        deltaW1 = np.dot(X.T, self.loss)
        self.W1 = self.W1-(lr*deltaW1)

    def train(self, trainX, trainY, batch_size, epochs = 100, learningRate = 0.001, validationX = None, validationY = None):
        n_examples = trainX.shape[0]
        n_batches = n_examples//batch_size
        print("train")
        epoch_loss = []
        epoch_training_accuracy = []
        epoch_validation_accuracy = []
        for epoch in range(epochs):
            loss=0
            for i in range(n_batches):
                j, k = i*batch_size, i*batch_size+batch_size
                if k>n_examples:
                    k = n_examples-1
                y_hat = self.feedforward(trainX[j:k])
                loss += self.ce(trainY[j:k], y_hat)
                self.backwardpropagate(trainX[j:k], trainY[j:k], y_hat,learningRate)
            epoch_loss.append(loss/n_batches)
            accuracy = self.accuracy(trainX, trainY)
            epoch_training_accuracy.append(accuracy)
            accuracy = self.accuracy(validationX, validationY)
            epoch_validation_accuracy.append(accuracy)
            if not epoch % 50:
                print(f"epoch {epoch}: accuracy is {accuracy}")
        return epoch_loss, epoch_training_accuracy, epoch_validation_accuracy

    #activation functions
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def d_sigmoid(self, s):
        return self.sigmoid(s)*(1-self.sigmoid(s))

    def tanh(self, s):
        return (np.exp(s) - np.exp(-s))/(np.exp(s) + np.exp(-s))

    def d_tanh(self, s):
        return 1 - self.tanh(s) * self.tanh(s)
    
    def relu(self, s):
        return np.maximum(0.0, s)

    def d_relu(self, s):
        return (s > 0) * 1

    #loss functions
    def mse(self, y, y_pred):
        return (1/y.shape[0])*np.sum(np.square(y_pred -y))
   
    def d_mse(self, y, y_pred):
        return (y_pred - y)

    def ce(self, y, y_pred):
        return -np.sum(xlogy(y, y_pred) + xlogy(1 - y, 1 - y_pred))/len(y)

    def d_ce(self, y, y_pred):
        return ((1-y)/(1-y_pred) - (y/y_pred))

    #prediction and accuracy
    def predict(self, X):
        y_pred = self.feedforward(X)
        return np.where(y_pred<=0.5, 0, 1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y==y_pred)*100

def plot_graph(results, legend, legend_loc, xlabel, ylabel):
    for r in results:
        plt.plot(r, linewidth=1)
    plt.legend(legend, loc=legend_loc)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.show()

def visualize_dataset(data, label):
    reds = label == 0
    blues = label == 1
    plt.scatter(data[reds, 0], data[reds, 1], c="red", s=20, edgecolor='k')
    plt.scatter(data[blues, 0], data[blues, 1], c="blue", s=20, edgecolor='k')
    plt.show() 

def main():   
    data, label = ds.make_circles(n_samples=5000, factor=.7, noise=0.05)
    
    #visualize the dataset
    visualize_dataset(data, label)

    #data preprocessing
    label = label.reshape(label.shape[0], 1)

    #shuffle data
    N = len(label)
    ind_list = [i for i in range(N)]
    random.shuffle(ind_list)
    data  = data[ind_list, :]
    label = label[ind_list]

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
   
    model = Neural_Network(2, 5)
    acc = model.accuracy(X_test, y_test)
    print("X_test acc before training: ", acc)
    epoch_loss, eta, eva = model.train(X_train, y_train, 50, 2000, 0.001, X_val, y_val)
    
    #plot loss and accuracy graphs
    plot_graph([epoch_loss],["training loss"],"upper right", "epoch", "loss")
    plot_graph([eta,eva] ,["training acc", "validation accuracy"],"upper left", "epoch", "accuracy")

    acc = model.accuracy(X_test, y_test)
    print("X_test acc after training: ", acc)

if __name__ == '__main__':
    main()