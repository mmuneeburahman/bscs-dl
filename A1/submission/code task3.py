# Include libraries which may use in implementation
import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
from matplotlib import image as img
from scipy.special import xlogy
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

# Create a Neural_Network class
class Neural_Network(object): 
    def __init__(self, layers_size=None, load_model=None):
        self.no_of_layers = None
        self.layers_size = None
        self.weights = None
        if load_model:
            m = self.load_model(load_model)
            self.no_of_layers = m['no_of_layers']
            self.layers_size = m['layers_size']
            self.weights = m['weights']
            print("loaded model: ", self.layers_size)
            # print("self.no_of_layers: ", self.no_of_layers)
            # print("self.layers_size: ", self.layers_size)
            # print("self.weights: ", self.weights)
            
        else:
            self.no_of_layers = len(layers_size)
            self.layers_size = layers_size
            self.weights = [np.random.random_sample((l+1, l_)) for l, l_ in zip(layers_size[:-1], layers_size[1:])]
    
    
    def feedforward(self, X):
        inp = X
        inp = np.append(np.ones((inp.shape[0], 1)), inp, axis=1)
        self.linear_outputs = []
        self.activation_outputs = []
        for w in range(self.no_of_layers-1):
            z = np.dot(inp, self.weights[w])
            self.linear_outputs.append(z)
            if w != self.no_of_layers-2:
                z = np.append(np.ones((z.shape[0], 1)), z, axis=1)
            a = None
            if w == self.no_of_layers-2:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z)
            self.activation_outputs.append(a)
            inp = a
        return self.activation_outputs[-1]

    def backwardpropagate(self,X, Y, y_pred, lr):
        losses = []
        deltaWeights = []
        #losses
        for i in reversed(range(1, self.no_of_layers)):
            if i == self.no_of_layers-1:
                output_loss = y_pred - Y #self.d_ce(Y, y_pred)
                dadz = self.d_softmax(self.activation_outputs[i-1]) #without activation output
                loss = dadz*output_loss
            else:
                temp = np.matmul(losses[0], self.weights[i].T)[:, 1:]
                loss = temp*self.d_sigmoid(self.linear_outputs[i-1])
            losses.insert(0, loss) 
        
        #delta weights
        for i in range(0, self.no_of_layers-1):
            x = None
            if i == 0:
                x = np.append(np.ones((X.shape[0], 1)), X, axis=1)
            else:
                x = self.activation_outputs[i-1]
            dw = np.dot(x.T, losses[i])
            deltaWeights.append(dw)
        
        #updating weights
        for i in range(0, self.no_of_layers-1):
            self.weights[i] = self.weights[i]-(lr*deltaWeights[i])

    def train(self, trainX, trainY, batch_size, epochs = 100, learningRate = 0.001, validationX = None, validationY = None):
        n_examples = trainX.shape[0]
        n_batches = n_examples//batch_size
        print("train")
        epoch_loss = []
        epoch_training_accuracy = []
        epoch_validation_accuracy = []
        for epoch in range(epochs):
            if epoch != 0 and epoch%200 ==0:
                # learningRate/=2
                print("learning rate: ", learningRate)
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
            val_accuracy = self.accuracy(validationX, validationY)
            epoch_validation_accuracy.append(val_accuracy)
            # if not epoch % 50:
            print(f"epoch {epoch}: accuracy is {accuracy}  val_acc: {val_accuracy}")
        return epoch_loss, epoch_training_accuracy, epoch_validation_accuracy

    #activation functions
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def d_sigmoid(self, s):
        return self.sigmoid(s)*(1-self.sigmoid(s))
        
    def tanh(self, s):
        return 0

    def d_tanh(self, s):
        return 0
    
    def relu(self, s):
        return np.maximum(0.0, s)

    def d_relu(self, s):
        return (s > 0) * 1
    
    def softmax(self, s):
        return np.exp(s)/(np.sum(np.exp(s), axis=1)).reshape(-1, 1)

    def d_softmax(self, s):
        return s*(np.ones(s.shape)-s)
    
    #loss functions
    def ce(self, y, y_pred):
        return -np.sum(xlogy(y, y_pred) + xlogy(1 - y, 1 - y_pred))/len(y)

    def d_ce(self, y, y_pred):
        return ((1-y)/(1-y_pred) - (y/y_pred))
    
    #save and load model
    def save_model(self, path):
        data = {
            'no_of_layers': self.no_of_layers,
            'layers_size': self.layers_size,
            'weights': self.weights
        }
        with open(path, 'wb') as fp:
            pickle.dump(data, fp)

    def load_model(self, path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)
        # weights = pickle.load(model)
        # return weights

    #prediction and accuracy
    def predict(self, X):
        return self.feedforward(X)
    
    def accuracy(self, X, y):
        pred = self.predict(X)
        pred = np.argmax(pred, axis=1)
        y = np.argmax(y, axis=1)
        return np.sum( pred == y ) / len(y)*100
    
    def confusion_matrix(self, y, y_pred):
        fig, ax = plt.subplots(figsize=(15, 5))
        cm = confusion_matrix(y, y_pred, normalize="true")
        ax = sns.heatmap(cm, annot=True, cmap='Blues')
        plt.show()

    #visulize results
    def plot_tsne(self, X, y):
        ans = self.feedforward(X)
        activations = self.activation_outputs.copy()
        for a in activations:
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_result = tsne.fit_transform(a)
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 8)
            hue_labels = np.argmax(y, axis=1)
            sns.scatterplot(tsne_result[:,0], tsne_result[:,1], hue=hue_labels, legend='full')
            plt.show()

def loadDataset(path):
    print('Loading Dataset...')
    train_x, train_y, test_x, test_y = [], [], [], []
    train_count = 0
    test_count = 0
    for i in range(10):
        train_count = 1
        for filename in glob.glob(f"{path}/train/{str(i)}/*.png"):
            if train_count:
                im=img.imread(filename)
                im = im.flatten() #image flattened and normalized
                train_x.append(im)
                train_y.append(i)
                train_count+=1
            else:
                break
    for i in range(10):
        test_count = 1
        for filename in glob.glob(f"{path}/test/{str(i)}/*.png"):
            if test_count:
                im=img.imread(filename)
                im = im.flatten() #image flattened and normalized
                test_x.append(im)
                test_y.append(i)
                test_count+=1
            else:
                break
    print('Dataset loaded...')
    return np.array(train_x), np.array(train_y), np.array(test_x),np.array(test_y)
def mean_subtraction(X, y):
    #Mean Subtraction
    mean = np.mean(X, axis=0)
    mean = mean.reshape(1, -1)
    X -=mean
    print("mean: ", mean.shape)
    myimg = mean.copy().reshape(28, 28)
    print("myimg.shape: ", myimg.shape)
    plt.imshow(myimg)
    return X
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

def plot_confu(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)
    model.confusion_matrix(y, y_pred)

def main():
    train_x, train_y, test_x, test_y = loadDataset("H:\mnist")
    print("train_x.shape: ", train_x.shape)
    print("train_y.shape: ", train_y.shape)
    print("test_x.shape: ", test_x.shape)
    print("test_y.shape: ", test_y.shape)
    X, y = np.append(train_x, test_x, axis=0), np.append(train_y, test_y, axis=0)
    print("X.shape: ", X.shape)
    print("Y.shape: ", y.shape)

    print("---------mean subtractions data--------")
    X = mean_subtraction(X, y)
    #shuffle the data
    print("---------shuffle data--------")
    X, y = shuffle(X, y)

    #split train and test
    print("---------splitting data--------")
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=42)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.10, random_state=42)

    #reshape the train_y and test_y to column vectorsz
    train_y = train_y.reshape(train_y.shape[0], 1)
    val_y = val_y.reshape(val_y.shape[0], 1)
    test_y = test_y.reshape(test_y.shape[0], 1)
    print("train_x.shape: ", train_x.shape)
    print("train_y.shape: ", train_y.shape)
    print("val_x.shape: ", val_x.shape)
    print("val_y.shape: ", val_y.shape)
    print("test_x.shape: ", test_x.shape)
    print("test_y.shape: ", test_y.shape)

    #one hot encoding
    print("-----------One Hot encoding---------")
    onehot_encoder = OneHotEncoder(sparse=False)
    train_y = onehot_encoder.fit_transform(train_y)
    test_y = onehot_encoder.fit_transform(test_y)
    val_y = onehot_encoder.fit_transform(val_y)
    print("train_y_encoded.shape: ", train_y.shape)
    print("test_y_encoded.shape: ", test_y.shape)
    print("val_y_encoded.shape: ", val_y.shape)

    #show example
    print("-----------Show Example---------")
    im = train_x[50].copy()
    im = im.reshape(28, 28)
    print('im.shape: ', im.shape)
    plt.imshow(im)
    print(train_y[50])

    #creating, training and saving model
    print("-----------training and saving model---------")
    model = Neural_Network([784, 128, 64, 10])
    acc = model.accuracy(test_x, test_y) 
    print("acc: ", acc)

    epoch_loss, epoch_training_accuracy, eva = model.train(train_x, train_y, 50, 150, 0.01, val_x, val_y)
    plot_graph([epoch_loss],["training loss"],"upper right", "epoch", "loss")
    plot_graph([epoch_training_accuracy,eva] ,["training acc", "validation acc"],"upper left", "epoch", "accuracy")
    print("max accuracy: ", max(epoch_training_accuracy))
    acc = model.accuracy(test_x, test_y)
    print("X_test acc after training: ", acc)

    model.save_model("model.pkl")

    #plotting confusion matrixs
    print("-----------plot confusion matrixs---------")
    plot_confu(model, train_x, train_y)
    plot_confu(model, test_x, test_y)
    plot_confu(model, val_x, val_y)

    #t-sne plots
    print("-----------t-sne plots---------")
    # print("------------------train---------------")
    model.plot_tsne(train_x, train_y)
    # print("------------------test---------------")
    model.plot_tsne(test_x, test_y)
    # print("------------------validation---------------")
    model.plot_tsne(val_x, val_y)

if __name__ == '__main__':
    main()