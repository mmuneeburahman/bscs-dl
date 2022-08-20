# -*- coding: utf-8 -*-
"""dl_a3_Colab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W08WAHOSBKnhD6RfPzRbNdgr3vTPjk-_
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !unzip /content/drive/MyDrive/dl-a3/test.zip
# !unzip /content/drive/MyDrive/dl-a3/train.zip

import os
import torch
import time
import numpy as np
import pandas as pd
import imageio as io
from torch import nn
from torch import optim
from torchvision import utils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class MNIST(Dataset):
    def __init__(self, csv_file_dir, data_dir, transform=None):
        data = pd.read_csv(csv_file_dir, skiprows=1)
        self.X = data.iloc[:, 0]
        self.y = data.iloc[:, 1]
        self.shape = len(self.X)
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.X[index])
        image = io.imread(img_path)
        y_label = self.y[index]
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)

    def __len__(self):
        return self.shape

def load_data(train_csv, train_content, test_csv, test_content, train_data_amount = 1, val_ratio=0.2):
    all_training_data = MNIST(csv_file_dir=train_csv, data_dir=train_content,
        transform=transforms.ToTensor())
    size = len(all_training_data)
    train_size = int(size*train_data_amount)
    remain_size = size-train_size
    train_dataset, remaining_dataset = random_split(all_training_data, [train_size, remain_size])
    test_dataset = MNIST(csv_file_dir=test_csv, data_dir=test_content,
        transform=transforms.ToTensor())

    size = len(train_dataset)
    val_size = int(size*val_ratio)
    train_size = size-val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    return train_dataset, val_dataset, test_dataset

def dataset_check(dataset, index):
    img = dataset[100][0].transpose(0, 1).transpose(1, 2)
    plt.imshow(torch.squeeze(img))
    plt.show()
    print(f"class is {dataset[100][1]}")

def get_loaders(train_dataset, train_BS, val_dataset, val_BS, test_dataset, test_BS):
    train_loader = DataLoader(train_dataset, batch_size=train_BS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_BS, shuffle=True)
    return train_loader, val_loader, test_loader

def get_distribution(dataloader):
    dis = torch.zeros(10).to(device)
    for batch in dataloader:
        dis+=torch.bincount(batch[1].to(device)).to(device)
    return dis

def plot(distribution, title):
    plt.bar(torch.arange(0,10), distribution)
    plt.xlabel("numbers")
    plt.ylabel("frequency")
    plt.title(title)
    plt.show()

class ConvBlock(nn.Module):
    '''Simple Convolution Block'''
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size=3, 
            stride=stride
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ConvDw(nn.Module):
    '''Mobile Net Block'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvDw, self).__init__()
        '''Depth wise convolution'''
        self.dwc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, 
                    stride=stride, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        '''Point wise convolution'''
        self.pc = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
        stride = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.dwc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pc(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class A1(nn.Module):
    def __init__(self, ):
        super(A1, self).__init__()
        self.conv = ConvBlock(1, 32, 2)
        self.mobnet1 = ConvDw(32, 32)
        self.mobnet2 = ConvDw(32, 64)
        self.mobnet3 = ConvDw(64, 128)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        out = self.conv(x)
        out = self.mobnet1(out)
        out = self.mobnet2(out)
        out = self.mobnet3(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out, dim=1)

class A1_Moduler(nn.Module):
    def __init__(self, conv_parameters, mobnet_parameters, num_classes):
        super(A1_Moduler, self).__init__()
        self.convs = nn.Sequential()
        self.mobnets = nn.Sequential()
        for i, parameter in enumerate(conv_parameters):
            self.convs.add_module("conv"+str(i), ConvBlock(*parameter))
        for i, parameter in enumerate(mobnet_parameters):
            self.mobnets.add_module("mobnet"+str(i), ConvDw(*parameter))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(mobnet_parameters[-1][1], num_classes)

    def forward(self, x):
        out = x
        for layer in self.convs:
            out = layer(out)
        for layer in self.mobnets:
            out = layer(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out, dim=1)

class A2(nn.Module):
    def __init__(self):
        super(A2, self).__init__()
        self.conv1 = ConvBlock(1, 32, 2) #500x1x28x28
        self.conv2 = ConvBlock(32, 64, 1) #
        self.conv3 = ConvBlock(64, 128, 1)
        self.conv4 = ConvBlock(128, 256, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out= self.fc(out)
        return F.softmax(out, dim=1) #be careful here

def init_network(conv_block_count, mobilenet_block_count, input_dim, num_classes):
    convs_layers = []
    mobilenets_layers =[]
    inp, out = input_dim, 32
    for _ in range(conv_block_count):
        convs_layers.append((inp, out, 2))
        inp = out
        out*=2
    out = 32
    for _ in range(mobilenet_block_count):
        mobilenets_layers.append((inp, out))
        inp = out
        out*=2
    return A1_Moduler(convs_layers, mobilenets_layers, num_classes)

def plot_graph(results, legend, legend_loc, xlabel, ylabel):
    for r in results:
        plt.plot(r, linewidth=1)
    plt.legend(legend, loc=legend_loc)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.show()

def train(net, train_loader, val_loader, training_epochs, loss_func, optimizer):
    training_loss = []
    validation_loss = []
    for epoch in range(training_epochs):
        net.train()
        epoch_loss = 0
        count = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            count = i
        epoch_loss/=count
        print(f"{epoch}th training epoch_loss: {epoch_loss}")
        training_loss.append(epoch_loss)

        net.eval()
        epoch_loss = 0
        count = 0
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = loss_func(output, labels)
            epoch_loss+=loss.item()
            count = i
        epoch_loss/=count
        print(f"{epoch}th validation epoch_loss: {epoch_loss}")
        validation_loss.append(epoch_loss)
    plot_graph([training_loss,validation_loss] ,["training loss", "validation loss"],"upper left", "epoch", "loss")

    return net

def save_model(net, path):
    torch.save(net.state_dict(), path)

def load_model(net, path):
    print("loading model...")
    state_dict = torch.load(path, map_location='cpu')
    net.load_state_dict(state_dict)

def test(net, test_loader):
    correct = 0
    total = 0
    output = None
    for images, labels in test_loader:
        out = net(images)
        if output == None:
            output = out
        else:
            output = torch.cat((output, out), dim=0)
        c = (torch.argmax(out, dim=1) == labels).sum()
        correct+=c
        total+=labels.shape[0]
    print(f"test dataset accuracy is {(correct/total)*100}")
    return output

import seaborn as sns
def plot_confu(model, X, y):
    y_pred = model(X)
    y_pred = torch.argmax(y_pred, axis=1)
    fig, ax = plt.subplots(figsize=(15, 5))
    cm = confusion_matrix(y, y_pred, normalize="true")
    ax = sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()

def show_true_pred(images, labels, y_pred):
    done = torch.zeros(10)
    count = 0
    i =0
    fig, axes = plt.subplots(1, 4,figsize=(12,4))
    fig.suptitle('True Predictions')
    while count<4:
        if y_pred[i] == labels[i] and done[labels[i]] == 0:
            done[labels[i]] = 1
            img = images[i]
            axes[count].imshow(torch.squeeze(img))
            axes[count].set_title(f"label: {labels[i]}")
            count+=1
        i+=1
    plt.show()
    done = torch.zeros(10)

def show_false_pred(images, labels, y_pred):
    done = torch.zeros(10)
    count = 0
    i =0
    fig, axes = plt.subplots(1, 4,figsize=(12,4))
    fig.suptitle('Wrong Predictions')
    while count<4:
        if y_pred[i] != labels[i] and done[labels[i]] == 0:
            done[labels[i]] = 1
            img = images[i]
            axes[count].imshow(torch.squeeze(img))
            axes[count].set_title(f"Predicted: {y_pred[i]}")
            count+=1
        i+=1
    plt.show()

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    '''https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch'''
    '''This fucntion is taken from this link'''
    '''It is used to display the filter learned by any layer'''
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()

from sklearn.metrics import roc_curve, auc
def plot_rocs(labels, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_pred, labels, pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(10):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC of {i} Class')
        plt.legend(loc="lower right")
        plt.show()

from sklearn.manifold import TSNE
def plot_tsne(labels, y_pred):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_result = tsne.fit_transform(y_pred)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    hue_labels = labels
    sns.scatterplot(tsne_result[:,0], tsne_result[:,1], hue=hue_labels, legend='full')
    plt.show()

#Hyperparamters
TRAIN_CSV = "H:/datasets/A3/train/train.csv"
TRAIN_DATA = "H:/datasets/A3/train/train_new"
TEST_CSV = "H:/datasets/A3/test/test.csv"
TEST_DATA = "H:/datasets/A3/test/test_new"
TRAIN_DATA_AMOUNT = 0.6

LOSS_FN = nn.CrossEntropyLoss()
TRAIN_BS = 1500
LR = 0.01
TRAINING_EPOCHS = 10

SAVE_MODEL_PATH = os.getcwd()
SAVE_MODEL_NAME = "/c1m3.pt"
LOAD_MODEL_PATH = os.getcwd()
LOAD_MODEL_NAME = "/c1m3.pt"

IS_TRAINING = False
SAVE_NETWORK = False
PLOT_DISTRIBUTIONS = False
PRINT_SUMMARY = False
CONFUSION_MATRIX = False
PRF1 = False
TRUE_FALSE_PREDICTIONS = False
SHOW_LAST_LAYER_PARAMETERS = False
ROCS_PLOTS = False
TSNE_PLOT = False

def main(network_type, is_training=True, save_network=True, plot_distributions=True, print_summary=True,
         confusion_matrix=True, PRF1=True, true_false_predictions=True, 
         show_last_layer_parameters=True, ROCs_plots=True, tsne_plot=True):
    print(f"device available is :{device}")
    train_dataset, val_dataset, test_dataset = load_data(TRAIN_CSV, TRAIN_DATA, TEST_CSV, TEST_DATA, train_data_amount=0.6)
    dataset_check(train_dataset, 100)
    train_loader, val_loader, test_loader = get_loaders(train_dataset, TRAIN_BS, val_dataset, 1000, test_dataset, 10000)
    if plot_distributions:
        train_distribution = get_distribution(train_loader).to("cpu")
        val_distribution = get_distribution(val_loader).to("cpu")
        test_distribution = get_distribution(test_loader).to("cpu")
        plot(train_distribution, "Training Data")
        plot(val_distribution, "Validation Data")
        plot(test_distribution, "Testing Data")
    torch.cuda.empty_cache()
    if network_type == "mobilenet":
        net = init_network(conv_block_count=1, mobilenet_block_count=3, input_dim=1, num_classes=10)
        net = net.to(device)
    elif network_type == "conv":
        net = A2()
        net = net.to(device)
    if print_summary:
        summary(net, (1, 28, 28))
    if is_training:
        training_epochs = TRAINING_EPOCHS
        loss_func = LOSS_FN
        optimizer = optim.Adam(net.parameters(), lr = LR)
        train(net, train_loader, val_loader, training_epochs, loss_func, optimizer)
        net.to(device="cpu")
    else:
        if network_type == "conv":
            load_model(net, LOAD_MODEL_PATH+LOAD_MODEL_NAME)
        elif network_type == "mobilenet":
            load_model(net, LOAD_MODEL_PATH+LOAD_MODEL_NAME)
    if save_model:
        save_model(net, SAVE_MODEL_PATH + SAVE_MODEL_NAME )
    net.to(device="cpu")
    output = test(net, test_loader)
    print(f"output size {output.shape}")
    # check here if len(dataset) == len(dataloader)
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    out = net(images)
    y_pred = torch.argmax(out, axis=1)
    if confusion_matrix:
        plot_confu(net, images, labels)
    if PRF1:
        cr = classification_report(labels, y_pred, digits=3)
        print(cr)
    if true_false_predictions:
        show_true_pred(images, labels, y_pred)
        show_false_pred(images, labels, y_pred)
    if show_last_layer_parameters:
        if network_type == "conv":
            filter = net.conv4.conv.weight.data.clone()
            print(filter.shape)
            visTensor(filter, ch=0, allkernels=False)
        elif network_type == "mobilenet":
            filter = net.mobnets[2].dwc1.weight.data.clone()
            print(filter.shape)
            visTensor(filter, ch=0, allkernels=False)
            filter = net.mobnets[2].pc.weight.data.clone()
            print(filter.shape)
            visTensor(filter, ch=0, allkernels=False)
    if ROCs_plots:
        plot_rocs(labels, y_pred)
    if tsne_plot:
        plot_tsne(labels, out.detach().numpy())

main("mobilenet",
     is_training=IS_TRAINING,
     save_network = SAVE_NETWORK,
     plot_distributions=PLOT_DISTRIBUTIONS, 
     print_summary=PRINT_SUMMARY,
     confusion_matrix=CONFUSION_MATRIX, 
     PRF1=PRF1, 
     true_false_predictions=TRUE_FALSE_PREDICTIONS, 
     show_last_layer_parameters=SHOW_LAST_LAYER_PARAMETERS, 
     ROCs_plots=ROCS_PLOTS,
     tsne_plot=TSNE_PLOT
     )

main("mobilenet",
     is_training=IS_TRAINING,
     save_network = SAVE_NETWORK,
     plot_distributions=PLOT_DISTRIBUTIONS, 
     print_summary=PRINT_SUMMARY,
     confusion_matrix=CONFUSION_MATRIX, 
     PRF1=PRF1, 
     true_false_predictions=TRUE_FALSE_PREDICTIONS, 
     show_last_layer_parameters=SHOW_LAST_LAYER_PARAMETERS, 
     ROCs_plots=ROCS_PLOTS,
     tsne_plot=TSNE_PLOT
     )

