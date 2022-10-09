import numpy as np
import torch
from torchvision import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow import keras

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler,random_data=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.random_data = random_data
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        # X_train_sample_class_1 = self.Y_train[self.Y_train == 1][:10]
        # X_train_sample_class_0 = self.Y_train[self.Y_train == 0][:10]
        class_1 = [i for i, x in enumerate(self.Y_train == 1) if x]
        class_0 = [i for i, x in enumerate(self.Y_train == 0) if x]
        half_num = int(num/2)
        np.random.shuffle(class_1)
        np.random.shuffle(class_0)
        tmp_idxs = class_1[:half_num] + class_0[:half_num]
        # print(class_0)
        # tmp_idxs = np.arange(self.n_pool)
        # np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        # print(preds)
        # print(self.Y_test)
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)


def get_HeadAndNeck(handler):
    # Load the dataset
    # dataset = pd.read_csv('dataset2.csv',header=None)
    dataset = pd.read_csv('new_dataset_normalized.csv',header=None)
    random_data = pd.read_csv('random_data.csv',header=None)

    X_data = dataset.iloc[:,:-1]
    y_data = dataset.iloc[:,-1:]

    # Train and test set
    # trainX, trainY, testX, testY = train_test_split(X_data, y_data, stratify=y_data,test_size=0.3,random_state=1)
    test_size = int(np.floor(0.30*X_data.shape[0]) )
    trainX, testX = X_data[:-test_size], X_data[-test_size:]
    trainY, testY = y_data[:-test_size], y_data[-test_size:]

    X_trainTensor = torch.tensor(trainX.values).float()
    y_trainTensor = torch.LongTensor(trainY.values).squeeze(1)
    X_testTensor = torch.tensor(testX.values).float()
    y_testTensor = torch.LongTensor(testY.values).squeeze(1)
    random_data = torch.tensor(random_data.values).float()
    return Data(X_trainTensor, y_trainTensor, X_testTensor, y_testTensor, handler,random_data)
