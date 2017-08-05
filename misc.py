# 2017/07/29

from data_preprocessing import DataPreProcessing
from data_postprocessing import DataPostProcessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
class DimensionalityReduction(DataPreProcessing, DataPostProcessing):
    
    def __init__(self):
        self.name = ''
        self.abbrev = ''
                
    def fitLogisticRegToDataset(self, **kwargs):
        self.classifier = LogisticRegression(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)
                
    def generateTitle(self, what_visualizing):
        return 'Logistic Regression' + ' ' + '(' + what_visualizing + ')'
    
    def visualizeResults(self, what_visualizing, tuple_colors, xlabel, ylabel):
        X_set, y_set = self.X_train, self.y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(tuple_colors))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(tuple_colors)(i), label = j)
        plt.title(self.generateTitle(what_visualizing))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        
    def visualizeTrainingSetResults(self, tuple_colors, xlabel, ylabel):
        self.visualizeResults('Training Set', tuple_colors, xlabel, ylabel)
        
    def visualizeTestSetResults(self, tuple_colors, xlabel, ylabel):
        self.visualizeResults('Test Set', tuple_colors, xlabel, ylabel)

        
from sklearn.decomposition import PCA
class PCA(DimensionalityReduction):
    
    def __init__(self):
        self.name = 'PCA'
        
    def applyPCA(self, **kwargs):
        pca = PCA(**kwargs)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        self.explained_variance = pca.explained_variance_ratio_
        

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
class LDA(DimensionalityReduction):
    
    def __init__(self):
        self.name = 'LDA'
        
    def applyLDA(self, **kwargs):
        self.lda = LDA(**kwargs)
        self.X_train = self.lda.fit_transform(self.X_train, self.y_train)
        self.X_test = self.lda.transform(self.X_test)

        
from sklearn.decomposition import KernelPCA
class KernelPCA(PCA):
    
    def __init__(self):
        self.name = 'KernelPCA'
    
    def applyKernelPCA(self, **kwargs):
        self.kpca = KernelPCA(**kwargs)
        self.X_train = self.kpca.fit_transform(self.X_train)
        self.X_test = self.kpca.transform(self.X_test)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
class ModelSelection(DataPreProcessing, DataPostProcessing):
    
    def fitToTrainingSet(self, **kwargs):
        self.classifier = SVC(**kwargs)
        self.classifier.fit(self.X_train, self.y_train)
    
    def applyKFoldCrossValidation(self, **kwargs):
        self.accuracies = cross_val_score(estimator=self.classifier, X=self.X_train, y=self.y_train, **kwargs)
        self.mean = self.accuracies.mean()
        self.std = self.accuracies.std()
        
    def applyGridSearchToFindBestModels(self):
        parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 10,
                                   n_jobs = -1)
        grid_search = grid_search.fit(X_train, y_train)
        self.best_accuracy = grid_search.best_score_
        self.best_parameters = grid_search.best_params_
        
    def visualizeTrainingSetResults(self, title, xlabel, ylabel):
        self.visualizeResults(self.X_train, self.y_train, title, xlabel, ylabel)
    
    def visualizeTestSetResults(self, title, xlabel, ylabel):
        self.visualizeResults(self.X_test, self.y_test, title, xlabel, ylabel)

    def visualizeResults(self, X_visualizing, y_visualizing, title, xlabel, ylabel):
        X_set, y_set = X_visualizing, y_visualizing
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

        
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
class XGBoost(DataPreProcessing, DataPostProcessing):
    
    def fitToTrainingSet(self, **kwargs):
        #
        self.classifier = XGBClassifier()
        self.classifier.fit(self.X_train, self.y_train)
    
    def applyKFoldCrossValidation(self, **kwargs):
        self.accuracies = cross_val_score(**kwargs)
        self.mean = self.accuracies.mean()
        self.std = self.accuracies.std()


from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom ##add this file to misc
from pylab import bone, pcolor, colorbar, plot, show
class SOM(DataPreProcessing, DataPostProcessing):
    
    def scaleFeatures(self):
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.X = self.sc.fit_transform(X)
                
    def train(self, **kwargs):
        self.som = MiniSom(**kwargs)
        self.som.random_weights_init(X)
        self.som.train_random(data = X, num_iteration = 100)
    
    def visualizeResults():
        bone()
        pcolor(self.som.distance_map().T)
        colorbar()
        markers = ['o', 's']
        colors = ['r', 'g']
        for i, x in enumerate(self.X):
            w = som.winner(x)
            plot(w[0] + 0.5,
                 w[1] + 0.5,
                 markers[y[i]],
                 markeredgecolor = colors[y[i]],
                 markerfacecolor = 'None',
                 markersize = 10,
                 markeredgewidth = 2)
        show()
        
    def findFrauds(self):
        mappings = som.win_map(self.X)
        self.frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
        self.frauds = self.sc.inverse_transform(self.frauds)

        
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
class ReducedBoltzmannMachines(DataPreProcessing, DataPostProcessing):
    
    def convertIntoBinary(self):
        self.training_set[self.training_set == 0] = -1
        self.training_set[self.training_set == 1] = 0
        self.training_set[self.training_set == 2] = 0
        self.training_set[self.training_set >= 3] = 1
        self.test_set[self.test_set == 0] = -1
        self.test_set[self.test_set == 1] = 0
        self.test_set[self.test_set == 2] = 0
        self.test_set[self.test_set >= 3] = 1
        
    def createNN(self, nv, nh):
        self.batch_size = 100
        self.rbm = RBM(self.nv, self.nh)
        
        self.W = torch.randn(self.nh, self.nv)
        self.a = torch.randn(1, self.nh)
        self.b = torch.randn(1, self.nv)
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
        
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
        
    def trainRBM(self, v0, vk, ph0, phk):
        # Helper for training the RBM
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        """
        class RBM():
            def __init__(self, nv, nh):
                self.W = torch.randn(nh, nv)
                self.a = torch.randn(1, nh)
                self.b = torch.randn(1, nv)
            def sample_h(self, x):
                wx = torch.mm(x, self.W.t())
                activation = wx + self.a.expand_as(wx)
                p_h_given_v = torch.sigmoid(activation)
                return p_h_given_v, torch.bernoulli(p_h_given_v)
            def sample_v(self, y):
                wy = torch.mm(y, self.W)
                activation = wy + self.b.expand_as(wy)
                p_v_given_h = torch.sigmoid(activation)
                return p_v_given_h, torch.bernoulli(p_v_given_h)
            def train(self, v0, vk, ph0, phk):
                self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
                self.b += torch.sum((v0 - vk), 0)
                self.a += torch.sum((ph0 - phk), 0)
        nv = len(training_set[0])
        nh = 100
        batch_size = 100
        rbm = RBM(nv, nh)
        """

    def train(self, nb_epoch):
        # Training the RBM
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            for id_user in range(0, self.nb_users - self.batch_size, self.batch_size):
                vk = self.training_set[id_user:id_user+self.batch_size]
                v0 = self.training_set[id_user:id_user+self.batch_size]
                ph0,_ = self.rbm.sample_h(v0)
                for k in range(10):
                    _,hk = self.rbm.sample_h(vk)
                    _,vk = self.rbm.sample_v(hk)
                    vk[v0<0] = v0[v0<0]
                phk,_ = self.rbm.sample_h(vk)
                self.rbm.trainRBM(v0, vk, ph0, phk)
                train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
                s += 1.
            print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

            
    def test(self):
        test_loss = 0
        s = 0.
        for id_user in range(self.nb_users):
            v = self.training_set[id_user:id_user+1]
            vt = self.test_set[id_user:id_user+1]
            if len(vt[vt>=0]) > 0:
                _,h = self.rbm.sample_h(v)
                _,v = self.rbm.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
                s += 1.
        print('test loss: '+str(test_loss/s))

        
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
class AutoEncoder(DataPreProcessing, DataPostProcessing):
    class SAE(nn.Module):
        def __init__(self, ):
            super(SAE, self).__init__()
            self.fc1 = nn.Linear(self.nb_movies, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 20)
            self.fc4 = nn.Linear(20, self.nb_movies)
            self.activation = nn.Sigmoid()
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.fc3(x))
            x = self.fc4(x)
            return x
     
    def createNN(self):
        self.sae = SAE()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.sae.parameters(), lr = 0.01, weight_decay = 0.5)
        """
        # Creating the architecture of the Neural Network
        class SAE(nn.Module):
            def __init__(self, ):
                super(SAE, self).__init__()
                self.fc1 = nn.Linear(nb_movies, 20)
                self.fc2 = nn.Linear(20, 10)
                self.fc3 = nn.Linear(10, 20)
                self.fc4 = nn.Linear(20, nb_movies)
                self.activation = nn.Sigmoid()
            def forward(self, x):
                x = self.activation(self.fc1(x))
                x = self.activation(self.fc2(x))
                x = self.activation(self.fc3(x))
                x = self.fc4(x)
                return x
            self.sae = SAE()
            self.criterion = nn.MSELoss()
            self.optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
        """
        
    def train(self, nb_epoch):
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            for id_user in range(self.nb_users):
                input = Variable(self.training_set[id_user]).unsqueeze(0)
                target = input.clone()
                if torch.sum(target.data > 0) > 0:
                    output = self.sae(input)
                    target.require_grad = False
                    output[target == 0] = 0
                    loss = self.criterion(output, target)
                    mean_corrector = self.nb_movies/float(torch.sum(target.data > 0) + 1e-10)
                    loss.backward()
                    train_loss += np.sqrt(loss.data[0]*mean_corrector)
                    s += 1.
                    self.optimizer.step()
            print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
    def test(self):
        test_loss = 0
        s = 0.
        for id_user in range(self.nb_users):
            input = Variable(self.training_set[id_user]).unsqueeze(0)
            target = Variable(self.test_set[id_user])
            if torch.sum(target.data > 0) > 0:
                output = self.sae(input)
                target.require_grad = False
                output[target == 0] = 0
                loss = self.criterion(output, target)
                mean_corrector = self.nb_movies/float(torch.sum(target.data > 0) + 1e-10)
                test_loss += np.sqrt(loss.data[0]*mean_corrector)
                s += 1.
        print('test loss: '+str(test_loss/s))
        
