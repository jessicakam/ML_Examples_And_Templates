# 2017/07/29

from data_preprocessing import DataPreProcessing
from data_postprocessing import DataPostProcessing
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
class SOM(DataPreProcessing, DataPostProcessing):
    
    def scaleFeatures(self):
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.X = self.sc.fit_transform(self.X)
                
    def train(self, **kwargs):
        self.som = MiniSom(**kwargs)
        self.som.random_weights_init(self.X)
        self.som.train_random(data = self.X, num_iteration = 100)
    
    def visualizeResults(self):
        bone()
        pcolor(self.som.distance_map().T)
        colorbar()
        markers = ['o', 's']
        colors = ['r', 'g']
        for i, x in enumerate(self.X):
            w = self.som.winner(x)
            plot(w[0] + 0.5,
                 w[1] + 0.5,
                 markers[self.y[i]],
                 markeredgecolor = colors[self.y[i]],
                 markerfacecolor = 'None',
                 markersize = 10,
                 markeredgewidth = 2)
        show()
        
    def findFrauds(self):
        mappings = self.som.win_map(self.X)
        self.frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
        self.frauds = self.sc.inverse_transform(self.frauds)

        
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
class ReducedBoltzmannMachines(DataPreProcessing, DataPostProcessing):
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
            
    def convertIntoBinary(self):
        # override
        #1==Liked, 0==Not Liked
        self.training_set[self.training_set == 0] = -1
        self.training_set[self.training_set == 1] = 0
        self.training_set[self.training_set == 2] = 0
        self.training_set[self.training_set >= 3] = 1
        self.test_set[self.test_set == 0] = -1
        self.test_set[self.test_set == 1] = 0
        self.test_set[self.test_set == 2] = 0
        self.test_set[self.test_set >= 3] = 1
        
    def createNN(self, nv, nh):
        self.nv = nv
        self.nh = nh
        self.batch_size = 100
        self.rbm = self.RBM(nv=self.nv, nh=self.nh)
        self.W = torch.randn(self.nh, self.nv)
        self.a = torch.randn(1, self.nh)
        self.b = torch.randn(1, self.nv)

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
            super(self.SAE, self).__init__()
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
        self.sae = self.SAE()
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
        
