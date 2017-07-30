"""
unit9
dimensionality reduction
PCA
LDA
kernel PCA

unit10
GridSearch
KFoldCrossValidation
Model Selection
XG Boost

#Date: 2017/07/29
"""

from data_preprocessing import DataPreprocessing

class DimensionalityReduction(DataPreprocessing):
    def __init__(self):
        self.name = ''
        self.abbrev = ''
        
    def scaleFeatures(self):
        #
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
    def fitLogisticRegToDataset(self):
        #
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
        
    def predictResults(self):
        #
        y_pred = classifier.predict(X_test)
    
    def makeConfusionMatrix(self):
        #
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
    def generateTitle(self, what_visualizing):
        return 'Logistic Regression' + ' ' + '(' + what_visualizing + ')'
    
    def visualizeResults(self, what_visualizing, tuple_colors, xlabel, ylabel):
        from matplotlib.colors import ListedColormap
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
        plt.title(self.generateTitle(what_visualizing)) #'Logistic Regression (Training set)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        
    def visualizeTrainingSetResults(self, tuple_colors, xlabel, ylabel):
        self.visualizeResults(what_visualizing='Training Set', tuple_colors, xlabel, ylabel)
        
    def visualizeTestSetResults(self, tuple_colors, xlabel, ylabel):
        self.visualizeResults(what_visualizing='Test Set', tuple_colors, xlabel, ylabel)

class PCA(DimensionalityReduction):
    def __init__(self):
        self.name = 'PCA'
        
    def applyPCA(self):
        #
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        

class LDA(DimensionalityReduction):
        def __init__(self):
        self.name = 'LDA'
        
    def applyLDA(self):
        #
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = 2)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)

class KernelPCA(PCA):
    def __init__(self):
        self.name = 'KernelPCA'
    
    def applyKernelPCA(self):
        #
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(n_components = 2, kernel = 'rbf')
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)
####
class ModelSelection(DataPreprocessing):
    def scaleFeatures(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    
    def fitToTrainingSet(self, **kwargs):
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
    
    def predictResults(self):
        y_pred = self.classifier.predict(self.X_test)

    def makeConfusionMatrix(self):
        #
        self.cm = confusion_matrix(y_test, y_pred)

    def applyKFoldCrossValidation(estimator=self.classifier, X=self.X_train, y=self.y_train, cv=10):
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        accuracies.mean()
        accuracies.std()
        
    def applyGridSearchToFindBestModels(self):
        from sklearn.model_selection import GridSearchCV
        parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 10,
                                   n_jobs = -1)
        grid_search = grid_search.fit(X_train, y_train)
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        
    def visualizeTrainingSetResults(self):
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Kernel SVM (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
    
    def visualizeTestSetResults(self):
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Kernel SVM (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
class XGBoost(DataPreprocessing):
    def fitToTrainingSet(self, **kwargs):
        #
        self.classifier = XGBClassifier()
        self.classifier.fit(self.X_train, self.y_train)
    
    def predictResults(self):
        #
        y_pred = self.classifier.predict(self.X_test)
        
    def makeConfusionMatrix(self):
        #
        self.cm = confusion_matrix(y_test, y_pred)

    def applyKFoldCrossValidation(self, **kwargs):
        ##
        self.accuracies = cross_val_score(**kwargs)
        self.accuracies.mean()
        self.accuracies.std()

#####

from sklearn.preprocessing import MinMaxScaler
class SOM(DataPreprocessing):
    def scaleFeatures(self):
        sc = MinMaxScaler(feature_range = (0, 1))
        X = sc.fit_transform(X)
                
    def train(self):
        from minisom import MiniSom ##add this file to misc
        som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
        som.random_weights_init(X)
        som.train_random(data = X, num_iteration = 100)
    
    def visualizeResults():
        from pylab import bone, pcolor, colorbar, plot, show
        bone()
        pcolor(som.distance_map().T)
        colorbar()
        markers = ['o', 's']
        colors = ['r', 'g']
        for i, x in enumerate(X):
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
        mappings = som.win_map(X)
        frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
        frauds = sc.inverse_transform(frauds)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
class ReducedBoltzmannMachines(DataPreprocessing):
    def importDataset(self):            
        movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
        users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
        ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

    def prepareTrainingAndTestSets(self):
        training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
        training_set = np.array(training_set, dtype = 'int')
        test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
        test_set = np.array(test_set, dtype = 'int')
        
        # Getting the number of users and movies
        nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
        nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
    
    def convertData(self):
        # Converting the data into an array with users in lines and movies in columns
        new_data = []
        for id_users in range(1, nb_users + 1):
            id_movies = data[:,1][data[:,0] == id_users]
            id_ratings = data[:,2][data[:,0] == id_users]
            ratings = np.zeros(nb_movies)
            ratings[id_movies - 1] = id_ratings
            new_data.append(list(ratings))
        training_set = convert(training_set)
        test_set = convert(test_set)
    
    def convertIntoTensors(self):
        # Converting the data into Torch tensors
        training_set = torch.FloatTensor(training_set)
        test_set = torch.FloatTensor(test_set)

    def convertIntoBinary(self):
        training_set[training_set == 0] = -1
        training_set[training_set == 1] = 0
        training_set[training_set == 2] = 0
        training_set[training_set >= 3] = 1
        test_set[test_set == 0] = -1
        test_set[test_set == 1] = 0
        test_set[test_set == 2] = 0
        test_set[test_set >= 3] = 1
        
    def createNN(self):
        # Creating the architecture of the Neural Network
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

    def train(self):
        # Training the RBM
        nb_epoch = 10
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            for id_user in range(0, nb_users - batch_size, batch_size):
                vk = training_set[id_user:id_user+batch_size]
                v0 = training_set[id_user:id_user+batch_size]
                ph0,_ = rbm.sample_h(v0)
                for k in range(10):
                    _,hk = rbm.sample_h(vk)
                    _,vk = rbm.sample_v(hk)
                    vk[v0<0] = v0[v0<0]
                phk,_ = rbm.sample_h(vk)
                rbm.train(v0, vk, ph0, phk)
                train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
                s += 1.
            print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

            
    def test(self):
        # Testing the RBM
        test_loss = 0
        s = 0.
        for id_user in range(nb_users):
            v = training_set[id_user:id_user+1]
            vt = test_set[id_user:id_user+1]
            if len(vt[vt>=0]) > 0:
                _,h = rbm.sample_h(v)
                _,v = rbm.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
                s += 1.
        print('test loss: '+str(test_loss/s))
        
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
class AutoEncoder(DataPreprocessing):
    def importDataset(self):
        movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
        users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
        ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

    def prepareTrainingAndTestSets(self):
                
        training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
        training_set = np.array(training_set, dtype = 'int')
        test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
        test_set = np.array(test_set, dtype = 'int')
        
        # Getting the number of users and movies
        nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
        nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
    
    def convertData(self);:
        # Converting the data into an array with users in lines and movies in columns
        new_data = []
        for id_users in range(1, nb_users + 1):
            id_movies = data[:,1][data[:,0] == id_users]
            id_ratings = data[:,2][data[:,0] == id_users]
            ratings = np.zeros(nb_movies)
            ratings[id_movies - 1] = id_ratings
            new_data.append(list(ratings))
        training_set = convert(training_set)
        test_set = convert(test_set)
        
    def convertIntoTensors(self):
        training_set = torch.FloatTensor(training_set)
        test_set = torch.FloatTensor(test_set)
        
    def createNN(self):
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
            sae = SAE()
            criterion = nn.MSELoss()
            optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

        
    def train(self):
        nb_epoch = 200
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            for id_user in range(nb_users):
                input = Variable(training_set[id_user]).unsqueeze(0)
                target = input.clone()
                if torch.sum(target.data > 0) > 0:
                    output = sae(input)
                    target.require_grad = False
                    output[target == 0] = 0
                    loss = criterion(output, target)
                    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
                    loss.backward()
                    train_loss += np.sqrt(loss.data[0]*mean_corrector)
                    s += 1.
                    optimizer.step()
            print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
    def test(self):
        test_loss = 0
        s = 0.
        for id_user in range(nb_users):
            input = Variable(training_set[id_user]).unsqueeze(0)
            target = Variable(test_set[id_user])
            if torch.sum(target.data > 0) > 0:
                output = sae(input)
                target.require_grad = False
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
                test_loss += np.sqrt(loss.data[0]*mean_corrector)
                s += 1.
        print('test loss: '+str(test_loss/s))
        
