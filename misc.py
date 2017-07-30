"""
unit9
dimensionality reduction
PCA
LDA
kernel PCA

unit10
model Selection
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

#####


class SOM(DataPreprocessing):
    pass

class BoltzmannMachines(DataPreprocessing):
    pass

class AutoEncoder(DataPreprocessing):
    pass
