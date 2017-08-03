# 2017/07/29

from data_preprocessing import DataPreProcessing
from data_postprocessing import DataPostProcessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NN(DataPreProcessing, DataPostProcessing):
    def __init__(self):
        pass
    
    def compileNN(self, **kwargs):
        self.classifier.compile(**kwargs)



import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
class ANN(NN):
    def scaleFeatures(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
    def build(self):
        self.classifier = Sequential()
        self.classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
        self.classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
        self.classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return self.classifier
    
    def fitToTrainingSet(self, **kwargs):
        self.classifier.fit(self.X_train, self.y_train, **kwargs)
        
    def predictResults(self):
        self.y_pred = classifier.predict(X_test)
        self.y_pred = (y_pred > 0.5)

    def makeConfusionMatrix(self):
        self.cm = confusion_matrix(y_test, y_pred)

    def makeNewPrediction(self, lst_feature_values):
        new_prediction = classifier.predict(sc.transform(np.array([lst_feature_values])))
        return new_prediction > 0.5
        
    def evaluate(self):
        #
        classifier = KerasClassifier(build_fn = self.build, batch_size = 10, epochs = 100)
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
        mean = accuracies.mean()
        variance = accuracies.std()
        
    def improve(self):
        #
        classifier = KerasClassifier(build_fn = self.build)
        parameters = {'batch_size': [25, 32],
                      'epochs': [100, 500],
                      'optimizer': ['adam', 'rmsprop']}
        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 10)
        grid_search = grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
class CNN(ANN):
    def __init__(self):
        self
        
    def build(self):
        #
        # Initialising the CNN
        classifier = Sequential()
        
        # Step 1 - Convolution
        classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
        
        # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a second convolutional layer
        classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Step 3 - Flattening
        classifier.add(Flatten())
        
        # Step 4 - Full connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        
    def fitToImages(self):
        ##
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
        
        test_datagen = ImageDataGenerator(rescale = 1./255)
        
        training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                         target_size = (64, 64),
                                                         batch_size = 32,
                                                         class_mode = 'binary')
        
        test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
        
        classifier.fit_generator(training_set,
                                 samples_per_epoch = 8000,
                                 nb_epoch = 25,
                                 validation_data = test_set,
                                 nb_val_samples = 2000)
        
    def makeNewPrediction(self):
        import numpy as np
        from keras.preprocessing import image
        test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
     
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math      
from sklearn.metrics import mean_squared_error        
class RNN(NN):
    def importTrainingSet(self):
        #
        training_set = pd.read_csv('Google_Stock_Price_Train.csv')
        training_set = training_set.iloc[:,1:2].values

    def scaleFeatures(self):
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        training_set = sc.fit_transform(training_set)

    def getInputsAndOutputs(self):
        X_train = training_set[0:1257]
        y_train = training_set[1:1258]

    def reshape(self):
        X_train = np.reshape(X_train, (1257, 1, 1))
        
    def build(self):
        # Initialising the RNN
        regressor = Sequential()
        
        # Adding the input layer and the LSTM layer
        regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
        
        # Adding the output layer
        regressor.add(Dense(units = 1))
    
    def compileNN(self):
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
    def fitToTrainingSet(self):
        regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)
        
    def makePredictions(self):
        # Getting the real stock price of 2017
        test_set = pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price = test_set.iloc[:,1:2].values
        
        # Getting the predicted stock price of 2017
        inputs = real_stock_price
        inputs = sc.transform(inputs)
        inputs = np.reshape(inputs, (20, 1, 1))
        predicted_stock_price = regressor.predict(inputs)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    def visualizeResults(self):
        plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
        plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.show()
        
    def evaluate(self):
        rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
