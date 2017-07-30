#Date: 2017/07/29

from data_preprocessing import DataPreprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense
class ANN(DataPreprocessing):
    def scaleFeatures(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    def initialize(self):
        classifier = Sequential()
        
    def addInputLayerAndFirstHidden(self, **kwargs):
        classifier.add(Dense(**kwargs))
        
    def addHiddenLayer(self, **kwargs):
        classifier.add(Dense(**kwargs))
      
    def addOutputLayer:(self, **kwargs):
        classifier.add(Dense(**kwargs))
    
    def compileNN(self, **kwargs):
        classifier.compile(**kwargs)
    
    def fitToTrainingSet(self, **kwargs):
        classifier.fit(self.X_train, self.y_train, **kwargs)
        
    def predictResults(self):
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)

    def makeConfusionMatrix(self):
        cm = confusion_matrix(y_test, y_pred)
      
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
class CNN(ANN):
    def __init__(self):
        classifier = Sequential()
        
    def convolution(self, **kwargs):
        classifier.add(Convolution2D(**kwargs))
        
    def.maxPooling(self, **kwargs):
        classifier.add(MaxPooling2D(**kwargs))
        
    def addConvolutionLayer(self, **kwargs)):
        ##
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
    def flatten(self):
        classifier.add(Flatten())
        
    def fullConnection(self):
        ##
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
        
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
        
class RNN(DataPreprocessing):
    pass
