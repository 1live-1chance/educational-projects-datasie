import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


class NeuralNetworkTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None
        
    def load_and_preprocess_data(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
        test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
        
        (self.train_images, self.val_images, 
         self.train_labels, self.val_labels) = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42
        )
        
        self.test_images = test_images
        self.test_labels = test_labels
        
    def build_model(self):
        self.model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
            layers.Dense(10, activation='softmax')
        ])
        
    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train_model(self, epochs=10, batch_size=128):
        self.history = self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.val_images, self.val_labels),
            verbose=1
        )
        
    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(
            self.test_images, self.test_labels
        )
        return test_accuracy
        
    def get_predictions(self):
        test_predictions = self.model.predict(self.test_images)
        return np.argmax(test_predictions, axis=1)
    
    def get_training_history(self):
        return self.history.history if self.history else None
    
    def get_test_data(self):
        return self.test_images, self.test_labels