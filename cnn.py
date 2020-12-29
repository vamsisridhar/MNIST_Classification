import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D
from tensorflow.keras import Model

class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()

