import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras import Model
from tqdm import tqdm
import numpy as np

class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(32, (3,3), activation='relu', input_shape = (28, 28, 1))
        self.pool1 = MaxPooling2D((2,2))
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.pool2 = MaxPooling2D((2,2))
        self.conv3 = Conv2D(64, (3,3), activation='relu')
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x

def activate_nn(train_df, test_df):
    model = CNN()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    @tf.function
    def train_step(images, labels):
        print(images.shape)
        with tf.GradientTape() as tape:
            predictions = model(images, training = True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
    
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss =    loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        batch_size = 50
        for i in range(int(60000/batch_size)):
            train_step(np.array(train_df[0][(i*batch_size):(i*batch_size)+batch_size]), 
                        np.array(train_df[1][(i*batch_size):(i*batch_size)+batch_size]))

        for i in range(int(10000/batch_size)):
            test_step(np.array(test_df[0][(i*batch_size):(i*batch_size)+batch_size]),
                        np.array(test_df[1][(i*batch_size):(i*batch_size)+batch_size]))

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )

