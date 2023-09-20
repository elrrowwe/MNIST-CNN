from Convolutional import Convolutional
from Dense import *
from MaxPool import MaxPool
import tensorflow.keras as keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import accuracy_score

"""
This is the main file, in which the network is initialized and trained. 
"""

#loading the data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#taking a batch of data and normalizing it 
X_train = train_images[:5000] / 255.0
y_train = train_labels[:5000]

X_test = train_images[5000:10000] / 255.0
y_test = train_labels[5000:10000]

#one-hot encoding the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#initializing the layers
conv = Convolutional(X_train[0].shape, 3, 1)

pool = MaxPool(3)

dense = Dense(243, 10)

def train_network(X, y, conv, dense, pool, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        for i in range(len(X)):
            #forward pass
            conv_out = conv.forward(X[i])
            
            pool_out = pool.forward(conv_out)
            
            dense_out = dense.forward(pool_out)

            loss = cross_entropy(dense_out.flatten(), y[i])
            total_loss += loss

            # performing one-hot encoding on the output of the dense layer
            one_hot = np.zeros_like(dense_out)
            one_hot[np.argmax(dense_out)] = 1
            one_hot = one_hot.flatten()

            prediction = np.argmax(one_hot)
            target = np.argmax(y[i])

            if prediction == target:
                correct_predictions += 1

            dL_dout = cross_entropy_gradient(y[i], dense_out.flatten()).reshape((-1,1))
            
            dout_dense = dense.backward(dL_dout, learning_rate)
            
            dout_pool = pool.backward(dout_dense, learning_rate)
            
            dout_conv = conv.backward(dout_pool, learning_rate)

        # Print epoch statistics
        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X_train) * 100.0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")

        if accuracy >= 90.2:
            break
        
train_network(X_train, y_train, conv, dense, pool)

#the prediction funciton
def predict(inp, conv, pool, dense):
    conv_out = conv.forward(inp)
    pool.out = pool.forward(conv_out)

    flat_pool_out = pool.out.flatten().reshape(1, -1)

    predictions = dense.forward(flat_pool_out)

    return predictions

#testing the network on the test set
predictions = []

for x in X_test:
    pred = predict(x, conv, pool, dense)
    pred_one_hot = np.zeros_like(pred)
    pred_one_hot[np.argmax(pred)] = 1
    predictions.append(pred_one_hot.flatten())

predictions = np.array(predictions)

print(f'The accuracy of the network upon testing:{accuracy_score(predictions, y_test)}')

#visualizing the results
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap='binary')
    ax.set_xlabel(f"Actual: {y_test[i].argmax()}\nPredicted: {predictions[i].argmax()}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()