import numpy as np
# from tensorflow.keras import utils
# import tensorflow as tf
# tf.python_io = tf

# # Set random seed
# np.random.seed(42)

# # Our data
# X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
# y = np.array([[0],[1],[1],[0]]).astype('float32')

# # Initial Setup for Keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Activation

# # Building the model
# xor = Sequential()

# # Add required layers
# xor.add(Dense(8, input_shape = (X.shape[1],)))  # Set the first layer to have 2 input nodes and 8 output nodes
# xor.add(Activation('tanh'))
# xor.add(Dense(1))
# xor.add(Activation('sigmoid'))

# # Specify loss as "binary_crossentropy", optimizer as "adam",
# # and add the accuracy metric
# xor.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Print the model architecture
# xor.summary()

# # Fitting the model
# history = xor.fit(X, y, epochs=500, verbose=0)

# # Scoring the model
# score = xor.evaluate(X, y)
# print("\nAccuracy: ", score[-1])

# # Checking the predictions
# print("\nPredictions:")
# print(xor.predict(X))