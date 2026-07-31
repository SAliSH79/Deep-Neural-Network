import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Input features
input_features = 20

# Build a 3 layers Deep Neural Network model
model = Sequential(
    Dense(64, activation="relu", input_shape=(input_features,)) #Hidden Layer 1
    Dense(32, activation="relu"), #Hidden Layer 2
    Dense(1, activation="sigmoid") #Output layer

)
