import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Start")
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,)),
    keras.layers.Dense(10),
    keras.layers.Dense(1),
])

model.compile(optimizer='adam',
                loss=keras.losses.mean_squared_error)

model.summary()

model.fit([0.0, 1.0], [0.4, 0.4], epochs=100)

output = model.predict([0.2])
print(output)
