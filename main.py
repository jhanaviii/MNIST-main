import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28))) 
model.add(layers.Dense(units =128, activation='relu'))  
model.add(layers.Dropout(0.2))                    
model.add(layers.Dense(units = 10, activation='softmax')) 

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels , epochs=5, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

model.save('mnist_model.h5')

import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('mnist_model.h5')

(_, _), (test_images, test_labels) = mnist.load_data()

random_index = np.random.randint(0, test_images.shape[0])

image = test_images[random_index]
label = test_labels[random_index]

input_image = np.expand_dims(image, axis=0)

input_image = input_image / 255.0

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title(f"Original Image\nTrue Label: {label}")

predictions = model.predict(input_image)

predicted_label = np.argmax(predictions)

plt.subplot(1, 2, 2)
plt.imshow(input_image[0], cmap='gray')
plt.title(f"Normalized Image\nPredicted Label: {predicted_label}\nTrue Label: {label}")

plt.show()
