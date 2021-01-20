import numpy as np
import tensorflow as tf
import os
import tensorflow_datasets as tfds
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']


num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.

    return image, label


scaled_train_and_validation_data = mnist_train.map(scale)

test_data = mnist_test.map(scale)


BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)

train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)

validation_data = validation_data.batch(num_validation_samples)

test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))


input_size = 784
output_size = 10
hidden_layer_size = 100
    
def create_model():
    model = tf.keras.Sequential([
    
    tf.keras.layers.Flatten(input_shape=(28,28,1)), 
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 3rd hidden layer

    tf.keras.layers.Dense(output_size, activation="softmax"), # output layer

    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    return model


NUM_EPOCHS = 10

model = tf.keras.models.load_model('saved_model')

#model = create_model()
#tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir', histogram_freq=1)
#model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets),validation_steps=1, verbose =2, callbacks=[tensorboard_callbacks])

#model.save('saved_model/')

size = (28,28)

#model.evaluate(train_data)

def getArray():
    i = Image.open('mnist.png').convert('L')
    data = np.asarray(i)
    #np_img = np.reshape(np_img, (1,28,28,1))
    new_img = Image.fromarray(data)
    plt.imshow(new_img)
    plt.show()
    data = np.reshape(data, (1,28,28,1))
    data = tf.cast(data, tf.float32)
    data /= 255
    return data

def guess(data):

    prediction = model.predict(data)
    print(f"I think it is {np.argmax(prediction)}")

if __name__ == '__main__':
    guess(getArray())

    
# activate tensorflow
# tensorboard --logdir tb_callback_dir
