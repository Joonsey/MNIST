import tensorflow as tf
import matplotlib.pyplot as plt

#from tensorflow.keras import Model


def make_noisy_data(m=0.1, b=0.3, n=100):
    x = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(len(x),),stddev=0.01)
    y = n * x * b * noise
    return x, y


x_train, y_train = make_noisy_data()



m = tf.Variable(0.)
b = tf.Variable(0.)

def predict(x):
    y = m * x + b
    return y

def squared_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

loss = squared_error(predict(x_train), y_train)
print("starting loss", loss.numpy())


learning_rate = 0.15
steps = 200

for i in range(steps):
    with tf.GradientTape() as tape:
        predictions = predict(x_train)
        loss = squared_error(predictions, y_train)

    gradients = tape.gradient(loss, [m,b])

    m.assign_sub(gradients[0] * learning_rate)
    m.assign_sub(gradients[1] * learning_rate)

    if i % 20 == 0:
        print(f"Step {i}, loss {loss.numpy()}")

print(f"m: {m.numpy()}, b: {b.numpy()}")


plt.plot(x_train, y_train, "b.")
plt.plot(x_train, predict(x_train))
plt.show()