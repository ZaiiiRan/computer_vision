import tensorflow as tf
import time

_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test  = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
y_test  = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.load_model('./models/mnist_cnn.keras')
model.summary()

start_test = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test)
end_test = time.time()

print(f'Точность: {test_acc:.4f}')
print(f'Время работы: {end_test - start_test:.2f}')
