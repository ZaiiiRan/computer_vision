import tensorflow as tf
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 784)).astype('float32') / 255
x_test = x_test.reshape((10000, 784)).astype('float32') / 255

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_train = time.time()
model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=100,
    validation_split=0.1
)
end_train = time.time()
print(f'\nВремя обучения: {end_train - start_train:.2f}')

start_test = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test)
end_test = time.time()

print(f'Точность: {test_acc:.4f}')
print(f'Время работы: {end_test - start_test:.2f}')


model.save('./models/mnist_mlp.keras')
