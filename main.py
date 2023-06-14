import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Generate a sample training dataset
def generate_training_data():
    x_train = np.zeros((1000, 64, 64, 1))
    y_train = np.zeros((1000, 4))

    for i in range(1000):
        start_x = np.random.randint(10, 55)
        start_y = np.random.randint(10, 55)
        end_x = np.random.randint(start_x + 5, 60)
        end_y = np.random.randint(start_y + 5, 60)

        x_train[i, start_x:start_x + 5, start_y:start_y + 5] = 1
        x_train[i, end_x:end_x + 5, end_y:end_y + 5] = 1

        y_train[i] = [start_x, start_y, end_x, end_y]

    return x_train, y_train


# Prepare the training dataset
X_train, y_train = generate_training_data()

# Data augmentation for training dataset
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             brightness_range=(0.8, 1.2))

datagen.fit(X_train)

# Prepare the network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          steps_per_epoch=len(X_train) // 32,
          epochs=10)


# Test the model on an example
def test_model(model):
    x_test = np.zeros((1, 64, 64, 1))
    start_x = np.random.randint(10, 55)
    start_y = np.random.randint(10, 55)
    end_x = np.random.randint(start_x + 5, 60)
    end_y = np.random.randint(start_y + 5, 60)
    x_test[0, start_x:start_x + 5, start_y:start_y + 5] = 1
    x_test[0, end_x:end_x + 5, end_y:end_y + 5] = 1
    y_true = np.array([start_x, start_y, end_x, end_y])

    predictions = model.predict(x_test)
    y_pred = predictions[0]

    # Display the image
    plt.imshow(x_test[0, :, :, 0], cmap='gray')
    plt.plot([y_true[1], y_true[3]], [y_true[0], y_true[2]], color='r', linewidth=2)
    plt.plot([y_pred[1], y_pred[3]], [y_pred[0], y_pred[2]], color='b', linewidth=2)
    plt.show()

    print("True coordinates: ", y_true)
    print("Predicted coordinates: ", y_pred)


# Evaluate the model's accuracy
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    accuracy = np.mean(np.abs(predictions - y_test))
    return accuracy


# Test the model
test_model(model)

# Prepare a test dataset
X_test, y_test = generate_training_data()

# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
print("Model accuracy:", accuracy)
