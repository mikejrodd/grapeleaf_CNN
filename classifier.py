import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import shutil

# Define paths
original_data_dir = '/Users/michaelrodden/Desktop/original_grape_data'
train_dir = '/Users/michaelrodden/Desktop/original_grape_data/binary_train'
test_dir = '/Users/michaelrodden/Desktop/original_grape_data/binary_test'

# Function to clear and create directories
def clear_and_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Create new directories for binary classification
clear_and_create_dir(train_dir)
clear_and_create_dir(test_dir)

for category in ['healthy', 'esca']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Function to move files to new directories
def move_files(src_dir, dst_dir, category):
    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if category == 'healthy' and folder != 'ESCA':
                    shutil.copy(file_path, os.path.join(dst_dir, 'healthy'))
                elif category == 'esca' and folder == 'ESCA':
                    shutil.copy(file_path, os.path.join(dst_dir, 'esca'))

# Move files for training and testing
move_files(os.path.join(original_data_dir, 'train'), train_dir, 'healthy')
move_files(os.path.join(original_data_dir, 'train'), train_dir, 'esca')
move_files(os.path.join(original_data_dir, 'test'), test_dir, 'healthy')
move_files(os.path.join(original_data_dir, 'test'), test_dir, 'esca')

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Define Focal Loss function
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
        focal_loss_value = -alpha_t * tf.math.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss_value)
    return focal_loss_fixed

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with gradient clipping
optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)
model.compile(
    loss=focal_loss(gamma=2., alpha=0.25),
    optimizer=optimizer,
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model in the new Keras format
model.save('grapeleaf_classifier_new.keras')
