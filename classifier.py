import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import shutil

# image paths
original_data_dir = '/Users/michaelrodden/Desktop/original_grape_data'
train_dir = '/Users/michaelrodden/Desktop/original_grape_data/binary_train'
test_dir = '/Users/michaelrodden/Desktop/original_grape_data/binary_test'

def clear_and_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

clear_and_create_dir(train_dir)
clear_and_create_dir(test_dir)

for category in ['healthy', 'esca']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

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

# combine images to create healthy and esca paths
move_files(os.path.join(original_data_dir, 'train'), train_dir, 'healthy')
move_files(os.path.join(original_data_dir, 'train'), train_dir, 'esca')
move_files(os.path.join(original_data_dir, 'test'), test_dir, 'healthy')
move_files(os.path.join(original_data_dir, 'test'), test_dir, 'esca')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

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

# focal loss due to complex nature of training data (normal leaves and non-esca diseases being "healthy")
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
        focal_loss_value = -alpha_t * tf.math.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss_value)
    return focal_loss_fixed

# model building
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Adjust class weights to reduce ESCA false negatives
class_weights = {0: 1.0, 1: 2.3} 

# Compile the model with class weights and learning rate adjustment
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
model.compile(
    optimizer=optimizer,
    loss=focal_loss(gamma=2., alpha=0.25),
    metrics=['accuracy']
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with class weights and early stopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# save keras model
model.save('grapeleaf_classifier_best.keras')

# evaluate
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy}, Test loss: {loss}')


# Save the model in the new Keras format
model.save('grapeleaf_classifier_new.keras')
