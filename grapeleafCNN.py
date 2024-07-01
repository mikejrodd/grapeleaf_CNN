import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import shutil


original_data_dir = '/content/drive/MyDrive/gtprac/original_grape_data'
train_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_train_2'
test_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_test_2'

# def clear_and_create_dir(directory):
#     if os.path.exists(directory):
#         shutil.rmtree(directory)
#     os.makedirs(directory)

# clear_and_create_dir(train_dir)
# clear_and_create_dir(test_dir)

# for category in ['healthy', 'esca']:
#     os.makedirs(os.path.join(train_dir, category), exist_ok=True)
#     os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# def move_files(src_dir, dst_dir, category):
#     for folder in os.listdir(src_dir):
#         folder_path = os.path.join(src_dir, folder)
#         if os.path.isdir(folder_path):
#             for file in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, file)
#                 if category == 'healthy' and folder != 'ESCA':
#                     shutil.copy(file_path, os.path.join(dst_dir, 'healthy'))
#                 elif category == 'esca' and folder == 'ESCA':
#                     shutil.copy(file_path, os.path.join(dst_dir, 'esca'))

# move_files(os.path.join(original_data_dir, 'train'), train_dir, 'healthy')
# move_files(os.path.join(original_data_dir, 'train'), train_dir, 'esca')
# move_files(os.path.join(original_data_dir, 'test'), test_dir, 'healthy')
# move_files(os.path.join(original_data_dir, 'test'), test_dir, 'esca')

print(f"Training set image counts: {sum([len(files) for r, d, files in os.walk(train_dir)])}")
print(f"Test set image counts: {sum([len(files) for r, d, files in os.walk(test_dir)])}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Custom function to map filenames to their corresponding class indices
def custom_flow_from_directory(directory, batch_size, target_size=(150, 150), shuffle=True, seed=None):
    classes = ['esca', 'healthy', 'blight', 'rot']
    class_indices = {cls: idx for idx, cls in enumerate(classes)}
    filepaths = []
    labels = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg', 'JPG')):
                filepath = os.path.join(root, file)
                if 'esca' in root.lower():
                    labels.append(class_indices['esca'])
                else:
                    if 'L.Blight' in file:
                        labels.append(class_indices['blight'])
                    elif 'B.Rot' in file:
                        labels.append(class_indices['rot'])
                    else:
                        labels.append(class_indices['healthy'])
                filepaths.append(filepath)
    
    if not filepaths:
        raise ValueError("No image files found in the directory.")
    
    filepaths = np.array(filepaths)
    labels = np.array(labels, dtype=np.int32)  # Ensure labels are integers

    def decode_img(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = img / 255.0
        return img
    
    def process_path(file_path, label):
        file_path = tf.py_function(func=lambda z: tf.convert_to_tensor(z.numpy().decode(), dtype=tf.string), inp=[file_path], Tout=tf.string)
        img = decode_img(file_path)
        label = tf.one_hot(label, depth=len(classes))
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = custom_flow_from_directory(train_dir, batch_size=256)
validation_dataset = custom_flow_from_directory(test_dir, batch_size=256)

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
    Dense(4, activation='softmax') 
])

optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001, clipvalue=1.0)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

loss, accuracy = model.evaluate(validation_dataset)
print(f'Test accuracy: {accuracy}, Test loss: {loss}')

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, classification_report, confusion_matrix

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=256,
    class_mode='categorical',
    shuffle=False  
)

file_paths = test_generator.filepaths

# Predictions
y_pred_prob = model.predict(test_generator)
y_true = test_generator.classes

average_pred_healthy = None
average_pred_esca = None
average_pred_blight = None
average_pred_rot = None

if 'healthy' in test_generator.class_indices:
    average_pred_healthy = np.mean(y_pred_prob[y_true == test_generator.class_indices['healthy'], test_generator.class_indices['healthy']])
    print(f'Average prediction value for healthy: {average_pred_healthy}')

if 'esca' in test_generator.class_indices:
    average_pred_esca = np.mean(y_pred_prob[y_true == test_generator.class_indices['esca'], test_generator.class_indices['esca']])
    print(f'Average prediction value for esca: {average_pred_esca}')

if 'blight' in test_generator.class_indices:
    average_pred_blight = np.mean(y_pred_prob[y_true == test_generator.class_indices['blight'], test_generator.class_indices['blight']])
    print(f'Average prediction value for blight: {average_pred_blight}')

if 'rot' in test_generator.class_indices:
    average_pred_rot = np.mean(y_pred_prob[y_true == test_generator.class_indices['rot'], test_generator.class_indices['rot']])
    print(f'Average prediction value for rot: {average_pred_rot}')

non_esca_preds = [pred for pred in [average_pred_healthy, average_pred_blight, average_pred_rot] if pred is not None]
if not non_esca_preds or average_pred_esca is None:
    raise ValueError("Missing predictions for calculating the threshold.")
average_pred_non_esca = np.mean(non_esca_preds)
custom_threshold = (average_pred_non_esca + average_pred_esca) / 2
print(f'Custom threshold: {custom_threshold}')

binary_pred = (y_pred_prob[:, test_generator.class_indices['esca']] > custom_threshold).astype(int)

predicted_classes = []
true_classes = []
for i, file_path in enumerate(file_paths):
    true_class = y_true[i]
    predicted_class = binary_pred[i]
    print(f'File: {file_path}, Predicted class: {"esca" if predicted_class == 1 else "non-esca"}, True class: {"esca" if true_class == test_generator.class_indices["esca"] else "non-esca"}')
    predicted_classes.append(predicted_class)
    true_classes.append(1 if true_class == test_generator.class_indices['esca'] else 0)

binary_true = np.array(true_classes)
binary_pred = np.array(predicted_classes)

thresholds = np.arange(0.01, 1.0, 0.01)
f1_scores = []

for threshold in thresholds:
    binary_pred = (y_pred_prob[:, test_generator.class_indices['esca']] > threshold).astype(int)
    f1 = f1_score(binary_true, binary_pred)
    f1_scores.append(f1)

ideal_threshold = thresholds[np.argmax(f1_scores)]
print(f'Ideal threshold: {ideal_threshold}')
print(f'Best F1 score at ideal threshold: {max(f1_scores)}')

binary_pred = (y_pred_prob[:, test_generator.class_indices['esca']] > ideal_threshold).astype(int)

print("Binary Classification Report (esca vs others):")
print(classification_report(binary_true, binary_pred, target_names=['non-esca', 'esca']))

print("Binary Confusion Matrix (esca vs others):")
print(confusion_matrix(binary_true, binary_pred))
