import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define paths
test_dir = '/Users/michaelrodden/Desktop/original_grape_data/binary_test'

# Data augmentation and normalization
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Important to set shuffle=False to match predictions with true labels
)

# Load the model in the new Keras format
model = load_model('grapeleaf_classifier_new.keras', compile=False)

# Compile the model with the custom loss function
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
        focal_loss_value = -alpha_t * tf.math.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss_value)
    return focal_loss_fixed

model.compile(
    loss=focal_loss(gamma=2., alpha=0.25),
    optimizer='adam',
    metrics=['accuracy']
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy}, Test loss: {loss}')

# Get the ground truth labels
y_true = test_generator.classes

# Get the predicted labels
y_pred = model.predict(test_generator)
y_pred = np.round(y_pred).astype(int).flatten()

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
