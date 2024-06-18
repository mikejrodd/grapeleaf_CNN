# Esca Disease Classifier for Grape Vines

This repository contains a classifier for identifying Esca disease in grape vines from leaf images. Esca is a serious disease affecting grapevines, characterized by symptoms like tiger stripes on leaves, white rot in wood, and vine dieback. This classifier aims to identify Esca disease from leaf images, even when other diseases might be present.

## Dataset

The dataset used for training and testing the model can be downloaded from [Kaggle: Grape Disease Dataset](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original?resource=download). 

### Download the Dataset

1. Go to the [Grape Disease Dataset](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original?resource=download) on Kaggle.
2. Download the dataset to your local machine.
3. Extract the dataset and place it in a directory named `original_grape_data` under the root of this repository.

## Training the Classifier

1. **Prepare the Data**:
    - The script will automatically create binary classification folders for training and testing.
    - Ensure the dataset is placed correctly as mentioned above.

2. **Update File Paths**
    - Add correct file paths to your downloaded data to `classifier.py`

3. **Run the Training Script**:

    ```sh
    python classifier.py
    ```

   The `classifier.py` script will:
   - Prepare the data by creating `binary_train` and `binary_test` folders.
   - Train the classifier using the Keras Tuner for hyperparameter optimization.
   - Save the best model in the `grapeleaf_classifier_best.keras` file.

## Esca Disease in Grape Vines

Esca is a complex and destructive disease affecting grapevines, caused by a group of fungi. Symptoms include:

- **Tiger Stripes**: Yellow or red stripes on the leaves, which are often necrotic.
- **White Rot**: The decay of wood in the vine, leading to the vine's collapse.
- **Dieback**: Sudden dieback of shoots and parts of the vine.

### Goal of the Classifier

The classifier's goal is to identify Esca disease from a leaf image, even when other diseases like Black Rot or Leaf Blight may be present. By accurately identifying Esca, vineyard managers can take timely actions to control the spread of the disease and protect their grapevines.

## Model Performance Evaluation

### Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **esca**      | 0.79      | 0.97   | 0.87     | 480     |
| **healthy**   | 0.99      | 0.90   | 0.94     | 1325    |

### Explanation of Metrics

- **Precision**: The ratio of true positive predictions to the total predicted positives. Precision indicates how many of the positive predictions were actually correct.
- **Recall**: The ratio of true positive predictions to the total actual positives. Recall indicates how many of the actual positives were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.
- **Support**: The number of actual occurrences of each class in the dataset.

### Accuracy

The overall accuracy of the classifier is 0.92 (92%), which indicates the ratio of correctly predicted instances to the total instances.

### Confusion Matrix

- **True Positives (TP)**: `esca` correctly identified as `esca`: 468
- **True Negatives (TN)**: `healthy` correctly identified as `healthy`: 1197
- **False Positives (FP)**: `healthy` incorrectly identified as `esca`: 12
- **False Negatives (FN)**: `esca` incorrectly identified as `healthy`: 128

## Detailed Explanation of Model Training Components

This section explains the various components and steps involved in training the Esca Disease Classifier for grape leaf images.

### Libraries Needed

- **TensorFlow**: An open-source platform for machine learning, used to build and train the neural network.
- **Keras**: A high-level neural networks API, written in Python and capable of running on top of TensorFlow.
- **shutil**: A Python module for high-level file operations, used for moving files to the appropriate directories.

### Data Preparation

#### Directory Setup
The directories `binary_train` and `binary_test` are created to store the processed training and testing data. The data is separated into two categories: `healthy` and `esca`.

#### Data Moving Function
A function `move_files` is used to move images from the original dataset to the `binary_train` and `binary_test` directories based on their categories.

### ImageDataGenerator

The `ImageDataGenerator` class is used to generate batches of tensor image data with real-time data augmentation. This helps in increasing the diversity of the training dataset and improving the model's generalization.

#### Training Data Generator

```python
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
```
- **rescale**: Scales the pixel values to the range [0, 1].
- **rotation_range**: Randomly rotates images within the specified range of degrees.
- **width_shift_range**: Randomly shifts the images horizontally.
- **height_shift_range**: Randomly shifts the images vertically.
- **shear_range**: Applies random shear transformations.
- **zoom_range**: Randomly zooms into images.
- **horizontal_flip**: Randomly flips the images horizontally.
- **fill_mode**: Specifies the fill mode for points outside the boundaries of the input.

#### Testing Data Generator

```python
test_datagen = ImageDataGenerator(rescale=1./255)
```
- **rescale**: Scales the pixel values to the range [0, 1].

### Model Architecture

The model is built using the `Sequential` class from Keras, with multiple layers added to form a convolutional neural network (CNN).

```python
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
```
- **Conv2D**: Convolutional layers that apply a convolution operation to the input.
- **MaxPooling2D**: Pooling layers that reduce the spatial dimensions of the output volume.
- **Flatten**: Flattens the input to prepare it for the dense layer.
- **Dense**: Fully connected layers.
- **Dropout**: Regularization technique to prevent overfitting.

### Loss Function

#### Focal Loss
Due to the complex nature of the training data (normal leaves and non-esca diseases being classified as "healthy"), focal loss is used. Focal loss focuses on hard-to-classify examples, which helps in handling class imbalance. It was also chosen due to:

- Focal loss is designed to address class imbalance, which is a common issue in many real-world datasets, including the dataset for grapevine diseases. By focusing more on hard-to-classify examples, focal loss helps to balance the contribution of different classes to the loss function.
- In many classification tasks, the model can quickly learn to classify easy examples correctly, but struggles with hard examples. Focal loss reduces the loss contribution from easy examples and increases the focus on hard examples, thereby improving the model's performance on challenging cases.
- The parameters gamma and alpha in focal loss allow for adjusting the focus on hard examples and balancing the importance of positive/negative examples. The gamma parameter adjusts the rate at which easy examples are down-weighted, while the alpha parameter balances the importance of positive and negative examples.
- In the context of detecting diseases like Esca, it is crucial to accurately identify minority classes (e.g., diseased leaves) which might be underrepresented in the dataset. Focal loss helps to improve the model's sensitivity to these minority classes, reducing false negatives.

### Class Weights

Class weights are adjusted to reduce the initial high false negative rate for `esca`.

```python
class_weights = {0: 1.0, 1: 2.3}
```

### Optimizer

The Adam optimizer with a learning rate adjustment is used for the below reasons:

- The Adam optimizer computes adaptive learning rates for each parameter. This means that it adjusts the learning rate individually for each parameter based on the estimates of first and second moments of the gradients. This helps the optimizer to converge faster and more efficiently, especially in scenarios with sparse gradients or noisy data.
- Adam combines the benefits of two other popular optimizers, RMSProp and Momentum. RMSProp adjusts the learning rate based on a moving average of recent gradients, which helps in dealing with non-stationary objectives. Momentum helps accelerate gradients vectors in the right direction, thus leading to faster converging.
- The adaptive learning rate and momentum help Adam perform well even when the data is noisy, which can be particularly useful for complex tasks like image classification where the data might have a lot of variability.
- Adam generally requires less tuning of the hyperparameters compared to other optimizers. The default settings for the learning rates are often suitable, making it easier to use.
- Adam is computationally efficient and has low memory requirements, which makes it suitable for large datasets and high-dimensional parameter spaces.
- Empirically, Adam has been shown to work well across a wide range of deep learning architectures and tasks, including convolutional neural networks used in image recognition tasks like this one.

### Model Compilation

The model is compiled with the Adam optimizer, focal loss, and accuracy as the metric.

```python
model.compile(
    optimizer=optimizer,
    loss=focal_loss(gamma=2., alpha=0.25),
    metrics=['accuracy']
)
```

### Early Stopping

Early stopping is used to prevent overfitting. The training stops if the validation loss does not improve for 10 consecutive epochs.

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

### Model Training

The model is trained with the training data generator, class weights, and early stopping callback.

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping]
)
```
