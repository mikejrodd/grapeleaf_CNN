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



