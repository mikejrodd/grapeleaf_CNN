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
| **esca**      | 0.52      | 1.00   | 0.69     | 480     |
| **healthy**   | 1.00      | 0.67   | 0.80     | 1325    |
| **macro avg** | 0.76      | 0.83   | 0.74     | 1805    |
| **weighted avg** | 0.87   | 0.76   | 0.77     | 1805    |

### Accuracy

The overall accuracy of the classifier is 0.76 (76%), which indicates the ratio of correctly predicted instances to the total instances.

### Confusion Matrix

- **True Positives (TP)**: `esca` correctly identified as `esca`: 480
- **True Negatives (TN)**: `healthy` correctly identified as `healthy`: 887
- **False Positives (FP)**: `healthy` incorrectly identified as `esca`: 0
- **False Negatives (FN)**: `esca` incorrectly identified as `healthy`: 438

The classifier has a high recall for the `esca` class, correctly identifying all actual `esca` samples. However, there are a significant number of false negatives for the `healthy` class, which affects the overall accuracy. The model performs reasonably well, but there is room for improvement, especially in reducing false negatives for `healthy` samples.


