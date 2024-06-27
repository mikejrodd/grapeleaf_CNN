import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Define the path to the training data
data_dir = '/Users/michaelrodden/Desktop/original_grape_data/train'

# Define the disease categories
diseases = ['Black Rot', 'ESCA', 'Leaf Blight', 'Healthy']

# Initialize a list to store images and their labels
images = []

# Iterate through each disease folder and select one random image
for disease in diseases:
    disease_path = os.path.join(data_dir, disease)
    image_files = os.listdir(disease_path)
    selected_image = random.choice(image_files)
    image_path = os.path.join(disease_path, selected_image)
    image = Image.open(image_path)
    images.append((image, disease))

# Create a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Plot each image in the grid
for i, ax in enumerate(axs.flat):
    if i < len(images):
        image, label = images[i]
        ax.imshow(image)
        ax.set_title(label)
        ax.axis('off')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
