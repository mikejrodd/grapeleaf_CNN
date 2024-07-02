import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from collections import Counter

original_data_dir = '/Users/michaelrodden/Desktop/original_grape_data'
train_dir = os.path.join(original_data_dir, 'binary_train')
test_dir = os.path.join(original_data_dir, 'binary_test')

def load_images_from_directory(directory, size=(150, 150)):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                image = imread(file_path)
                image_resized = resize(image, size, anti_aliasing=True)
                images.append(image_resized)
                labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = load_images_from_directory(train_dir)
test_images, test_labels = load_images_from_directory(test_dir)

def display_images(images, labels, category, n=5):
    category_images = images[labels == category]
    plt.figure(figsize=(15, 5))
    for i in range(min(n, len(category_images))):
        plt.subplot(1, n, i+1)
        plt.imshow(category_images[i])
        plt.title(category)
        plt.axis('off')
    plt.show()

print("Displaying sample images from the 'healthy' category:")
display_images(train_images, train_labels, 'healthy')

print("Displaying sample images from the 'esca' category:")
display_images(train_images, train_labels, 'esca')

def print_basic_statistics(images, labels):
    print(f'Total number of images: {len(images)}')
    print(f'Image shape: {images[0].shape}')
    label_counts = Counter(labels)
    print(f'Distribution of categories: {label_counts}')
    for label, count in label_counts.items():
        print(f'{label}: {count} images')

print("Basic statistics for training dataset:")
print_basic_statistics(train_images, train_labels)

print("Basic statistics for testing dataset:")
print_basic_statistics(test_images, test_labels)

def visualize_image_size_distribution(images):
    heights = [image.shape[0] for image in images]
    widths = [image.shape[1] for image in images]
    plt.figure(figsize=(10, 5))
    sns.histplot(heights, kde=True, color='blue', label='Height')
    sns.histplot(widths, kde=True, color='red', label='Width')
    plt.legend()
    plt.title('Distribution of Image Sizes')
    plt.xlabel('Size (pixels)')
    plt.ylabel('Frequency')
    plt.show()

print("Visualizing the distribution of image sizes in the training dataset:")
visualize_image_size_distribution(train_images)

def visualize_color_distribution(images, labels):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    categories = ['healthy', 'esca']
    colors = ('r', 'g', 'b')
    
    for ax, category in zip(axes, categories):
        category_images = images[labels == category]
        for i, color in enumerate(colors):
            # Concatenate all images in the category and calculate histogram
            all_pixels = np.concatenate([img[:, :, i].ravel() for img in category_images])
            histogram, bin_edges = np.histogram(all_pixels, bins=256, range=(0, 1))
            ax.plot(bin_edges[0:-1], histogram, color=color)
        ax.set_title(f'Color Distribution for {category.capitalize()} Images')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

print("Visualizing the color distribution for 'healthy' and 'esca' images:")
visualize_color_distribution(train_images, train_labels)

def calculate_mean_std(images):
    images_flat = images.reshape(-1, images.shape[-1])
    mean = np.mean(images_flat, axis=0)
    std = np.std(images_flat, axis=0)
    return mean, std

mean, std = calculate_mean_std(train_images)
print(f'Mean pixel value: {mean}')
print(f'Standard deviation of pixel values: {std}')

def visualize_pixel_value_distribution(images):
    images_flat = images.reshape(-1, images.shape[-1])
    plt.figure(figsize=(10, 5))
    sns.histplot(images_flat, bins=50, kde=True)
    plt.title('Distribution of Pixel Values')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

print("Visualizing the distribution of pixel values in the training dataset:")
visualize_pixel_value_distribution(train_images)
