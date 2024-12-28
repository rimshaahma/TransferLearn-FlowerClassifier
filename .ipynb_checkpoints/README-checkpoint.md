
# Transfer Learning with Pre-trained Models for Image Classification

## Project Overview

In this project, we will leverage **transfer learning** using **pre-trained models** to classify flowers into different categories. The project involves using three well-known pre-trained models: **ResNet50**, **VGG16**, and **InceptionV3**. We will fine-tune these models for the flower classification task using the **Flowers dataset**, which consists of images from five different flower classes: **Daisy**, **Dandelion**, **Roses**, **Sunflowers**, and **Tulips**.

This project demonstrates the power of transfer learning in real-world machine learning applications, where we can take advantage of models pre-trained on large datasets (like ImageNet) and adapt them to our specific problem, thereby saving time and computational resources.

---

## Table of Contents

- [Theory and Concepts](#theory-and-concepts)
- [Steps Overview](#steps-overview)
  - [Step 1: Install Required Libraries](#step-1-install-required-libraries)
  - [Step 2: Load and Preprocess Data](#step-2-load-and-preprocess-data)
  - [Step 3: Build the Model Using Transfer Learning](#step-3-build-the-model-using-transfer-learning)
  - [Step 4: Train the Model](#step-4-train-the-model)
  - [Step 5: Evaluate the Model](#step-5-evaluate-the-model)
  - [Step 6: Fine-Tuning](#step-6-fine-tuning)
  - [Step 7: Model Inference and Evaluation](#step-7-model-inference-and-evaluation)
  - [Step 8: Save and Load the Model](#step-8-save-and-load-the-model)
  - [Step 9: Build a GUI for Image Classification](#step-9-build-a-gui-for-image-classification)
- [Dataset](#dataset)
- [Conclusion](#conclusion)

---

## Theory and Concepts

### 1. **Transfer Learning**

**Transfer learning** is a machine learning technique where a model trained on one task is reused for a different but related task. Instead of starting from scratch, we use the knowledge learned by a pre-trained model (usually trained on a large dataset) and fine-tune it for a specific task.

In this project, we used pre-trained models that were originally trained on the **ImageNet** dataset, which consists of millions of labeled images across thousands of categories. The pre-trained models (ResNet50, VGG16, and InceptionV3) have learned useful features such as edges, shapes, textures, and patterns that are general and applicable to many other tasks.

By applying transfer learning, we save significant time and computational resources since we do not need to train the model from scratch.

### 2. **Pre-trained Models: ResNet50, VGG16, and InceptionV3**

- **ResNet50**: 
  - ResNet50 is a deep convolutional neural network architecture designed to enable very deep networks by introducing **residual connections** that help to mitigate the vanishing gradient problem.
  - It is known for its **high accuracy** in image classification tasks.

- **VGG16**:
  - VGG16 is a deep neural network architecture known for its simple and uniform architecture. It uses a series of 3x3 convolutional layers and pooling layers.
  - VGG16 is deep (16 layers) and is recognized for its robustness and simplicity.

- **InceptionV3**:
  - InceptionV3 is a more complex architecture that uses **convolutional layers of different sizes** to capture features at various scales. It uses the **Inception module** to improve computational efficiency and performance.
  - InceptionV3 is one of the most efficient models with state-of-the-art performance.

### 3. **Fine-Tuning a Pre-trained Model**

Fine-tuning is a process where you unfreeze some layers of a pre-trained model and train them on the new task. Fine-tuning can be done by:

1. **Freezing the earlier layers**: The first few layers of a model usually capture general features like edges and textures, which are useful for most tasks. These layers can remain frozen.
2. **Unfreezing the deeper layers**: The deeper layers are more specific to the original task. In fine-tuning, we adjust the weights of these layers to adapt to the new task.

By doing this, we can adapt the pre-trained model to work well with the new dataset while still taking advantage of the knowledge the model has learned from its original task.

### 4. **Data Augmentation**

Data augmentation refers to techniques used to artificially increase the size of the training dataset by creating modified versions of images in the dataset. Common techniques include:

- **Rotation**
- **Shifting** (Width and height shifts)
- **Zooming**
- **Flipping**
- **Shearing**

These transformations help improve model generalization and reduce overfitting by introducing more variation into the training data.

### 5. **Image Preprocessing and Normalization**

When working with image data, it is common to normalize pixel values to the range [0, 1]. This helps in the convergence of the model during training by making the data more suitable for the model’s activation functions (e.g., ReLU). In our case, this normalization is done using `rescale=1./255`.

---

## Steps Overview

### Step 1: Install Required Libraries

Before we start coding, we need to install the necessary libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Step 2: Load and Preprocess Data

We used **Keras ImageDataGenerator** to preprocess the data. It allows us to easily scale images and perform augmentation.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to the range [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

### Step 3: Build the Model Using Transfer Learning

We used **ResNet50**, **VGG16**, and **InceptionV3** as base models. These models are pre-trained on the ImageNet dataset.

```python
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras import layers, models

# Example for ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
```

### Step 4: Train the Model

We trained the model using the pre-processed data for 10 epochs, monitoring accuracy during training and validation.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

### Step 5: Evaluate the Model

Once the model is trained, we evaluate it using the test dataset. The evaluation includes accuracy and loss metrics.

```python
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")
```

### Step 6: Fine-Tuning

After training the model, we may unfreeze some of the top layers and retrain the model. Fine-tuning can improve accuracy, especially when the dataset is small.

```python
for layer in base_model.layers:
    layer.trainable = True  # Unfreeze all layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```

### Step 7: Model Inference and Evaluation

We make predictions using the trained model. We can visualize the model's performance with metrics like **precision**, **recall**, and the **confusion matrix**.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing confusion matrix
cm = confusion_matrix(true_class_indices, predicted_class_indices)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

### Step 8: Save and Load the Model

Once the model is trained, we save it for future use or deployment.

```python
model.save('flower_classification_model.h5')
```

### Step 9: Build a GUI for Image Classification

Finally, we create a simple **Tkinter** GUI that allows users to upload images and receive real-time predictions from the trained model.

```python
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to classify image
def classify_image(image_path):
    ...
```

---

## Dataset

The dataset used in this project consists of images of flowers, categorized into five classes: **Daisy**, **Dandelion**, **Roses**, **Sunflowers**, and **Tulips**. The dataset is organized into two directories:

- **train**: Contains images used for training, organized by flower category.
- **test**: Contains images used for evaluation, also organized by flower category.

You can download the dataset from the following link:

[Flower Classification Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

---

## Conclusion

In this project, we demonstrated how **transfer learning** can be applied to a flower classification task using pre-trained models like **ResNet50**, **VGG16**, and **InceptionV3**. Through fine-tuning and evaluating the models, we observed how transfer learning can improve performance, especially when working with smaller datasets. We also discussed how to deploy the model using a GUI for real-time image classification.

This project provides hands-on experience with key concepts like transfer learning, fine-tuning, and model deployment, which are essential skills for real-world machine learning applications.
