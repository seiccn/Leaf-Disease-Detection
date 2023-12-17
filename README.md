# Leaf Disease Detection
This project focuses on detecting leaf diseases in plants using a Convolutional Neural Network (CNN). The CNN is implemented using the Keras library with a TensorFlow backend. The goal is to provide a tool for identifying diseases in plant leaves based on images.

## 1. Data Preparation
The project begins by importing necessary libraries and reading in images to create a dataframe containing image paths and class labels. The dataset is split into training and testing sets, with information about the number of classes and the distribution of images across classes.

## 2. Data Augmentation
Image data generators are created for both the training and testing sets. These generators apply transformations such as rescaling, shearing, zooming, and horizontal flipping to augment the dataset.

## 3. CNN Architecture
The Convolutional Neural Network (CNN) architecture is defined using various layers, including convolutional layers, max-pooling layers, batch normalization, and dropout layers. The model is compiled with the Adam optimizer and categorical crossentropy loss.

## 4. Model Training
The model is trained using the training set, and the training process is visualized with plots showing the training and validation loss and accuracy over epochs.

## 5. Model Evaluation
The trained model's summary is displayed, providing details about each layer, the total number of parameters, and whether each layer is trainable.

## 6. Label Assignment
A list of labels is provided, associating each class index with a specific plant disease.

## 7. Predicting Output
A function is defined to predict the output for a given image path. The model can identify the plant disease based on the input image.

## Images for Training 👇

![image](https://github.com/seiccn/Leaf-Disease-Detection/assets/4949583/d16e26c5-db07-4bda-9548-bb19a4274f6b)

## Images for Testing 👇

![image](https://github.com/seiccn/Leaf-Disease-Detection/assets/4949583/edd271c7-6b5a-4fc8-835f-f8ee8c0aa959)




![image](https://github.com/seiccn/Leaf-Disease-Detection/assets/4949583/2c4a2ae5-216e-4936-8508-6e2eb01d6780)
