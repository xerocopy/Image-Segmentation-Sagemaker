# Sagemaker


- Machine Learning is the hottest topic in the current era and the leading cloud provider Amazon web service (AWS) provides lots of tools to explore Machine Learning, creating models with a high accuracy rate.
  makes you familiar with one of those services on AWS i.e. Amazon SageMaker which helps in creating efficient and more accuracy rate 
- Machine learning models and the other benefit is that can use other AWS services in your model such as S3 bucket, amazon Lambda for monitoring the performance of your ML model you can use AWS CloudWatch which is a monitoring tool.

# TensorFlow

- TensorFlow is a software library or framework, designed by the Google team to implement machine learning and deep
  learning concepts in the easiest manner. It combines the computational algebra of optimization techniques for easy
  calculation of many mathematical expressions.
- TensorFlow is well-documented and includes plenty of machine learning libraries. It offers a few important
  functionalities and methods for the same.
- TensorFlow is also called a “Google” product. It includes a variety of machine learning and deep learning algorithms.
  TensorFlow can train and run deep neural networks for handwritten digit classification, image recognition, word
  embedding and creation of various sequence models.

# Image Sementation

- Image segmentation is a method in which a digital image is broken down into various subgroups called Image segments
  which helps in reducing the complexity of the image to make further processing or analysis of the image simpler.
  Segmentation in easy words is assigning labels to pixels.

- All picture elements or pixels belonging to the same category have a common label assigned to them. For example: Let’s
  take a problem where the picture has to be provided as input for object detection.

- Rather than processing the whole image, the detector can be inputted with a region selected by a segmentation
  algorithm. This will prevent the detector from processing the whole image thereby reducing inference time.

- ### semantic image segmentation
  - The goal of semantic image segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we’re predicting for every pixel in the image, this task is commonly referred to as dense prediction.
   Note that unlike the previous tasks, the expected output in semantic segmentation are not just labels and bounding box parameters. The output itself is a high resolution image (typically of the same size as input image) in which each pixel is classified to a particular class. Thus it is a pixel level image classification.

## Convolutional Neural Net

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. 
The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. 
While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. 
Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.

## Architecture

An image is nothing but a matrix of pixel values, right? So why not just flatten the image (e.g. 3x3 image matrix into a 9x1 vector) and feed it to a Multi-Level Perceptron for classification purposes?
In cases of extremely basic binary images, the method might show an average precision score while performing prediction of classes but would have little to no accuracy when it comes to complex images having pixel dependencies throughout.
A ConvNet is able to successfully capture the Spatial and Temporal dependencies in an image through the application of relevant filters. 
The architecture performs a better fitting to the image dataset due to the reduction in the number of parameters involved and reusability of weights. In other words, the network can be trained to understand the sophistication of the image better.
 The role of the ConvNet is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction. 
This is important when we are to design an architecture which is not only good at learning features but also is scalable to massive datasets.

- Pooling Layer
    :The Pooling layer is responsible for reducing the spatial size of the Convolved Feature. 
This is to decrease the computational power required to process the data through dimensionality reduction. 
Furthermore, it is useful for extracting dominant features which are rotational and positional invariant, thus maintaining the process of effectively training of the model.
There are two types of Pooling: Max Pooling and Average Pooling. Max Pooling returns the maximum value from the portion of the image covered by the Kernel. 
On the other hand, Average Pooling returns the average of all the values from the portion of the image covered by the Kernel.

- Transposed Convolutional Layer
    :The transposed Convolutional Layer is also (wrongfully) known as the Deconvolutional layer.
     A deconvolutional layer reverses the operation of a standard convolutional layer i.e. if the output generated through a standard convolutional layer is deconvolved, you get back the original input. The transposed convolutional layer is similar to the deconvolutional layer in the sense that the spatial dimension generated by both are the same. 
     Transposed convolution doesn’t reverse the standard convolution by values, rather by dimensions only.

## Code Description

    File Name : Engine.py
    File Description : Main class for starting the model training lifecycle

    File Name : Loading_Data.py
    File Description : Class for Loading Data from S3
    
    File Name : Plotting.py
    File Description : Class to Plot the Graphs and Images

    File Name : Data_Processing.py
    File Description : Class to load and transform the dataset. 
    
    File Name : Model_Training
    File Name : Class to train the Model that is Defined.

## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements
- Run Code `py Engine.py`
- Check output for all the visualization

### IPython SageMaker Notebook

Follow the instructions in the notebook `Image_Segmentation.ipynb`
 