# Importing Packages :
import os
import boto3
import numpy as np
import segmentation_models as sm
from sklearn.model_selection import train_test_split

from MLPipeline.Loading_Data import Loading_Data
from MLPipeline.Data_Processing import Data_Processing
from MLPipeline.Plotting import Plotting
from MLPipeline.Model_Training import Model_Training

F1 = Loading_Data()
F2 = Data_Processing()
F3 = Plotting()
F4 = Model_Training()

# Loading The Data from S3 Bucket:
s3 = boto3.resource(
    service_name='s3',
    region_name='',
    aws_access_key_id='',
    aws_secret_access_key='')

# Defining the DATA Location in the S3 bucket :
default_location = "s3://appledatabucket/Apple/"
print(default_location)
print(os.listdir())
training_dir = "Model"

"""# Getting Data From the Bucket"""
img_data_array, mask_data_stack = F1.loading_data(s3, default_location)
print(len(img_data_array))

"""# Patching the Images"""
images = F2.Image_Patching(img_data_array)

"""# Patching the Mask"""
masks = F2.mask_patching(mask_data_stack)

# Printing the Shapes of Images:
print("---Shape of the Images and Masks---")
print(images.shape)
print(masks.shape)
print("Pixel values in the mask are: ", np.unique(masks))

"""Preprocessing the Image for RESNET:"""

BACKBONE = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE)
images1 = preprocess_input1(images)  # Preprocessing the Image data in corresponding to the 'Resnet34' specification.
print(images1.shape)
print(masks.shape)
#

""" Splitting the Dataset"""
X_train, X_test, y_train, y_test = train_test_split(images1,
                                                    masks,
                                                    test_size=0.25, random_state=42)

"""Sanity check, view few mages"""
F3.sanity_check(X_train, y_train)

"""Augmentation of Image and Patches"""
image_generator, valid_img_generator, mask_generator, valid_mask_generator = F2.augmentation(X_train,
                                                                                             X_test,
                                                                                             y_train,
                                                                                             y_test)
""""Plotting the Image and Masks"""
F3.plot_generator(image_generator, mask_generator)

"""Image and Patches as Generator"""
train_data_generator = F2.image_mask_generator(image_generator, mask_generator)
validation_datagen = F2.image_mask_generator(valid_img_generator, valid_mask_generator)

"""Compliling and Training the Model"""

# Defining the Model:
print("Loading the Model.....")
sm.set_framework('tf.keras')
sm.framework()

"""Initialising the Model:"""

model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
# printing the Model:
print(model.summary())

# Training the Model:
epochs = 1
history, model1 = F4.Model_Train(model, train_data_generator, validation_datagen, 50, 50, epochs)

score = F4.iou_score(model1, X_test, y_test)
print("IoU socre is: ", score)

# plot the training and validation accuracy and loss at each epoch:
F3.loss(history)

# Plotting Accuracy:
F3.accuracy(history, epochs)

# Plotting the Test :
F3.plotting_test(model1, X_test, y_test)

# Saving the model to the memory:
# if os.path.isfile('C:\Project\Image_Segmentation_sagemaker\Output\Model.h5') is False:
#     model1.save('C:\Project\Image_Segmentation_sagemaker\Output\Model.h5')
