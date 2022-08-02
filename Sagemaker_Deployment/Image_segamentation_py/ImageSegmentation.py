
# Importing Packages :
import os
import sys
import boto3
import subprocess


# Installing all the Pack Required :
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tifffile"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "patchify"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py==2.10.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "segmentation-models==1.0.1"])

# Importing nessary packages:
import io
import argparse
import tensorflow
import numpy as np
from numpy import asarray
from tensorflow import keras
from patchify import patchify
import segmentation_models as sm
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# The training code will be contained in a main gaurd (if __name__ == '__main__') so SageMaker will execute the code found in the main.

if __name__ == '__main__':

    # Parser to get the arguments:
    
    from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--validation_steps', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=50)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args, _ = parser.parse_known_args()
    
    # Defining the Hyperparameters:
    
    epochs          = args.epochs
    valid_step      = args.validation_steps
    steps_epoch     = args.steps_per_epoch
    gpu_count       = args.gpu_count
    model_dir       = args.model_dir
    
    args, _        = parser.parse_known_args()
    
    
    
    # Loading The Data:
    s3 = boto3.resource(
            service_name='s3',
            region_name='us-east-1',
            aws_access_key_id='AKIASCVPXXOPVBCKOLUF',
            aws_secret_access_key='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'
        )

    
    # Defining the DATA Location in the S3 bucket :
    
    default_location = "s3://appledatabucket/Apple/"
    print(default_location)    
    print(os.listdir())
    training_dir="Model"
    
    
    # Function To Load The Data:
    
    print("Loading Data....")
    def loading_data(default_location):
        img_data_array = []
        mask_data_stack = []
    
#         print("Reading the images")
        s3_bucket = "appledatabucket"
        keys = []
        for obj in s3.Bucket(s3_bucket).objects.all():
            keys.append(obj.key)

        for key in keys:
            file_stream = io.BytesIO()
            s3.Bucket(s3_bucket).Object(key).download_fileobj(file_stream)
            if ".jpg" in key and "Apple" in key:
                print(key)
                img = plt.imread(file_stream, format='jpg')
                print(img.shape)
                img_data_array.append(img)
            elif ".tiff" in key and "Apple" in key:
                mask = plt.imread(file_stream, format='tiff')
                print(mask.shape)
                mask_data_stack.append(mask)
                
        return img_data_array, mask_data_stack
    
        
    # Getting Images and Mask:
    
    img_data_array, mask_data_stack = loading_data(default_location)

        
    
    # Function to Patch the Images:
    
    print("Patching Images....")
    def Image_Patching(img_data_array):
        all_img_patches = []
        shapes = []
        for img in range(len(img_data_array)):
            large_image = img_data_array[img]
            shapes.append(large_image.shape)
            patches_img = patchify(large_image, (128, 128, 3), step=128)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :]
                    single_patch_img = (single_patch_img.astype('float32')) / 255.
                    all_img_patches.append(single_patch_img)
                    
        images = np.array(all_img_patches)
        images = np.reshape(images, (730, 128, 128, 3))
        return images
    
     #Getting the Image Patches:
        
    images = Image_Patching(img_data_array)


    # Function to patch the images:
    
    print("Patching the Masks.....")
        
    def mask_patching(mask_data_stack):
        all_mask_patches = []
        for img in range(len(mask_data_stack)):
            large_mask = mask_data_stack[img]
            patches_mask = patchify(large_mask, (128, 128), step=128)
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i, j, :, :]
                    single_patch_mask = single_patch_mask / 255.
                    all_mask_patches.append(single_patch_mask)
                    
        masks = np.array(all_mask_patches)
        masks = np.expand_dims(masks, -1)
        return masks
    
    
    #Getting the Mask Patches:
    
    masks = mask_patching(mask_data_stack)

    #Printing the Shapes of Images:
    
    print("---Shape of the Images and Masks---")
    print(images.shape)
    print(masks.shape)
    print("Pixel values in the mask are: ", np.unique(masks))
    
    # Building The Model:

    BACKBONE = 'resnet34'
    preprocess_input1 = sm.get_preprocessing(BACKBONE)
    images1 = preprocess_input1(images) # Preprocessing the Image data in corresponding to the 'Resnet34' specification.
    print(images1.shape)
    print(masks.shape)
    
    
    # Splitting Data to Train and Test:

    X_train, X_test, y_train, y_test = train_test_split(images1,
                                                        masks,
                                                        test_size=0.25, random_state=42)
    # print(X_train.shape)
    # print(X_test.shape)
    
    #New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks.  
    #This gives a binary mask rather than a mask with interpolated values. 
    
    seed=24
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    
    #Defining the ImageDataGenerator Parameters:
    
    img_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

    mask_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect',
                         preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype))         
    
    
    #Putting Images to the Generator for Augmentation:
    
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(X_train, augment=True, seed=seed)
    image_generator = image_data_generator.flow(X_train, seed=seed)
    valid_img_generator = image_data_generator.flow(X_test, seed=seed)  
    
    #Putting Masks to the Generator for Augmentation:
    
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_data_generator.fit(y_train, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(y_train, seed=seed)
    valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)
    
    # Creating the Train and Validation Generator:
    
    def image_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            yield (img, mask)

    train_data_generator = image_mask_generator(image_generator, mask_generator)

    validation_datagen = image_mask_generator(valid_img_generator, valid_mask_generator)
    

    # Defining the Model:
    print("Loading the Model.....")
    
    sm.set_framework('tf.keras')
    sm.framework()
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
       
    #printing the Model:
    
    print(model.summary())

    # Training :
    history = model.fit(train_data_generator,
                           validation_data = validation_datagen,
                           steps_per_epoch = steps_epoch,
                           validation_steps= valid_step,
                           epochs= epochs)
    
    
    # Function to get the IOU Score:
    
    def iou_score(model):
        y_pred = model.predict(X_test)
        y_pred_thresholded = y_pred > 0.5
        intersection = np.logical_and(y_test, y_pred_thresholded)
        union = np.logical_or(y_test, y_pred_thresholded)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
    
    
    #Printing The IOU Score:
    
    score = iou_score(model)
    print("IoU socre is: ", score)
    

    # Saving the Model:
    sess = K.get_session()
    tensorflow.saved_model.simple_save(sess,
                                       os.path.join(model_dir, 'model/1'),
                                       inputs={'inputs': model.input},
                                       outputs={t.name: t for t in model.outputs})
