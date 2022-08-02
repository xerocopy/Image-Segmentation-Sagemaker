import numpy as np
from patchify import patchify
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Data_Processing:

    def Image_Patching(self, img_data_array):
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

    # Function to patch the Masks:
    def mask_patching(self, mask_data_stack):
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

    def augmentation(self, X_train, X_test, y_train, y_test):
        seed = 24
        # Defining the ImageDataGenerator Parameters:
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
                                  preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype))
        # Putting Images to the Generator for Augmentation:

        image_data_generator = ImageDataGenerator(**img_data_gen_args)  # Initialising the Image Generator Model
        image_data_generator.fit(X_train, augment=True, seed=seed)
        image_generator = image_data_generator.flow(X_train, seed=seed)
        valid_img_generator = image_data_generator.flow(X_test, seed=seed)
        # Putting Masks to the Generator for Augmentation:

        mask_data_generator = ImageDataGenerator(**mask_data_gen_args)  # Initialising the Mask Generator Model
        mask_data_generator.fit(y_train, augment=True, seed=seed)
        mask_generator = mask_data_generator.flow(y_train, seed=seed)
        valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)
        return image_generator, valid_img_generator, mask_generator, valid_mask_generator


    def image_mask_generator(self,image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            yield img, mask
