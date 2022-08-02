import random
import numpy as np
import matplotlib.pyplot as plt


class Plotting:

    def sanity_check(self, X_train, y_train):
        image_number = random.randint(0, len(X_train))
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(X_train[image_number, :, :, :])
        plt.subplot(122)
        plt.imshow(np.reshape(y_train[image_number], (128, 128)), cmap='gray')
        plt.show()

    def plot_generator(self,image_generator,mask_generator):
        x = image_generator.next()
        y = mask_generator.next()
        for i in range(0, 1):
            image = x[i]
            mask = y[i]
            plt.subplot(1, 2, 1)
            plt.imshow(image[:, :, 0])
            plt.subplot(1, 2, 2)
            plt.imshow(mask[:, :, 0])
            plt.show()

    def loss(self,history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def accuracy(self,history, epochs):
        acc = history.history['iou_score']
        # acc = history.history['accuracy']
        val_acc = history.history['val_iou_score']
        # val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'y', label='Training IOU')
        plt.plot(epochs, val_acc, 'r', label='Validation IOU')
        plt.title('Training and validation IOU')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.legend()
        plt.show()

    def plotting_test(self,model,X_test,y_test):
        test_img_number = random.randint(0, len(X_test) - 1)
        test_img = X_test[test_img_number]
        test_img_input = np.expand_dims(test_img, 0)
        ground_truth = y_test[test_img_number]
        prediction = model.predict(test_img_input)
        prediction = prediction[0, :, :, 0]
        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_img[:, :, 0], cmap='gray')
        plt.subplot(232)
        plt.title('Testing Label')
        plt.imshow(ground_truth[:, :, 0], cmap='gray')
        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(prediction)
        plt.show()




