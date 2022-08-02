import numpy as np

class Model_Training:

    def Model_Train(self, model, train_data_generator, validation_datagen, steps_epoch, valid_step, epochs):
        steps_epoch = steps_epoch
        valid_step = valid_step
        epochs = epochs
        # Training :
        history = model.fit(train_data_generator,
                            validation_data=validation_datagen,
                            steps_per_epoch=steps_epoch,
                            validation_steps=valid_step,
                            epochs=epochs)
        return history, model

    def iou_score(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_thresholded = y_pred > 0.5
        intersection = np.logical_and(y_test, y_pred_thresholded)
        union = np.logical_or(y_test, y_pred_thresholded)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
    #
