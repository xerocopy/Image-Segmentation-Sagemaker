# Image Segmentation Project with Sagemaker


## In this project, semantic image segmentation is used to discriminate the rotton apple from the good ones. The traininig and inference models are completed in the aws sagemaker enviornment. 


## Assets:
MLPipeline:
  -Data_Processing.py
  -Loading_Data.py
  -Model_Training.py
  -Plotting.py

Notebook:
  -ImageSegmentation.ipynb

Sagemaker Deployment:
  -ImageSegmentation_Deployment.ipynb
  -IMageSegmentation.py

Engine.py
README.md

## Steps from developement to deployment

#### 1. In the DevOps VM:
  - select the runtime: tensorflow 1.15 python3.6
  - loading data from s3
  - patching the images
  - image augmentation
  - split the dataset
  - traing the model
  - evaluate the model
  
  
#### 2. In Sagemaker Studio:
  - select instance based on the job and cost (ml.t3.medium /ml.g4dn.xlarge)
  - run the deployment notebook, initialize the training on tf_estimator
  - create the inference endpoint
  - check training process, model and endpoint on aws web interface
  - delete the endpoint and terminate the instance to avoid over charge







