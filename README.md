## Classification Project: KNN and SVM
This project focuses on classifying data into four classes using two popular classification algorithms: K-Nearest Neighbors (KNN) and Support Vector Machines (SVM). The goal is to provide accurate predictions for unseen data based on the trained models.

## Dataset
The dataset used for this project consists of labeled images. It includes a total of 880 images distributed across four classes. Each image is preprocessed and converted into a feature vector representation using techniques such as Histogram of Oriented Gradients (HOG).

The dataset is divided into two main subsets:
FOR KNN:
Training Set: Contains 80% of the images for training the models.
Test Set: Contains 20% of the images for evaluating the performance of the trained models. 

FOR SVM:
Training Set: Contains 75% of the images for training the models.
Test Set: Contains 25% of the images for evaluating the performance of the trained models. 


## Requirements
To run the code and reproduce the results, the following requirements should be met:

Python 3.11.3
Required Python libraries:

- numpy
- pandas
- matplotlib
- imutils
- seaborn
- opencv-python
- joblib
- scikit-image

Make sure you have these libraries installed in your Python environment before running the code.

You can install the required libraries using piP

## Code Organization

The code is organized into the following files:

train-classification-knn.ipynb: Implements the KNN classification algorithm. It includes functions for training the model, making predictions, and evaluating the performance.

train-classification-svm.ipynb: Implements the SVM classification algorithm. It includes functions for training the model, making predictions, and evaluating the performance.

images: Contains contains the whole dataset used to train and test models

ValidationData: Contains useen data to validate the model

knn_sample.yml: saved knn model

svm.joblib: saved SVM model

## To run the code and reproduce the classification results, follow these steps:

Clone the repository or download the code files.

Install the required Python libraries mentioned in the Requirements section.

Place the dataset files in the appropriate directory (e.g., data folder).

Open the train-classification-svm.ipynb and  train-classification-svm.ipynb file and adjust any necessary configurations, such as file paths, hyperparameters, or feature extraction techniques.

Run the  script. It will load the dataset, train the KNN and SVM models, make predictions on the test set, and display performance metrics (accuracy, confusion matrix, etc.).

## Conclusion
In this project, we successfully classified the dataset into four classes using the KNN and SVM classification algorithms. Both models achieved high accuracy on the test set, indicating their effectiveness in predicting unseen data. The results suggest that these models can be utilized for similar classification tasks with similar datasets.
