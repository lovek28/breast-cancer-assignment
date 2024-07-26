Breast Cancer Prediction App
Overview

This project is a web application for predicting breast cancer diagnoses using an Artificial Neural Network (ANN). The application is built with Streamlit and leverages a dataset to train the model. The ANN model is fine-tuned using feature selection and hyperparameter optimization to provide accurate predictions.
Features

    Interactive User Input: Allows users to input feature values via a web interface.
    Model Training and Evaluation: Uses an ANN model with hyperparameter tuning to achieve high accuracy.
    Feature Selection: Employs SelectKBest to select the most relevant features for the model.
    Streamlit Integration: Provides an easy-to-use web interface for predictions and model evaluation.

Installation
Prerequisites

    Python 3.7 or later
    Required Python packages listed in requirements.txt

Steps

    Clone the Repository

    bash

git clone https://github.com/yourusername/breast-cancer-prediction-app.git
cd breast-cancer-prediction-app

Install Dependencies

Install the required packages using pip:

bash

pip install -r requirements.txt

Download the Dataset

Ensure that the breast_cancer_dataset.csv file is in the root directory of the project.

Run the Application

Start the Streamlit app with:

bash

    streamlit run app.py

Usage

    Open the Application

    Navigate to the address provided by Streamlit in your terminal (usually http://localhost:8501).

    Interact with the App
        Use the sidebar to input feature values.
        The app will display a prediction of whether the cancer is Malignant or Benign based on the input features.

    View Model Accuracy

    The application also displays the model's accuracy on the test dataset.

Code Explanation
Data Preprocessing

    Load Dataset: The dataset is loaded and cleaned by removing unnamed columns.
    Encode Labels: The diagnosis column is mapped from categorical values ('B' and 'M') to numerical values (0 and 1).
    Feature and Target Separation: Features and target variables are separated for model training.

Feature Selection

    SelectKBest: The top 10 features are selected based on their statistical significance.

Model Training

    Grid Search: A grid search is performed to find the best hyperparameters for the MLPClassifier.
    Model Evaluation: The best model is evaluated on the test set to determine accuracy and other performance metrics.

Streamlit App

    User Input: Users can input values for various features through the Streamlit sidebar.
    Prediction: The model predicts the diagnosis based on the input features and displays the result.

Contributing

Feel free to open issues or submit pull requests if you have suggestions or improvements for the project.
License

This project is licensed under the MIT License - see the LICENSE file for details.
