PredictaHealth: Diabetes Prediction App
PredictaHealth is a machine learning-based web application designed to predict whether a person is diabetic based on various health metrics. Using a Random Forest model trained on a dataset of health-related features, the app allows users to input their health data and get a prediction on their diabetes status.

Features
User Input: Users can input the following health parameters:

Number of Pregnancies

Glucose Concentration

Blood Pressure (mm Hg)

Skin Thickness (mm)

Insulin Level (mu U/ml)

Body Mass Index (BMI)

Diabetes Pedigree Function

Age

Diabetes Prediction: Based on the provided inputs, the app predicts if the user is likely to have diabetes (1) or not (0).

Model: The app uses a Random Forest Classifier trained on a dataset containing medical attributes and diabetes outcomes.

Scalability: The application can be expanded to include more predictive models and a larger dataset.

Technology Stack
Frontend: Streamlit (for creating interactive web apps with Python)

Backend: Python, Scikit-learn (for machine learning and data preprocessing)

Model Serialization: Joblib (for saving and loading the trained model)

Deployment: The app can be deployed on platforms like Streamlit Cloud or Heroku.

Installation
To run the application locally, follow these steps:
1. Clone the repository
git clone https://github.com/AfnaanUddin/PredictaHealth.git
cd PredictaHealth

2. Install Dependencies
   pip install -r requirements.txt

3. Train the Model
   python train_model.py
This will generate two files:

diabetes_model.pkl (the trained Random Forest model)
scaler.pkl (the standard scaler used for data preprocessing)

4. Run the Streamlit App
   streamlit run app.py
This will start the Streamlit server, and you can view the app by opening http://localhost:8501 in your browser.

Usage
Open the app in your browser.

Enter the health data in the provided input fields.

Click "Submit" to get the diabetes prediction.

The result will show whether the person is diabetic or not.


NON-DIABETIC TEST CASE:

Pregnancies: 0
Glucose Concentration: 95
Blood Pressure: 70
Skin Thickness: 30
Insulin: 85
BMI: 26.3
Diabetes Pedigree Function: 0.2
Age: 25
Prediction: The person is not diabetic.

DIABETIC TEST CASE:

Pregnancies: 3
Glucose Concentration: 150
Blood Pressure: 80
Skin Thickness: 40
Insulin: 130
BMI: 35.2
Diabetes Pedigree Function: 0.7
Age: 45
Prediction: The person is diabetic.

Contributing
We welcome contributions! If you'd like to contribute to the development of PredictaHealth, please fork the repository, create a new branch, make your changes, and submit a pull request.
Steps to Contribute:
Fork the repository
Clone your forked repository
Create a new branch
Make changes to the code
Commit your changes and push them to your fork
Submit a pull request to the original repository
