# Importing the necessary libraries
import numpy as np  # For numerical operations (arrays and mathematical functions)
import pandas as pd  # For handling and processing data in the form of DataFrames
from sklearn.preprocessing import StandardScaler  # For feature scaling (standardizing the dataset)
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn import svm  # Importing Support Vector Machine for classification
from sklearn.metrics import accuracy_score  # To evaluate the accuracy of the model
import joblib  # For saving the trained model to a file

# Reading the dataset into a DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# Printing the first 5 rows of the DataFrame to inspect the data
diabetes_dataset.head()

# Output the number of rows and columns in the dataset (shape)
diabetes_dataset.shape

# Getting the statistical summary of the dataset (mean, std, min, max, etc.)
diabetes_dataset.describe()

# Counting how many cases are there for Diabetes (Outcome = 1) and Non-Diabetes (Outcome = 0)
diabetes_dataset['Outcome'].value_counts()

# Grouping by Outcome and calculating the mean for each feature in each class (diabetic vs non-diabetic)
diabetes_dataset.groupby('Outcome').mean()

# Separating the features (X) and the labels (Y)
X = diabetes_dataset.drop(columns='Outcome', axis=1)  # All columns except 'Outcome' are features
Y = diabetes_dataset['Outcome']  # 'Outcome' column is the label (diabetic or not)


# Initializing the StandardScaler for feature scaling
scaler = StandardScaler()

# Fitting the scaler on the feature data (X)
scaler.fit(X)

# Applying the scaling to the features, transforming the original data into standardized data
standardized_data = scaler.transform(X)

# Printing the standardized feature data
print(standardized_data)

# Reassigning the scaled data back to X and keeping the labels (Y) intact
X = standardized_data
Y = diabetes_dataset['Outcome']


# Splitting the dataset into training and testing sets (80% train, 20% test)
# The stratify parameter ensures the class distribution (diabetic vs non-diabetic) is preserved in both train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Printing the shapes of the dataset splits to confirm the split sizes
print(X.shape, X_train.shape, X_test.shape)

# Initializing the Support Vector Machine classifier with a linear kernel
classifier = svm.SVC(kernel='linear')

# Training the classifier on the training data (X_train, Y_train)
classifier.fit(X_train, Y_train)

# Making predictions on the training data
X_train_prediction = classifier.predict(X_train)

# Calculating the accuracy of the model on the training data
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy Score of the Training Data : ', training_data_accuracy)

# Making predictions on the test data
X_test_prediction = classifier.predict(X_test)

# Calculating the accuracy of the model on the test data
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy Score of the Test Data : ', test_data_accuracy)

# Example input data to predict (for a single person)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Converting the input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the input data to be a 2D array (as expected by the model for one instance)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardizing the input data (using the same scaler that was fitted on the training data)
std_data = scaler.transform(input_data_reshaped)

# Printing the standardized input data
print(std_data)

# Making the prediction using the trained classifier
prediction = classifier.predict(std_data)

# Printing the prediction result
print(prediction)

# Outputting the result based on the prediction
if (prediction[0] == 0):  # If the prediction is 0, the person is not diabetic
    print('THE PERSON IS NOT DIABETIC')
else:  # If the prediction is 1, the person is diabetic
    print('THE PERSON IS DIABETIC')

# Saving the trained model to a file (diabetes_model.pkl) using joblib
joblib.dump(classifier, 'diabetes_model.pkl')  # Save the model for future use or deployment
