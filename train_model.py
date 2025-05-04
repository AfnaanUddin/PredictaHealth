# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting dataset into training and test sets
from sklearn.preprocessing import StandardScaler  # For feature scaling (standardization)
from sklearn.ensemble import RandomForestClassifier  # For building a Random Forest classifier model
from sklearn.metrics import accuracy_score  # To measure model accuracy
import joblib  # For saving the model and scaler

# Load the dataset (ensure the path to the 'diabetes.csv' file is correct)
df = pd.read_csv('diabetes.csv')

# Separate features (X) and target variable (y)
X = df.drop(columns=['Outcome'])  # All columns except 'Outcome' are features
y = df['Outcome']  # The 'Outcome' column is the target variable (whether diabetic or not)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scaling them to have zero mean and unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test = scaler.transform(X_test)  # Transform the test data using the same scaler (without fitting)

# Train the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)  # Train the model on the training data

# Save the trained model and the scaler to disk for later use
joblib.dump(model, 'diabetes_model.pkl')  # Saving the trained model
joblib.dump(scaler, 'scaler.pkl')  # Saving the scaler to standardize input data in the future

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
