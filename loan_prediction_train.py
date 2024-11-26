import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # To save the trained model

# Sample data for training (you can replace this with actual data)
data = {
    'age': [25, 35, 45, 50, 60],
    'income': [30000, 50000, 70000, 100000, 120000],
    'loanAmount': [10000, 20000, 30000, 40000, 50000],
    'creditScore': [650, 700, 750, 800, 850],
    'employment': ['employed', 'self-employed', 'employed', 'unemployed', 'self-employed'],
    'loanApproval': [1, 1, 1, 0, 1]  # 1 = Approved, 0 = Not Approved
}

# Convert the categorical 'employment' feature into numeric values
employment_map = {'employed': 1, 'self-employed': 2, 'unemployed': 0}
employment_numeric = [employment_map[x] for x in data['employment']]

# Prepare the features and target variable
X = np.array([data['age'], data['income'], data['loanAmount'], data['creditScore'], employment_numeric]).T
y = np.array(data['loanApproval'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file (model.pkl)
joblib.dump(model, 'loan_prediction_model.pkl')

# Optionally, save the scaler to transform user input data in the same way as during training
joblib.dump(scaler, 'scaler.pkl')

# Print the accuracy of the model on the test set
accuracy = model.score(X_test, y_test)
print(f'Model accuracy on test set: {accuracy * 100:.2f}%')
