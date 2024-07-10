import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['loan_approval']

# Collection for raw CSV data
csv_collection = db['csv_data']

# Collection for machine learning results
result_collection = db['ml_results']

# Load the dataset from the 'csv_data' collection
data = pd.DataFrame(list(csv_collection.find()))

# Data preprocessing
# Selecting features and target variable based on the provided columns
X = data[['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount',
          'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value',
          'luxury_assets_value', 'bank_asset_value']]
y = data['loan_status']  # Target variable

# Encoding categorical variables
X = pd.get_dummies(X, columns=['education', 'self_employed'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Store the results and related data in the 'ml_results' collection
result_data = {
    'accuracy': accuracy,
    'X_train': X_train.to_dict(orient='records'),
    'X_test': X_test.to_dict(orient='records'),
    'y_train': y_train.tolist(),
    'y_test': y_test.tolist(),
}
result_collection.insert_one(result_data)

# Function to predict loan status for a new applicant
def predict_loan_status():
    # Collect user input
    new_data = {
        'no_of_dependents': int(input('Enter number of dependents: ')),
        'education': input('Enter education (Graduate/Not Graduate): '),
        'self_employed': input('Enter self employment status (Yes/No): '),
        'income_annum': float(input('Enter annual income: ')),
        'loan_amount': float(input('Enter loan amount: ')),
        'loan_term': int(input('Enter loan term in years: ')),
        'cibil_score': float(input('Enter CIBIL score: ')),
        'residential_assets_value': float(input('Enter value of residential assets: ')),
        'commercial_assets_value': float(input('Enter value of commercial assets: ')),
        'luxury_assets_value': float(input('Enter value of luxury assets: ')),
        'bank_asset_value': float(input('Enter value of bank assets: '))
    }

    # Convert new data to DataFrame
    new_data_df = pd.DataFrame([new_data])

    # Encode categorical variables in new data
    new_data_df = pd.get_dummies(new_data_df, columns=['education', 'self_employed'], drop_first=True)

    # Align new data columns with the training data columns
    missing_cols = set(X.columns) - set(new_data_df.columns)
    for col in missing_cols:
        new_data_df[col] = 0
    new_data_df = new_data_df[X.columns]

    # Predict loan status
    prediction = model.predict(new_data_df)
    return prediction[0]

# Example usage of the predict_loan_status function
predicted_status = predict_loan_status()
print(f'Predicted Loan Status: {predicted_status}')

# Close MongoDB connection
client.close()
